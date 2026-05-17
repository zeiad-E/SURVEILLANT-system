"""
modules/embedding/gallery.py
-----------------------------
GalleryManager — decides when a new embedding is novel enough to store,
drives pose-aware canonical-view coverage, and provides view-coverage scoring.

Phase 3.5 additions (Parts 2 & 6 of the Enhancement Proposal):

    Part 2 — CropQualityGate:
        maybe_update_gallery() pre-validates every crop before spending an
        embedding-extraction call on it.

    Part 6 — Pose-Aware Gallery:
        estimate_view() classifies a bounding box into one of four canonical
        viewpoints: frontal, right_moving, left_moving, side.

        When a canonical view slot has not yet been covered for a person, the
        new embedding is *force-accepted* regardless of cosine distance (subject
        only to MAX_GALLERY_SIZE and GALLERY_MAX_DISTANCE garbage checks). This
        ensures every person's gallery eventually represents all observable angles
        rather than accumulating near-duplicate frontal views.

        get_view_coverage() returns a 0.0–1.0 score used by the searcher to
        skip insufficiently-observed persons from cross-camera matching.
"""

import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple

from config.settings import (
    MAX_GALLERY_SIZE,
    FACE_GALLERY_ADD_DISTANCE,
    BODY_GALLERY_ADD_DISTANCE,
    GALLERY_MAX_DISTANCE,
    MIN_FRAMES_BETWEEN_SAMPLES,
    CANONICAL_VIEWS,
)
from modules.preprocessing.quality_gate import CropQualityGate


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def estimate_view(
    bbox: List[int],
    prev_bbox: Optional[List[int]] = None,
) -> str:
    """
    Classify a bounding box into one of four canonical Re-ID viewpoints.

    Logic (from the Enhancement Proposal §6.1):
        1. Narrow aspect ratio (width/height < 0.40) → person is sideways.
        2. Horizontal displacement from prev frame → direction of movement.
        3. Otherwise → frontal (facing the camera or stationary).

    Args:
        bbox:      Current [x1, y1, x2, y2] bounding box.
        prev_bbox: Previous [x1, y1, x2, y2] or None.

    Returns:
        One of CANONICAL_VIEWS: "frontal", "right_moving", "left_moving", "side".
    """
    x1, y1, x2, y2 = bbox
    aspect = (x2 - x1) / max(y2 - y1, 1)
    if aspect < 0.40:
        return "side"
    if prev_bbox is not None:
        dx = (x1 + x2) / 2.0 - (prev_bbox[0] + prev_bbox[2]) / 2.0
        if dx > 8:
            return "right_moving"
        if dx < -8:
            return "left_moving"
    return "frontal"


# ---------------------------------------------------------------------------
# GalleryManager
# ---------------------------------------------------------------------------

class GalleryManager:
    """
    Stateless gallery decision engine.
    All persistent state lives in the SQLite person_embeddings table.

    The quality gate and pose-aware logic are both stateless; thresholds
    come from config so tuning requires no code changes.
    """

    def __init__(self) -> None:
        self._quality_gate = CropQualityGate()

    # ----------------------------------------------------------------
    # Core should_add decision (diversity-based)
    # ----------------------------------------------------------------

    def should_add_to_gallery(
        self,
        new_embedding: np.ndarray,
        new_type: str,
        existing_gallery: List[Dict[str, Any]],
    ) -> bool:
        """
        Returns True if new_embedding is novel enough to store (diversity gate).

        This is the *fallback* path used when the canonical-view slot is already
        covered. For an uncovered slot the caller force-accepts regardless.
        """
        if not existing_gallery:
            return True
        if len(existing_gallery) >= MAX_GALLERY_SIZE:
            return False

        existing_vecs  = [e["embedding"] for e in existing_gallery
                          if e["embedding"].shape[0] == new_embedding.shape[0]]
        existing_types = [e["type"] for e in existing_gallery
                          if e["embedding"].shape[0] == new_embedding.shape[0]]

        if not existing_vecs:
            return True

        face_count = sum(1 for t in existing_types if t == "face")
        if (new_type == "body"
                and face_count >= 2
                and len(existing_gallery) >= int(MAX_GALLERY_SIZE * 0.75)):
            return False

        query_2d      = new_embedding.reshape(1, -1)
        gallery_array = np.array(existing_vecs)
        similarities  = cosine_similarity(query_2d, gallery_array)[0]
        max_sim       = float(np.max(similarities))
        distance      = 1.0 - max_sim

        if distance > GALLERY_MAX_DISTANCE:
            return False

        novelty_thresh = (
            FACE_GALLERY_ADD_DISTANCE if new_type == "face"
            else BODY_GALLERY_ADD_DISTANCE
        )
        return distance > novelty_thresh

    # ----------------------------------------------------------------
    # View coverage (Part 6)
    # ----------------------------------------------------------------

    def get_view_coverage(
        self,
        existing_gallery: List[Dict[str, Any]],
    ) -> float:
        """
        Fraction of canonical view slots covered (0.0 – 1.0).

        A coverage of 0.5 means at least 2 of the 4 canonical views are
        present — the minimum for reliable cross-camera matching.
        """
        covered = {e.get("angle_tag") for e in existing_gallery} & set(CANONICAL_VIEWS)
        return len(covered) / len(CANONICAL_VIEWS)

    # ----------------------------------------------------------------
    # Angle tag
    # ----------------------------------------------------------------

    def get_angle_tag(
        self,
        new_embedding: np.ndarray,
        existing_gallery: List[Dict[str, Any]],
        source_cam: Optional[int] = None,
        person_first_cam: Optional[int] = None,
        canonical_view: Optional[str] = None,
    ) -> str:
        """
        Return the angle tag to store with this embedding.

        Priority:
          canonical_view provided → use it directly (Part 6 pose-aware path).
          initial                 → first embedding ever.
          cross_cam_view          → source_cam differs from person_first_cam.
          very_different          → cosine distance > 0.5 (likely back view).
          same_cam_new_angle      → different angle, same camera.
          partial_view            → small difference.
        """
        if canonical_view is not None and canonical_view in CANONICAL_VIEWS:
            return canonical_view

        if not existing_gallery:
            return "initial"

        if (source_cam is not None
                and person_first_cam is not None
                and source_cam != person_first_cam):
            return "cross_cam_view"

        existing_vecs = [e["embedding"] for e in existing_gallery
                         if e["embedding"].shape[0] == new_embedding.shape[0]]
        if not existing_vecs:
            return "initial"

        query_2d      = new_embedding.reshape(1, -1)
        gallery_array = np.array(existing_vecs)
        similarities  = cosine_similarity(query_2d, gallery_array)[0]
        distance      = 1.0 - float(np.max(similarities))

        if distance > 0.5:
            return "very_different"
        elif distance > BODY_GALLERY_ADD_DISTANCE:
            return "same_cam_new_angle"
        else:
            return "partial_view"

    # ----------------------------------------------------------------
    # Gallery update orchestration
    # ----------------------------------------------------------------

    def maybe_update_gallery(
        self,
        person_id: str,
        crop: np.ndarray,
        embedder,
        db,
        frame_count: int,
        cam_id: int = 0,
        bbox: Optional[List[int]] = None,
        prev_bbox: Optional[List[int]] = None,
    ) -> bool:
        """
        Called on every frame for a bound track (rate-limited).

        Part 2: Quality gate rejects blurry / dark / tiny crops.
        Part 6: Canonical view is estimated from the bounding box; if the slot
                is uncovered the embedding is force-accepted (bypassing the
                diversity novelty gate) to ensure angle diversity.

        Returns True if a new embedding was added to the gallery.
        """
        if frame_count % MIN_FRAMES_BETWEEN_SAMPLES != 0:
            return False
        if crop is None or crop.size == 0:
            return False

        gallery_size = db.get_gallery_size(person_id)
        if gallery_size >= MAX_GALLERY_SIZE:
            return False

        # Part 2 — quality gate on the raw crop
        quality = self._quality_gate.assess(crop)
        if not quality.passes:
            print(
                f"[GALLERY] person {person_id[:8]} | crop REJECTED at quality gate — "
                f"{quality.summary()}"
            )
            return False

        new_emb  = embedder.extract_body_embedding(crop)
        new_type = "body"

        existing = db.get_gallery_typed(person_id)

        # Part 6 — pose-aware canonical view check
        canonical_view: Optional[str] = None
        force_accept   = False

        if bbox is not None:
            canonical_view = estimate_view(bbox, prev_bbox)
            existing_tags  = {e.get("angle_tag") for e in existing}
            if canonical_view in CANONICAL_VIEWS and canonical_view not in existing_tags:
                # This canonical slot is empty — force-accept to fill it.
                # Still enforce the garbage check (GALLERY_MAX_DISTANCE) and size cap.
                compat_vecs = [e["embedding"] for e in existing
                               if e["embedding"].shape[0] == new_emb.shape[0]]
                if compat_vecs:
                    query_2d  = new_emb.reshape(1, -1)
                    sims      = cosine_similarity(query_2d, np.array(compat_vecs))[0]
                    distance  = 1.0 - float(np.max(sims))
                    if distance <= GALLERY_MAX_DISTANCE:
                        force_accept = True
                    else:
                        print(
                            f"[GALLERY] person {person_id[:8]} | {canonical_view} slot "
                            f"REJECTED — garbage crop (dist={distance:.2f} > {GALLERY_MAX_DISTANCE})"
                        )
                        return False
                else:
                    force_accept = True   # first embedding — always accept

        # Standard diversity gate (used when canonical slot is already covered)
        if not force_accept and not self.should_add_to_gallery(new_emb, new_type, existing):
            if existing:
                compat = [e["embedding"] for e in existing
                          if e["embedding"].shape[0] == new_emb.shape[0]]
                if compat:
                    query_2d = new_emb.reshape(1, -1)
                    sims     = cosine_similarity(query_2d, np.array(compat))[0]
                    dist     = 1.0 - float(np.max(sims))
                    if dist > GALLERY_MAX_DISTANCE:
                        reason = f"too dissimilar/bad crop (dist={dist:.2f})"
                    elif gallery_size >= MAX_GALLERY_SIZE:
                        reason = "gallery full"
                    else:
                        reason = f"duplicate angle (dist={dist:.2f} < {BODY_GALLERY_ADD_DISTANCE})"
                    print(f"[GALLERY] person {person_id[:8]} | {new_type} REJECTED — {reason}")
            return False

        person_data = db.get_person(person_id)
        first_cam   = person_data.get("first_seen_cam") if person_data else None
        angle_tag   = self.get_angle_tag(
            new_emb, existing, cam_id, first_cam, canonical_view
        )

        now_str = datetime.datetime.now().isoformat()
        db.add_embedding_to_gallery(
            person_id       = person_id,
            embedding_bytes = embedder.serialize(new_emb),
            embedding_type  = new_type,
            angle_tag       = angle_tag,
            source_cam      = cam_id,
            captured_at     = now_str,
        )
        slot_label = f"[force:{canonical_view}]" if force_accept else ""
        print(
            f"[GALLERY] person {person_id[:8]} | {new_type} view added "
            f"angle={angle_tag} {slot_label}(gallery: {gallery_size}->{gallery_size+1})"
        )
        return True
