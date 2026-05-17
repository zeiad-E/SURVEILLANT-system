"""
modules/tracking/tracker.py
----------------------------
PersonTracker wraps ByteTrack (ultralytics built-in) for per-camera tracking.

Part 7 of the Enhancement Proposal — replacing DeepSORT with ByteTrack.

Key improvement: ByteTrack's two-stage association uses BOTH high-confidence
and low-confidence YOLO detections:

    Stage 1 (high-conf ≥ BYTETRACK_TRACK_THRESH):
        Standard IoU matching against all active tracks.

    Stage 2 (low-conf ≥ BYTETRACK_LOW_THRESH):
        Matches *only* against tracks that are already lost/coasting from
        Stage 1. This keeps occluded-person tracks alive through brief gaps
        when YOLO isn't confident (person partly hidden, turning, far away).

Result: far fewer track deaths during fast movement and occlusions compared
to DeepSORT, which discards detections below its single conf threshold entirely.

Public interface is identical to the old DeepSORT wrapper so main.py is
unchanged. reinforce_track() and get_appearance_hint() are preserved for API
compatibility; ByteTrack does not use an appearance model (pure IoU + Kalman).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from config.settings import (
    BYTETRACK_TRACK_THRESH,
    BYTETRACK_LOW_THRESH,
    BYTETRACK_MATCH_THRESH,
    BYTETRACK_TRACK_BUFFER,
    TRACKING_COAST_FRAMES,
    FPS_TARGET,
)


# ---------------------------------------------------------------------------
# Internal helper — wraps plain detection dicts in the object ByteTrack expects
# ---------------------------------------------------------------------------

class _DetResults:
    """
    Lightweight shim that exposes the .xywh, .conf, .cls attributes that
    BYTETracker.update() and BYTETracker.init_track() read.

    Supports numpy boolean indexing so ByteTrack can split detections into
    high- and low-confidence groups internally.
    """

    def __init__(
        self,
        bboxes_xyxy: np.ndarray,   # (N, 4) float32: x1, y1, x2, y2
        confs:        np.ndarray,   # (N,)   float32
    ) -> None:
        n = len(bboxes_xyxy)
        if n > 0:
            x1, y1, x2, y2 = bboxes_xyxy[:, 0], bboxes_xyxy[:, 1], bboxes_xyxy[:, 2], bboxes_xyxy[:, 3]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w  = x2 - x1
            h  = y2 - y1
            self.xywh = np.stack([cx, cy, w, h], axis=1).astype(np.float32)
        else:
            self.xywh = np.empty((0, 4), dtype=np.float32)
        self.conf = confs.astype(np.float32)
        self.cls  = np.zeros(len(confs), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx: Any) -> "_DetResults":
        sub = _DetResults.__new__(_DetResults)
        sub.xywh = self.xywh[idx]
        sub.conf = self.conf[idx]
        sub.cls  = self.cls[idx]
        return sub


# ---------------------------------------------------------------------------
# PersonTracker
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    Per-camera person tracker backed by ByteTrack.

    Public API (unchanged from DeepSORT wrapper):
        update(detections, frame) → List[Dict]
        reinforce_track(track_id, person_id, gallery_embeddings)
        get_appearance_hint(track_id)  → Optional[np.ndarray]
        get_person_id(track_id)        → Optional[str]
    """

    def __init__(self, cam_id: int) -> None:
        self.cam_id = cam_id

        args = SimpleNamespace(
            track_high_thresh = BYTETRACK_TRACK_THRESH,
            track_low_thresh  = BYTETRACK_LOW_THRESH,
            new_track_thresh  = BYTETRACK_TRACK_THRESH,
            match_thresh      = BYTETRACK_MATCH_THRESH,
            track_buffer      = BYTETRACK_TRACK_BUFFER,
            fuse_score        = False,
        )
        self._tracker = BYTETracker(args, frame_rate=FPS_TARGET)

        # Gallery hints kept for API compatibility (not used by ByteTrack itself)
        self._gallery_hints:   Dict[int, List[np.ndarray]] = {}
        self._track_to_person: Dict[int, str]              = {}

        print(f"[PersonTracker] ByteTrack initialized for camera {cam_id}.")

    def __repr__(self) -> str:
        return f"PersonTracker(cam_id={self.cam_id}, backend=ByteTrack)"

    # ------------------------------------------------------------------
    # Public API — tracking
    # ------------------------------------------------------------------

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Update ByteTrack with the latest detections.

        All detections (high- and low-confidence) are passed; ByteTrack splits
        them internally by BYTETRACK_TRACK_THRESH / BYTETRACK_LOW_THRESH.

        Returns confirmed, active tracks as:
            [{'track_id': int, 'bbox': [x1,y1,x2,y2], 'cam_id': int}, ...]
        """
        if detections:
            bboxes = np.array([d["bbox"] for d in detections], dtype=np.float32)
            confs  = np.array([d["confidence"] for d in detections], dtype=np.float32)
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
            confs  = np.empty((0,),   dtype=np.float32)

        results = _DetResults(bboxes, confs)

        # BYTETracker.update() returns (N, 8) array:
        # [x1, y1, x2, y2, track_id, score, cls, idx]
        raw = self._tracker.update(results, img=frame)

        confirmed: List[Dict[str, Any]] = []
        for row in raw:
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], int(row[4])

            # Find the STrack object to read time_since_update for coast filter
            strack = next(
                (t for t in self._tracker.tracked_stracks if t.track_id == tid),
                None,
            )
            if strack is not None:
                tsu = getattr(strack, "time_since_update", 0)
                if tsu > TRACKING_COAST_FRAMES:
                    continue

            confirmed.append({
                "track_id": tid,
                "bbox":     [int(x1), int(y1), int(x2), int(y2)],
                "cam_id":   self.cam_id,
            })

        return self._deduplicate(confirmed)

    # ------------------------------------------------------------------
    # Public API — gallery hint feedback (API-compatible with old wrapper)
    # ------------------------------------------------------------------

    def reinforce_track(
        self,
        track_id: int,
        person_id: str,
        gallery_embeddings: List[np.ndarray],
    ) -> None:
        """Store gallery embeddings as appearance hints (for future use)."""
        self._gallery_hints[track_id]   = gallery_embeddings
        self._track_to_person[track_id] = person_id
        print(
            f"[REINFORCE] cam{self.cam_id}_track{track_id} -> person {person_id[:8]} "
            f"({len(gallery_embeddings)}-view gallery stored)"
        )

    def get_appearance_hint(self, track_id: int) -> Optional[np.ndarray]:
        hints = self._gallery_hints.get(track_id)
        if hints:
            return np.mean(hints, axis=0)
        return None

    def get_person_id(self, track_id: int) -> Optional[str]:
        return self._track_to_person.get(track_id)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        tracks: List[Dict[str, Any]],
        iou_threshold: float = 0.50,
    ) -> List[Dict[str, Any]]:
        """Suppress heavily overlapping boxes — keeps the older (lower-id) track."""
        if len(tracks) <= 1:
            return tracks

        sorted_tracks = sorted(tracks, key=lambda t: t["track_id"])
        suppressed = set()

        for i in range(len(sorted_tracks)):
            if i in suppressed:
                continue
            for j in range(i + 1, len(sorted_tracks)):
                if j in suppressed:
                    continue
                if self._iou(sorted_tracks[i]["bbox"], sorted_tracks[j]["bbox"]) > iou_threshold:
                    suppressed.add(j)

        return [t for idx, t in enumerate(sorted_tracks) if idx not in suppressed]

    @staticmethod
    def _iou(box_a: list, box_b: list) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - inter
        return inter / union if union > 0 else 0.0
