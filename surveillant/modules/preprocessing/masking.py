"""
modules/preprocessing/masking.py
---------------------------------
Helpers for applying YOLO segmentation masks to person crops, isolating the
foreground from the background so the embedder is not contaminated by wall
color, signage, or whatever happens to be behind the subject. Implements
Part 4 of the Enhancement Proposal.

Two key entry points:

    * ``apply_mask_to_crop(crop, mask)`` — replace background pixels with
      neutral gray (not black: black creates a strong dark feature that
      biases the embedder).

    * ``associate_masks_to_tracks(tracks, detections)`` — DeepSORT
      reorders detections; this function maps each track back to its
      originating detection by IoU so we can recover the right mask.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import cv2
import numpy as np

from config.settings import (
    BACKGROUND_REPLACEMENT_COLOR,
    SEGMENTATION_MASK_THRESHOLD,
    SEGMENTATION_TRACK_IOU,
)


# ---------------------------------------------------------------------------
# Mask application
# ---------------------------------------------------------------------------

def apply_mask_to_crop(
    crop: np.ndarray,
    mask: Optional[np.ndarray],
    neutral_color: int = BACKGROUND_REPLACEMENT_COLOR,
) -> np.ndarray:
    """
    Zero out background pixels in ``crop`` according to ``mask``.

    Background pixels are replaced with a flat neutral gray (default 128).
    This keeps the embedder from inventing "dark-background" features that
    would otherwise bias same-person/different-background comparisons.

    The mask is expected to be the same H,W as the crop; if it isn't, it
    is resized with nearest-neighbour interpolation to preserve hard edges.
    """
    if crop is None or crop.size == 0:
        return crop
    if mask is None or mask.size == 0:
        return crop

    ch, cw = crop.shape[:2]
    if mask.shape[:2] != (ch, cw):
        mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

    binary = (mask > SEGMENTATION_MASK_THRESHOLD).astype(bool)
    if not binary.any():
        # Nothing classified as foreground — keep original crop so we don't
        # blank the embedder's input entirely.
        return crop

    neutral = np.full_like(crop, int(neutral_color))
    # Broadcast (H,W) → (H,W,1) so np.where picks per-channel pixels.
    return np.where(binary[:, :, None], crop, neutral)


# ---------------------------------------------------------------------------
# Track ↔ detection mask association
# ---------------------------------------------------------------------------

def _iou(box_a: List[int], box_b: List[int]) -> float:
    """Standard IoU on [x1, y1, x2, y2] integer boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def associate_masks_to_tracks(
    tracks: List[Dict],
    detections: List[Dict],
    iou_threshold: float = SEGMENTATION_TRACK_IOU,
) -> Dict[int, np.ndarray]:
    """
    Build a {track_id → bbox-aligned mask} lookup.

    DeepSORT may reorder, merge, or extrapolate detections, so we cannot
    rely on index alignment. We match by IoU between each track's predicted
    bbox and the original detection bboxes (which still carry their masks).

    Tracks with no detection match this frame (e.g. coasting through
    occlusion) get no entry — the caller will simply fall back to the
    raw crop, which is the right behavior.
    """
    if not tracks or not detections:
        return {}

    out: Dict[int, np.ndarray] = {}

    # Pre-extract detection bboxes once.
    det_bboxes = [d.get("bbox") for d in detections]

    for t in tracks:
        tid = t.get("track_id")
        tbox = t.get("bbox")
        if tid is None or tbox is None:
            continue

        best_iou = 0.0
        best_idx: Optional[int] = None
        for i, dbox in enumerate(det_bboxes):
            if dbox is None:
                continue
            iou = _iou(tbox, dbox)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        if best_idx is None or best_iou < iou_threshold:
            continue

        mask = detections[best_idx].get("mask")
        if mask is None:
            continue

        # Crop the full-frame mask down to the track bbox so the caller can
        # apply it directly to the cropped image.
        tx1, ty1, tx2, ty2 = tbox
        tx1 = max(0, int(tx1)); ty1 = max(0, int(ty1))
        tx2 = max(tx1 + 1, int(tx2)); ty2 = max(ty1 + 1, int(ty2))
        mh, mw = mask.shape[:2]
        # Clip to mask bounds in case the track bbox extrapolates outside the frame.
        cx1, cy1 = min(tx1, mw - 1), min(ty1, mh - 1)
        cx2, cy2 = min(tx2, mw),     min(ty2, mh)
        sub = mask[cy1:cy2, cx1:cx2]
        if sub.size == 0:
            continue
        out[tid] = sub

    return out
