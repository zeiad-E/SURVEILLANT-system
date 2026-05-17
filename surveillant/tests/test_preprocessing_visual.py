"""
tests/test_preprocessing_visual.py
-----------------------------------
Visual smoke-test for the Phase-3.5 preprocessing pipeline (Parts 1-4).

Runs all stages against a single frame and writes a side-by-side comparison
PNG to ``data/preprocessing_diagnostic.png``. Prints quality-gate verdicts
to stdout for each detected person.

Usage (from repo root):
    python surveillant/tests/test_preprocessing_visual.py
        --video data/videos/video1_1.avi
        --frame 100

You should see, top to bottom:
    1. Raw frame
    2. CLAHE + auto-gamma enhanced frame
    3. YOLO-seg overlay (mask drawn in green)
    4. Per-person crops: RAW | MASKED | quality verdict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Make ``surveillant`` package importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from modules.detection.detector import PersonDetector
from modules.preprocessing.enhancement import FrameEnhancer
from modules.preprocessing.quality_gate import CropQualityGate
from modules.preprocessing.masking import apply_mask_to_crop
from config.settings import YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ


def grab_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] could not open {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total:
        print(f"[WARN] frame {frame_idx} > total {total}, clamping")
        frame_idx = max(0, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"[ERROR] could not read frame {frame_idx}")
    return frame


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/videos/video1_1.avi")
    parser.add_argument("--frame", type=int, default=100)
    parser.add_argument("--out",   default="data/preprocessing_diagnostic.png")
    args = parser.parse_args()

    video_path = (REPO_ROOT.parent / args.video).resolve()
    if not video_path.exists():
        # Try repo-relative
        video_path = (REPO_ROOT / args.video).resolve()
    print(f"[TEST] video : {video_path}")
    print(f"[TEST] frame : {args.frame}")
    print(f"[TEST] model : {YOLO_MODEL}")

    raw = grab_frame(video_path, args.frame)
    print(f"[TEST] frame shape: {raw.shape}")

    # ── Stage 1: enhancement ────────────────────────────────────────────────
    enhancer = FrameEnhancer()
    enhanced = enhancer.enhance(raw)

    # ── Stage 2: detection (uses the SAME enhancer internally — re-running it
    #             on the raw frame here just gives us a side-by-side view).
    detector = PersonDetector(YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ)
    detections = detector.detect(raw)
    print(f"[TEST] detections found: {len(detections)}")
    print(f"[TEST] seg model in use: {detector.is_segmentation}")

    # ── Stage 3: draw overlay (mask in green, bbox in yellow) ───────────────
    overlay = raw.copy()
    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 255), 2)
        m = d.get("mask")
        if m is not None:
            green = np.zeros_like(overlay)
            green[..., 1] = 255
            mask3 = (m > 0).astype(np.uint8)[..., None]
            overlay = np.where(mask3, cv2.addWeighted(overlay, 0.5, green, 0.5, 0), overlay)

    # ── Stage 4: per-detection crop strip with gate verdicts ────────────────
    gate = CropQualityGate()
    panels = []
    for i, d in enumerate(detections[:6]):
        x1, y1, x2, y2 = d["bbox"]
        crop = raw[max(0, y1):y2, max(0, x1):x2]
        if crop.size == 0:
            continue

        mask_full = d.get("mask")
        if mask_full is not None:
            mh, mw = mask_full.shape
            sub = mask_full[max(0, y1):min(mh, y2), max(0, x1):min(mw, x2)]
        else:
            sub = None

        masked = apply_mask_to_crop(crop, sub)

        q = gate.assess(crop)
        print(f"[TEST] det#{i}: {q.summary()}")

        target_h = 240
        r = target_h / crop.shape[0]
        w = int(crop.shape[1] * r)
        crop_v = cv2.resize(crop,   (w, target_h))
        masked_v = cv2.resize(masked, (w, target_h))

        crop_v = label(crop_v,   f"#{i} RAW")
        verdict = "PASS" if q.passes else "FAIL"
        masked_v = label(masked_v, f"#{i} MASKED [{verdict}]")

        gap = np.full((target_h, 6, 3), 80, dtype=np.uint8)
        panels.append(np.hstack([crop_v, gap, masked_v]))

    # ── Compose final diagnostic image ──────────────────────────────────────
    big_w = max(raw.shape[1], 800)
    raw_show      = cv2.resize(label(raw,      "1. RAW FRAME"),                 (big_w, int(big_w * raw.shape[0] / raw.shape[1])))
    enhanced_show = cv2.resize(label(enhanced, "2. CLAHE + AUTO-GAMMA"),         (big_w, int(big_w * raw.shape[0] / raw.shape[1])))
    overlay_show  = cv2.resize(label(overlay,  f"3. YOLO-SEG  ({len(detections)} detections)"), (big_w, int(big_w * raw.shape[0] / raw.shape[1])))

    rows = [raw_show, enhanced_show, overlay_show]
    if panels:
        # Pad crop panels to the same width for vstack
        max_w = max(p.shape[1] for p in panels)
        padded = []
        for p in panels:
            if p.shape[1] < max_w:
                pad = np.full((p.shape[0], max_w - p.shape[1], 3), 30, dtype=np.uint8)
                p = np.hstack([p, pad])
            padded.append(p)
        crop_strip = np.vstack(padded)
        if crop_strip.shape[1] != big_w:
            crop_strip = cv2.resize(crop_strip, (big_w, int(big_w * crop_strip.shape[0] / crop_strip.shape[1])))
        crop_strip = label(crop_strip, "4. PER-PERSON: RAW | MASKED  +  QUALITY GATE VERDICT")
        rows.append(crop_strip)

    final = np.vstack(rows)
    out_path = (REPO_ROOT.parent / args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), final)
    print(f"[TEST] diagnostic image written: {out_path}")
    print(f"[TEST] image size: {final.shape}")


if __name__ == "__main__":
    main()
