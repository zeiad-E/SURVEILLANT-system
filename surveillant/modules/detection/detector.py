"""
modules/detection/detector.py
------------------------------
PersonDetector wraps YOLOv8 to detect only people (class 0) in a frame.

The YOLOv8 model is downloaded automatically on first use by `ultralytics`.

Enhancements (proposal Parts 3 & 4):
    * Every frame is run through ``FrameEnhancer`` (CLAHE + auto-gamma) before
      YOLO inference. This improves detection recall in dark / indoor scenes.
    * When the configured model is a segmentation variant (``-seg.pt``) the
      detector also emits a binary mask per detection, in the SAME pixel
      grid as the input frame. Downstream code uses these masks to wipe
      out the background before embedding (see modules/preprocessing/masking.py).
"""

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO

from config.settings import (
    USE_SEGMENTATION,
    SEGMENTATION_MASK_THRESHOLD,
)
from modules.preprocessing.enhancement import FrameEnhancer


class PersonDetector:
    """
    Detects people in a single video frame using YOLOv8.

    The model is loaded once at construction time and reused for every
    subsequent call to `detect()`.

    Args:
        model_name (str):  YOLOv8 model filename, e.g. ``"yolov8n-seg.pt"``.
        conf       (float): Minimum detection confidence (0–1).
        imgsz      (int):  Input image size passed to YOLO. Smaller = faster.
                           320 is ~3× faster than 640 for CPU inference.
        enhance    (bool|None): Whether to apply the FrameEnhancer to each
                                frame before YOLO. ``None`` honours the
                                ``ENABLE_FRAME_ENHANCEMENT`` setting.
    """

    PERSON_CLASS_ID = 0  # class 0 = person in the COCO dataset

    def __init__(
        self,
        model_name: str,
        conf: float,
        imgsz: int = 320,
        enhance: Optional[bool] = None,
    ) -> None:
        self.model_name: str   = model_name
        self.conf:       float = conf
        self.imgsz:      int   = imgsz

        # A "-seg" model exposes results.masks; a plain detector does not.
        # We respect both the model choice and the USE_SEGMENTATION toggle so
        # operators can disable masking without swapping weights.
        self.is_segmentation: bool = USE_SEGMENTATION and "-seg" in model_name

        print(f"[PersonDetector] Loading model: {model_name} (imgsz={imgsz}) …")
        self._model = YOLO(model_name)
        self._enhancer = FrameEnhancer() if (enhance is None or enhance) else None
        print(
            f"[PersonDetector] Model ready "
            f"(conf≥{conf}, imgsz={imgsz}, seg={self.is_segmentation}, "
            f"enhance={self._enhancer is not None})."
        )

    def __repr__(self) -> str:
        return (
            f"PersonDetector(model={self.model_name!r}, "
            f"conf={self.conf}, imgsz={self.imgsz}, "
            f"seg={self.is_segmentation})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Run person detection on a single frame.

        Args:
            frame (np.ndarray): BGR image as returned by OpenCV.

        Returns:
            List of dicts, each with keys:
                - ``bbox``       : ``[x1, y1, x2, y2]`` in pixel coordinates.
                - ``confidence`` : float between 0 and 1.
                - ``mask``       : full-frame binary uint8 mask (np.ndarray)
                                   OR ``None`` if the model is non-segmentation.

            Returns an empty list if no people are detected.
        """
        if frame is None or frame.size == 0:
            return []

        # Part 3 — enhance lighting BEFORE YOLO so detection itself benefits.
        # The enhancer no-ops cheaply when disabled or on a bright frame.
        input_frame = self._enhancer.enhance(frame) if self._enhancer is not None else frame

        results = self._model.predict(
            source=input_frame,
            conf=self.conf,
            classes=[self.PERSON_CLASS_ID],
            imgsz=self.imgsz,
            iou=0.40,    # aggressive NMS — prevents split detections (torso + full body) reaching the tracker
            verbose=False,
        )

        detections: List[Dict[str, Any]] = []
        fh, fw = frame.shape[:2]

        for result in results:
            boxes = result.boxes
            masks_attr = getattr(result, "masks", None) if self.is_segmentation else None

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                # Skip tiny boxes — likely false positives or body fragments
                if w < 20 or h < 40:
                    continue

                mask_full: Optional[np.ndarray] = None
                if masks_attr is not None and i < len(masks_attr.data):
                    raw = masks_attr.data[i]
                    # ultralytics returns masks as either a torch tensor or numpy
                    if hasattr(raw, "cpu"):
                        raw = raw.cpu().numpy()
                    # The mask is typically in the model's internal resolution;
                    # resize back to the original frame so downstream cropping
                    # can index it with bbox pixel coordinates directly.
                    if raw.shape[:2] != (fh, fw):
                        raw = cv2.resize(raw.astype(np.float32), (fw, fh))
                    mask_full = (raw > SEGMENTATION_MASK_THRESHOLD).astype(np.uint8)

                detections.append(
                    {
                        "bbox":       [int(x1), int(y1), int(x2), int(y2)],
                        "confidence": float(box.conf[0]),
                        "mask":       mask_full,
                    }
                )

        return detections
