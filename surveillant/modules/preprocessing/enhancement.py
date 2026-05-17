"""
modules/preprocessing/enhancement.py
-------------------------------------
FrameEnhancer — low-light / adverse-lighting preprocessing applied to every
frame BEFORE YOLO detection. Implements Part 3 of the Enhancement Proposal.

Two complementary stages:

    1. CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB space.
       Only the luminance (L) channel is equalized so colors stay faithful —
       critical because the embedder relies on appearance features.

    2. Auto-gamma — kicks in only when the frame's mean V channel falls below
       AUTO_GAMMA_THRESHOLD. Gamma < 1.0 brightens dark regions aggressively
       (LUT-based, ~0.1 ms per frame on CPU).

Note on infrared / VI-ReID (Part 3.3 of the proposal): the project has no
infrared camera, so cross-modality preprocessing is intentionally out of scope.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from config.settings import (
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    AUTO_GAMMA_THRESHOLD,
    AUTO_GAMMA_VALUE,
    ENABLE_FRAME_ENHANCEMENT,
)


class FrameEnhancer:
    """
    Stateless-ish frame enhancer. The CLAHE object and gamma LUT are cached
    on the instance so we don't rebuild them per frame.
    """

    def __init__(
        self,
        clip_limit: float = CLAHE_CLIP_LIMIT,
        tile_grid_size: Tuple[int, int] = CLAHE_TILE_GRID_SIZE,
        gamma_threshold: float = AUTO_GAMMA_THRESHOLD,
        gamma_value: float = AUTO_GAMMA_VALUE,
        enabled: bool = ENABLE_FRAME_ENHANCEMENT,
    ) -> None:
        self.enabled = bool(enabled)
        self.gamma_threshold = float(gamma_threshold)
        self.gamma_value = float(gamma_value)

        self._clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit),
            tileGridSize=tuple(tile_grid_size),
        )

        # Pre-compute gamma LUT once. gamma < 1.0 brightens.
        inv_gamma = 1.0 / max(self.gamma_value, 1e-6)
        self._gamma_lut = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )

    # ------------------------------------------------------------------
    # Stages
    # ------------------------------------------------------------------

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Equalize the L channel of LAB. Colors (A, B) are preserved so the
        embedder's color-sensitive features remain trustworthy.
        """
        if frame is None or frame.size == 0:
            return frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l_eq = self._clahe.apply(l)
        merged = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def apply_auto_gamma(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply gamma correction only if the frame is darker than the threshold.
        Avoids unnecessary work on well-lit scenes.
        """
        if frame is None or frame.size == 0:
            return frame
        mean_v = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].mean())
        if mean_v < self.gamma_threshold:
            return cv2.LUT(frame, self._gamma_lut)
        return frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        Full pipeline: conditional CLAHE → conditional auto-gamma.

        Both stages are gated on the frame's mean brightness so we don't waste
        ~10 ms per frame on already-bright outdoor / well-lit indoor scenes.
        CLAHE only fires when V-mean is below CLAHE_BRIGHTNESS_GATE; auto-gamma
        fires only when V-mean is below AUTO_GAMMA_THRESHOLD (stricter).
        """
        if not self.enabled or frame is None or frame.size == 0:
            return frame

        # Single V-channel reading drives both stages — avoids two HSV converts.
        mean_v = float(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2].mean())

        out = frame
        # CLAHE_BRIGHTNESS_GATE: skip CLAHE on bright scenes (V > 120 is bright).
        if mean_v < 120.0:
            out = self.apply_clahe(out)

        if mean_v < self.gamma_threshold:
            out = cv2.LUT(out, self._gamma_lut)

        return out
