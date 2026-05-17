"""
modules/preprocessing/quality_gate.py
--------------------------------------
CropQualityGate — three cheap checks that reject garbage crops BEFORE they
ever reach the embedder. Implements Part 2 of the Enhancement Proposal
("Garbage In, Garbage Out").

Checks performed:
    1. Minimum crop size (too far / too small → no discriminative power)
    2. Blur score          (motion blur / out-of-focus → corrupted features)
    3. Darkness score      (silhouette → embedder encodes shadow, not identity)

A crop must pass ALL three to be considered usable. The gate is stateless;
thresholds come from config.settings and can be tuned without code changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np

from config.settings import (
    CROP_BLUR_THRESHOLD,
    CROP_MIN_WIDTH,
    CROP_MIN_HEIGHT,
    CROP_DARKNESS_THRESHOLD,
)


@dataclass(frozen=True)
class QualityReport:
    """Outcome of a single crop assessment."""
    passes: bool
    reasons: Tuple[str, ...]      # human-readable rejection reasons (empty if passes)
    blur_score: float             # Laplacian variance (higher = sharper)
    brightness: float             # mean V channel (0–255)
    width: int
    height: int

    def summary(self) -> str:
        """One-line label for log lines."""
        if self.passes:
            return f"OK (blur={self.blur_score:.0f}, V={self.brightness:.0f}, {self.width}x{self.height})"
        return "REJECT — " + "; ".join(self.reasons)


class CropQualityGate:
    """
    Stateless quality gate. Thresholds are read once at construction so they
    can be overridden in tests without monkey-patching settings.
    """

    def __init__(
        self,
        blur_threshold: float = CROP_BLUR_THRESHOLD,
        min_width: int = CROP_MIN_WIDTH,
        min_height: int = CROP_MIN_HEIGHT,
        darkness_threshold: float = CROP_DARKNESS_THRESHOLD,
    ) -> None:
        self.blur_threshold = float(blur_threshold)
        self.min_width = int(min_width)
        self.min_height = int(min_height)
        self.darkness_threshold = float(darkness_threshold)

    # ------------------------------------------------------------------
    # Individual checks (kept independent so callers can sample one if
    # they only care about, say, darkness for an adaptive pipeline).
    # ------------------------------------------------------------------

    @staticmethod
    def blur_score(crop: np.ndarray) -> float:
        """Laplacian variance — higher = sharper. Returns 0.0 for empty crops."""
        if crop is None or crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def mean_brightness(crop: np.ndarray) -> float:
        """Mean of the HSV V channel — 0 (black) … 255 (white)."""
        if crop is None or crop.size == 0:
            return 0.0
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    # ------------------------------------------------------------------
    # Composite assessment
    # ------------------------------------------------------------------

    def assess(self, crop: np.ndarray) -> QualityReport:
        """
        Run all checks against ``crop`` and return a QualityReport.

        The report always contains the measured stats even when the crop
        fails, which makes telemetry/logs more useful than a bare bool.
        """
        if crop is None or crop.size == 0:
            return QualityReport(
                passes=False,
                reasons=("empty crop",),
                blur_score=0.0,
                brightness=0.0,
                width=0,
                height=0,
            )

        h, w = crop.shape[:2]
        reasons: List[str] = []

        if w < self.min_width or h < self.min_height:
            reasons.append(f"too small ({w}x{h} < {self.min_width}x{self.min_height})")

        blur = self.blur_score(crop)
        if blur < self.blur_threshold:
            reasons.append(f"blurry (lap_var={blur:.0f} < {self.blur_threshold:.0f})")

        brightness = self.mean_brightness(crop)
        if brightness < self.darkness_threshold:
            reasons.append(f"too dark (V_mean={brightness:.0f} < {self.darkness_threshold:.0f})")

        return QualityReport(
            passes=not reasons,
            reasons=tuple(reasons),
            blur_score=blur,
            brightness=brightness,
            width=w,
            height=h,
        )

    def passes(self, crop: np.ndarray) -> bool:
        """Convenience boolean — use ``assess()`` if you need the reasons."""
        return self.assess(crop).passes
