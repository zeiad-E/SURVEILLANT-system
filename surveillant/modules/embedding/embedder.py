"""
modules/embedding/embedder.py
------------------------------
Extracts 512-dimensional Re-ID feature vectors using OSNet x1.0 (torchreid).

CRITICAL: torchreid's `pretrained=True` flag loads only ImageNet weights —
NOT Re-ID weights. ImageNet-trained OSNet is barely better than ResNet-50
for person identification. The Market-1501 Re-ID weights must be loaded
separately, which is what we do below.

With proper Re-ID weights (Market-1501):
    Same person, front→back:   0.68–0.88
    Different people, similar: 0.20–0.50

With ImageNet-only weights (the wrong setup we had before):
    Same person, front→back:   0.55–0.85
    Different people, similar: 0.40–0.70

The two distributions overlap heavily with ImageNet weights, which is why
threshold tuning was so unstable.

IMPORTANT: changing the embedding backbone invalidates the database.
Delete surveillant/database/surveillant.db before the first run.
"""

import os
import numpy as np
import cv2
import torch
import torchvision.transforms as T

EMBEDDING_DIM = 512   # OSNet x1.0 output dimension

# Market-1501 Re-ID weights — torchreid's official Google Drive mirror.
OSNET_MARKET1501_URL = "https://drive.google.com/uc?id=1vduhq5DpN2q1g4fYEZfPI17MJeh9qyrA"
OSNET_MARKET1501_FILENAME = "osnet_x1_0_market1501.pth"


class PersonEmbedder:
    """
    Extracts appearance features using OSNet x1.0 (torchreid).

    Loads Market-1501 Re-ID weights on top of the ImageNet backbone.
    If the Re-ID weights cannot be downloaded (no internet, etc.) the
    embedder falls back to ImageNet weights with a loud warning — the
    user will see degraded Re-ID accuracy and unstable thresholds.
    """

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[PersonEmbedder] Initializing OSNet x1_0 on {self.device}...")

        import torchreid
        # Step 1: build model and load ImageNet weights (always succeeds).
        backbone = torchreid.models.build_model(
            name="osnet_x1_0",
            num_classes=1,
            pretrained=True,    # ImageNet weights ONLY — Re-ID weights loaded below
        )

        # Step 2: layer Market-1501 Re-ID weights on top.
        self._reid_weights_loaded = self._try_load_reid_weights(backbone)
        if not self._reid_weights_loaded:
            print("[PersonEmbedder] !! WARNING: running with ImageNet weights only.")
            print("[PersonEmbedder] !! Re-ID accuracy will be DEGRADED. Same-person/")
            print("[PersonEmbedder] !! different-person distributions will overlap.")
            print("[PersonEmbedder] !! Expect threshold instability and false merges.")

        self.model = backbone.eval().to(self.device)

        # Standard Re-ID preprocessing: 256×128 (H×W), ImageNet normalisation.
        # Using 256×128 instead of 224×224 because OSNet was trained on this
        # aspect ratio — it matches the expected body proportions.
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[PersonEmbedder] Embedder ready (Body / OSNet x1_0, dim={EMBEDDING_DIM}).")

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def _try_load_reid_weights(self, model) -> bool:
        """
        Attempt to load Market-1501 Re-ID weights into ``model``.

        Looks for a local cached file first; if missing, downloads via gdown
        from torchreid's official Google Drive URL. Returns True on success,
        False otherwise (caller falls back to ImageNet weights with a warning).
        """
        try:
            from pathlib import Path
            cache_dir = Path.home() / ".cache" / "torchreid" / "checkpoints"
            cache_dir.mkdir(parents=True, exist_ok=True)
            weights_path = cache_dir / OSNET_MARKET1501_FILENAME

            if not weights_path.exists():
                print(f"[PersonEmbedder] Downloading Market-1501 Re-ID weights -> {weights_path}")
                try:
                    import gdown
                    gdown.download(OSNET_MARKET1501_URL, str(weights_path), quiet=False)
                except Exception as e:
                    print(f"[PersonEmbedder] Download failed: {e}")
                    return False

            if not weights_path.exists() or weights_path.stat().st_size < 1_000_000:
                # File missing or too small to be a real weights file
                print("[PersonEmbedder] Weights file missing or corrupt.")
                return False

            # Load via torchreid's utility — handles state-dict key mismatches.
            # The submodule path varies by torchreid version; try both.
            try:
                from torchreid.reid.utils import load_pretrained_weights
            except ImportError:
                from torchreid.utils import load_pretrained_weights
            load_pretrained_weights(model, str(weights_path))
            print(f"[PersonEmbedder] OK -- Market-1501 Re-ID weights loaded from {weights_path}")
            return True

        except Exception as e:
            print(f"[PersonEmbedder] Re-ID weight loading failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_face_embedding(self, crop: np.ndarray):
        """Face extraction is bypassed — triggers body fallback automatically."""
        return None

    def extract_body_embedding(self, crop: np.ndarray) -> np.ndarray:
        """
        Extract a 512-dimensional normalized body feature vector.

        Args:
            crop: BGR image patch (any size — resized internally to 256×128).

        Returns:
            np.ndarray: 1-D L2-normalised float32 vector of length 512.
        """
        if crop is None or crop.size == 0:
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # In eval mode OSNet returns the 512-d feature, not class logits.
            feat = self.model(tensor).cpu().numpy().flatten()

        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm

        return feat.astype(np.float32)

    def aggregate_embeddings(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """
        Aggregate a list of embeddings into a single prototype.

        Uses the per-dimension MEDIAN rather than mean because:
            - The 4-frame identification buffer occasionally contains a bad
              frame (blur, partial occlusion, atypical pose) that the quality
              gate didn't catch.
            - Mean is dragged by such outliers; median ignores them.
            - For unimodal distributions (same person, similar angles) median
              and mean agree closely, so we don't lose accuracy on clean buffers.
        """
        if not embeddings:
            raise ValueError("No embeddings provided to aggregate.")

        agg = np.median(np.stack(embeddings, axis=0), axis=0)
        norm = np.linalg.norm(agg)
        if norm > 0:
            agg = agg / norm

        return agg.astype(np.float32)

    def serialize(self, embedding: np.ndarray) -> bytes:
        """Convert a numpy array to raw bytes for SQLite BLOB storage."""
        return embedding.tobytes()

    def deserialize(self, data: bytes) -> np.ndarray:
        """Convert raw bytes back into a numpy float32 array.

        The shape is implicit in the byte length (512 floats × 4 bytes = 2048 B).
        Old 2048-d ResNet-50 embeddings produce a 2048-element array; the
        dimension-check in searcher.py and reconciliation worker automatically
        skips them, so the DB won't crash — but please clear it for a fresh start.
        """
        return np.frombuffer(data, dtype=np.float32)
