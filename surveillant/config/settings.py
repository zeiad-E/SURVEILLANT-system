"""
SURVEILLANT — Central Configuration
All thresholds, paths, and model names live here.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VIDEOS_DIR    = BASE_DIR / "data" / "videos"
SNAPSHOTS_DIR = BASE_DIR / "data" / "snapshots"
DB_PATH       = BASE_DIR / "database" / "surveillant.db"
TRACK_REGISTRY_PATH = BASE_DIR / "database" / "track_registry_session.json"

# ---------------------------------------------------------------------------
# Camera / simulator
# ---------------------------------------------------------------------------
FPS_TARGET     = 30
DISPLAY_WIDTH  = 480
DISPLAY_HEIGHT = 270
GRID_COLS      = 3

# ---------------------------------------------------------------------------
# Detection (YOLOv8)
# ---------------------------------------------------------------------------
# Segmentation variant — provides per-person pixel masks used by the
# preprocessing stage to suppress background before embedding (Part 4).
YOLO_MODEL      = "yolov8n-seg.pt"
# Low threshold so ByteTrack receives both high- and low-confidence detections
# for its two-stage association (Part 7). Crops for embedding are gated
# separately by BYTETRACK_TRACK_THRESH so embedding quality is preserved.
DETECTION_CONF  = 0.10
DETECTION_CLASS = 0
DETECTION_IMGSZ = 256   # 256 instead of original 320 — reduces inference time ~30%

# ---------------------------------------------------------------------------
# Frame preprocessing (Part 3 — low-light / adverse lighting enhancement)
# ---------------------------------------------------------------------------
ENABLE_FRAME_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 2.0
CLAHE_TILE_GRID_SIZE     = (8, 8)
AUTO_GAMMA_THRESHOLD     = 60
AUTO_GAMMA_VALUE         = 0.5

# ---------------------------------------------------------------------------
# Crop quality gate (Part 2 — reject garbage crops before gallery storage)
# ---------------------------------------------------------------------------
CROP_BLUR_THRESHOLD     = 50.0   # Laplacian variance minimum (50 accepts mild motion blur)
CROP_MIN_WIDTH          = 48
CROP_MIN_HEIGHT         = 96
CROP_DARKNESS_THRESHOLD = 30     # HSV V mean minimum

# ---------------------------------------------------------------------------
# Background isolation via YOLO segmentation (Part 4)
# ---------------------------------------------------------------------------
USE_SEGMENTATION             = True
SEGMENTATION_MASK_THRESHOLD  = 0.5
SEGMENTATION_TRACK_IOU       = 0.4
BACKGROUND_REPLACEMENT_COLOR = 128   # neutral gray

# ---------------------------------------------------------------------------
# Tracking (ByteTrack — Part 7, replaces DeepSORT)
# ---------------------------------------------------------------------------
# ByteTrack two-stage association thresholds:
#   First stage  : high-confidence detections (>= BYTETRACK_TRACK_THRESH)
#   Second stage : low-confidence detections  (>= BYTETRACK_LOW_THRESH, < BYTETRACK_TRACK_THRESH)
#                  used to keep coasting tracks alive through brief occlusions
BYTETRACK_TRACK_THRESH = 0.45   # high-conf gate (first-stage association + embedding crops)
BYTETRACK_LOW_THRESH   = 0.10   # low-conf gate  (second-stage, keeps tracks alive)
BYTETRACK_MATCH_THRESH = 0.80   # IoU matching threshold
BYTETRACK_TRACK_BUFFER = 30     # frames to hold a lost track before dropping (~1 s @ 30 fps)

# Display filter: hide Kalman-predicted box after this many consecutive missed
# detection cycles. ByteTrack handles internal coasting up to BYTETRACK_TRACK_BUFFER;
# this is a stricter display-side filter to reduce ghost-box drift.
TRACKING_COAST_FRAMES = 4

# Display staleness — tracks older than this (seconds) are cleared from screen
STALE_TRACK_TIMEOUT = 3.0

# ---------------------------------------------------------------------------
# Embedding (OSNet x1.0 via torchreid — Part 5, replaces ResNet-50)
# ---------------------------------------------------------------------------
EMBEDDING_DIM            = 512  # OSNet x1.0 global feature dimension
FACE_DET_SIZE            = (640, 640)
MIN_FACE_CONF            = 0.5
NUM_FRAMES_FOR_EMBEDDING = 4

# ---------------------------------------------------------------------------
# Cross-camera identity matching thresholds (recalibrated for OSNet — Part 5)
# ---------------------------------------------------------------------------
# OSNet's same-/different-person distributions barely overlap, making 0.72
# a reliable decision boundary. ResNet-50 needed 0.63 because its distributions
# were much wider (same person front→back scored only 0.45–0.65).
FACE_MATCH_THRESHOLD  = 0.55
BODY_MATCH_THRESHOLD  = 0.65   # calibrated for real indoor surveillance:
                               # same person across pose changes: ~0.65-0.85
                               # different people in similar office clothes: ~0.40-0.62
                               # 0.60 caused false merges; 0.72 caused false splits
CROSS_TYPE_MULTIPLIER = 0.85

# Legacy aliases
FACE_SIMILARITY_THRESHOLD = FACE_MATCH_THRESHOLD
BODY_SIMILARITY_THRESHOLD = BODY_MATCH_THRESHOLD
SIMILARITY_THRESHOLD      = BODY_MATCH_THRESHOLD

MIN_GALLERY_FOR_MATCHING = 1

# ---------------------------------------------------------------------------
# Gallery update thresholds (recalibrated for OSNet — Part 5)
# ---------------------------------------------------------------------------
FACE_GALLERY_ADD_DISTANCE  = 0.25
BODY_GALLERY_ADD_DISTANCE  = 0.20   # was 0.40; OSNet embeddings cluster tighter
GALLERY_MAX_DISTANCE       = 0.55   # accepts challenging same-person poses (distance
                                   # up to 0.55 = similarity down to 0.45) while
                                   # blocking obviously-different-person embeddings;
                                   # 0.65 was too permissive — gallery got polluted
MAX_GALLERY_SIZE           = 10
MIN_FRAMES_BETWEEN_SAMPLES = 15

# Legacy aliases
GALLERY_NEW_VIEW_THRESHOLD  = BODY_GALLERY_ADD_DISTANCE
NEW_VIEW_DISTANCE_THRESHOLD = BODY_GALLERY_ADD_DISTANCE
MAX_VIEW_DISTANCE_TO_ACCEPT = GALLERY_MAX_DISTANCE
MIN_FRAMES_BEFORE_SAMPLE    = MIN_FRAMES_BETWEEN_SAMPLES

# ---------------------------------------------------------------------------
# Pose-aware gallery (Part 6)
# ---------------------------------------------------------------------------
# Four canonical viewpoints the gallery tries to cover for each person.
# estimate_view() in gallery.py maps a bounding box to one of these tags.
# Uncovered canonical slots get force-accepted regardless of cosine distance.
CANONICAL_VIEWS = ("frontal", "right_moving", "left_moving", "side")

# Minimum view coverage fraction (covered slots / 4) for a person to be
# considered a reliable cross-camera match target. Below this threshold the
# searcher skips the person to avoid matching on a single-angle prototype.
MIN_VIEW_COVERAGE_FOR_MATCHING = 0.5   # at least 2 distinct canonical views

# ---------------------------------------------------------------------------
# Background Reconciliation
# ---------------------------------------------------------------------------
RECONCILIATION_INTERVAL_SEC = 120

# Thresholds are for MEAN-POOL similarity (average across all compatible pairs),
# not max-pool. Mean-pool is far more reliable for merge decisions — a single
# accidentally-similar pair can no longer trigger a false proposal.
#
# Expected mean-pool scores (OSNet):
#   Same person, multi-angle gallery  : 0.60–0.82
#   Different people, similar clothes : 0.15–0.45
MERGE_CANDIDATE_THRESHOLD   = 0.58   # mean-pool: propose pairs for human review
AUTO_MERGE_THRESHOLD        = 0.82   # mean-pool: auto-merge only when very confident;
                                     # 0.75 was causing false auto-merges of different people
GHOST_TTL_SEC               = 180

# Minimum gallery size a person must have before being a reconciliation target.
# 2 is sufficient — 3 was preventing fresh duplicates from ever being caught.
MIN_GALLERY_FOR_RECONCILIATION = 2

# ---------------------------------------------------------------------------
# LLM (Ollama + Qwen2.5-VL)
# ---------------------------------------------------------------------------
OLLAMA_HOST = "http://localhost:11434"
LLM_MODEL   = "qwen2.5vl:2b"

# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------
SNAPSHOT_QUALITY         = 90
MAX_SNAPSHOTS_PER_PERSON = 5
