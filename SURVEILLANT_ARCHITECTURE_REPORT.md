# SURVEILLANT System Architecture Report

The SURVEILLANT system is a multi-camera real-time person re-identification (Re-ID) and tracking network. This document tracks the full evolution of the architecture through all completed phases.

---

## 1. Core Architecture Overview

The system is designed around a multi-threaded, asynchronous processing pipeline that separates high-frequency tasks (bounding box tracking) from low-frequency, CPU-intensive tasks (feature extraction and database reconciliation).

### 1.1 The Threading Model

Three dedicated thread domains eliminate CPU starvation:

1. **Detection Loop (Round-Robin Worker)**
   - Cycles through all cameras one at a time — one YOLO inference per cycle — eliminating parallel-process CPU contention.
   - Every frame is passed through `FrameEnhancer` (CLAHE + auto-gamma) before YOLO, improving detection in dark/indoor scenes.
   - Uses `yolov8n-seg.pt` which produces per-person pixel masks alongside bounding boxes.
   - Feeds detections — including **low-confidence** ones (≥ 0.10) — to **ByteTrack** for two-stage association.
   - Per-track binary masks are recovered post-tracking by IoU-matching tracks to their originating detections.

2. **Asynchronous Embedding Worker**
   - Consumes an `embed_queue` of `(cam_id, track_id, crop, ...)` items.
   - Crops are already mask-cleaned (background → neutral gray) and pre-gated for quality.
   - Only high-confidence crops (≥ `BYTETRACK_TRACK_THRESH = 0.45`) are added to the identification buffer.
   - Uses **OSNet x1.0 (torchreid, Market-1501 weights)** to extract 512-d Re-ID feature vectors.
   - Manages SQLite writes and gallery updates with pose-aware canonical-view tagging.

3. **Background Reconciliation Worker**
   - Daemon thread waking every 120 seconds.
   - Performs O(N²) cosine-similarity checks to auto-merge identities that were split across sessions.

---

## 2. Subsystem Details & Algorithms

### 2.1 Tracking Subsystem — ByteTrack (Part 7)

Replaced DeepSORT (Phase 1–3) with ByteTrack (ECCV 2022).

| Property | DeepSORT (old) | ByteTrack (current) |
|---|---|---|
| Low-confidence detections (0.10–0.45) | Discarded | Used for second-stage association |
| Occlusion handling | Poor — track dies | Survives through brief occlusion |
| Re-ID model dependency | Yes (appearance embedder) | No — IoU + Kalman only |
| Track identity switches | Frequent | Rare |
| `time_since_update` coasting | Hardcoded at 2 | Configurable `TRACKING_COAST_FRAMES = 4` |

**Two-stage association:**
- *Stage 1* (high-conf ≥ 0.45): IoU matching against all active tracks.
- *Stage 2* (low-conf ≥ 0.10, < 0.45): Matches only against tracks that went unmatched in Stage 1. A person partially occluded produces a low-conf detection — ByteTrack uses it to keep the track alive rather than killing and re-creating it.

`DETECTION_CONF = 0.10` so YOLO passes all detections to ByteTrack. High-confidence gating is applied downstream (in `main.py`) when adding crops to the identification buffer.

### 2.2 Embedding Backbone — OSNet x1.0 (Part 5)

Replaced ResNet-50 (ImageNet, 2048-d) with **OSNet x1.0** (torchreid, Market-1501 + DukeMTMC, 512-d).

**Why OSNet vs ResNet-50:**

| Pair Type | ResNet-50 (ImageNet) | OSNet (Market-1501) |
|---|---|---|
| Same person, same angle | 0.90–0.99 | 0.92–0.99 |
| Same person, 90° turn | 0.55–0.75 | 0.78–0.92 |
| Same person, front→back | 0.45–0.65 | 0.68–0.88 |
| Different people, similar clothes | 0.55–0.75 | 0.20–0.50 |

ResNet-50 encodes *semantic category* (ImageNet training); the same-person / different-person distributions overlap heavily, making thresholds unreliable. OSNet was purpose-built for individual Re-ID; its distributions barely touch, making `BODY_MATCH_THRESHOLD = 0.72` a stable, safe decision boundary.

**Architecture details:**
- Input: 256×128 px (H×W), standard Re-ID crop size
- Output: 512-d L2-normalized feature vector (eval mode returns features directly)
- Pre-trained weights: Market-1501 (downloaded automatically on first run)

**⚠ Database migration**: Old 2048-d ResNet-50 embeddings are incompatible. The dimension-check in `searcher.py` and the reconciliation worker automatically skip them (no crash), but **delete `database/surveillant.db`** before the first run after upgrading to OSNet.

### 2.3 Pose-Aware Gallery (Part 6)

The gallery now actively tracks which of four canonical viewpoints have been covered per person.

**Canonical views:** `frontal`, `right_moving`, `left_moving`, `side`

**View classification — `estimate_view(bbox, prev_bbox)`:**
```
aspect = width / height
  < 0.40            → "side"          (person sideways, narrow bbox)
horizontal Δcenter  → direction-based → "right_moving" or "left_moving"
else                → "frontal"
```

**Force-accept logic:** When a canonical slot is empty for a person, the new embedding is accepted regardless of cosine distance (subject only to `GALLERY_MAX_DISTANCE` garbage check and `MAX_GALLERY_SIZE`). Once all 4 slots are filled the system falls back to diversity-only updates.

**View coverage score:** `covered_slots / 4` (0.0–1.0). Persons with coverage < `MIN_VIEW_COVERAGE_FOR_MATCHING = 0.5` (fewer than 2 distinct canonical views) are skipped during cross-camera search — a single-angle gallery is too unreliable as a Re-ID target.

### 2.4 Auto-Learning Global Gallery

- Every registered `person_id` maintains up to 10 gallery embeddings.
- Embeddings are stored with an `angle_tag`: `frontal`, `right_moving`, `left_moving`, `side`, `initial`, `cross_cam_view`, `very_different`, `same_cam_new_angle`, `partial_view`.
- The quality gate (`CropQualityGate`) runs before every embedding extraction — gallery never accumulates blurry, dark, or tiny crops.
- Gallery updates carry `bbox` + `prev_bbox` so `estimate_view()` can classify the pose.

### 2.5 The Preprocessing Stage (Phase 3.5 — Embedding-Quality Hardening)

All live in `modules/preprocessing/`.

#### FrameEnhancer
- **CLAHE** in LAB space (L channel only) — preserves color for the embedder.
- **Auto-gamma** (γ = 0.5) — fires only when frame mean brightness < 60.

#### CropQualityGate
| Check | Threshold | Rejects |
|---|---|---|
| Laplacian variance | ≥ 50.0 | Motion blur, out-of-focus |
| Minimum size | 48 × 96 px | Far/edge detections |
| HSV V mean | ≥ 30 | Silhouettes / dark crops |

Gate runs on the **raw** (pre-mask) crop. Applied on the gallery-update path (both in `main.py` and defensively inside `GalleryManager`). NOT applied to identification crops — even blurry crops carry enough signal for a one-time "does this person exist?" search.

#### Mask Application
- `associate_masks_to_tracks()` — rebuilds `track_id → mask` by IoU after ByteTrack reorders detections.
- `apply_mask_to_crop()` — background pixels → neutral gray 128. Tracks coasting without a matched detection fall back to the raw crop.

### 2.6 Storage Engine

- **SQLite + WAL** for concurrent read (detection thread) + write (embedding worker) without deadlocks.
- Schema: `persons`, `person_embeddings` (BLOB + angle_tag), `camera_history`, `merge_proposals`.
- Embeddings stored as raw `float32` bytes — dimension-agnostic deserialization (`np.frombuffer`).

### 2.7 Configuration Surface

```python
# Tracking (ByteTrack)
DETECTION_CONF         = 0.10    # pass all to ByteTrack
BYTETRACK_TRACK_THRESH = 0.45    # high-conf gate (stage 1 + embedding crops)
BYTETRACK_LOW_THRESH   = 0.10    # low-conf gate  (stage 2 — occlusion survival)
BYTETRACK_MATCH_THRESH = 0.80    # IoU association threshold
BYTETRACK_TRACK_BUFFER = 30      # frames to hold lost track
TRACKING_COAST_FRAMES  = 4       # display filter

# Embedding (OSNet)
EMBEDDING_DIM          = 512

# Identity matching (OSNet-calibrated)
BODY_MATCH_THRESHOLD      = 0.72   # was 0.63 with ResNet-50
BODY_GALLERY_ADD_DISTANCE = 0.20   # was 0.40
GALLERY_MAX_DISTANCE      = 0.50   # was 0.70
MERGE_CANDIDATE_THRESHOLD = 0.72   # was 0.65
AUTO_MERGE_THRESHOLD      = 0.85   # was 0.80

# Pose-aware gallery
CANONICAL_VIEWS                = ("frontal", "right_moving", "left_moving", "side")
MIN_VIEW_COVERAGE_FOR_MATCHING = 0.5   # ≥ 2 distinct views required

# Frame enhancement
ENABLE_FRAME_ENHANCEMENT = True
CLAHE_CLIP_LIMIT         = 2.0
AUTO_GAMMA_THRESHOLD     = 60

# Crop quality gate (gallery only)
CROP_BLUR_THRESHOLD     = 50.0
CROP_MIN_WIDTH          = 48
CROP_MIN_HEIGHT         = 96
CROP_DARKNESS_THRESHOLD = 30

# Segmentation masking
USE_SEGMENTATION             = True
SEGMENTATION_MASK_THRESHOLD  = 0.5
BACKGROUND_REPLACEMENT_COLOR = 128
```

---

## 3. Process Flow (Detection to Identity)

```mermaid
graph TD;
    A[Camera Feed] --> A1[FrameEnhancer: CLAHE + auto-gamma];
    A1 --> B[YOLOv8-seg: bbox + mask<br/>ALL detections ≥ 0.10];
    B --> C[ByteTrack two-stage association<br/>Stage1: high-conf | Stage2: low-conf];
    C --> C1[associate_masks_to_tracks IoU];
    C1 --> D{Track ID already bound?};

    D -- Yes --> E[Read state cache];
    E --> F[Render color box + identity];
    E --> G[Sample every N frames];
    G --> G1{Quality gate on raw crop?};
    G1 -- No --> G2[Drop — blurry / dark / tiny];
    G1 -- Yes --> G3[apply_mask_to_crop];
    G3 --> G4[estimate_view: canonical slot empty?];
    G4 -- Yes → force-accept --> H[Store in gallery with canonical tag];
    G4 -- No → diversity gate --> H;

    D -- No --> Q0{Detection conf ≥ 0.45?};
    Q0 -- No --> Q0b[Track visible, skip crop];
    Q0 -- Yes --> Q1[apply_mask_to_crop];
    Q1 --> I{Buffer 4 frames?};
    I -- No --> J[Render white box - collecting];
    I -- Yes --> K[Send buffer to EmbedQueue];

    K --> L[Async Worker: OSNet 512-d embedding];
    L --> M{View coverage ≥ 0.5 AND sim ≥ 0.72?};
    M -- Yes → MATCH --> N[Bind to existing person_uuid];
    M -- No → NEW --> O[Create new person_uuid];
    O --> P[Store in DB + session cache];
```

---

## 4. Phase History

| Phase | Key Changes |
|---|---|
| 1 | Detection + tracking + display (YOLOv8 + DeepSORT) |
| 2 | Persistent Re-ID, async embedding, SQLite + WAL, reconciliation worker |
| 3 | Photo-search CLI, max-pool searcher, gallery multi-angle strategy |
| 3.5 | Embedding-quality hardening: CLAHE, quality gate, YOLO-seg masking |
| 4 | **OSNet x1.0 embedder** (2048-d ResNet-50 → 512-d Re-ID backbone) |
| 5 | **Pose-aware gallery** (canonical views, force-accept, view-coverage gate) |
| 6 | **ByteTrack** (DeepSORT → two-stage IoU tracker, low-conf occlusion survival) |

---

## 5. Next Steps (Parts 8–10)

| Part | Change | Expected Improvement |
|---|---|---|
| 8 | **FAISS vector index** replacing linear SQLite scan | Sub-millisecond search at 10 000+ persons |
| 9 | **Spatio-temporal camera constraints** (transition-time matrix) | Eliminate physically-impossible cross-camera matches |
| 10 | **LLM description as secondary matching signal** (Qwen2.5-VL) | Handles dark/occluded cases where visual matching is uncertain |
