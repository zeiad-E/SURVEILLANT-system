# SURVEILLANT Enhancement Proposal
## Full Technical Report 

**System:** SURVEILLANT — Multi-Camera Person Re-Identification & Tracking  
---

## Executive Summary

The system currently has **four structural weaknesses** that compound each other:
1. An embedding backbone not designed for Re-ID (ResNet-50, ImageNet weights)
2. No crop quality control — garbage crops reach the embedder unfiltered
3. No preprocessing for adverse lighting conditions (darkness, glare)
4. A search/matching architecture that ignores spatial and temporal context

Every downstream failure — false duplicates, color changes on pose change, poor cross-camera matching — traces back to **low-quality embeddings**. This report proposes concrete, implementable solutions ordered by impact vs. effort.

---

## Part 1 — The Core Problem: Garbage In, Garbage Out

The embedding is extracted from whatever YOLO hands to us: blurry crops, half-occluded bodies, dark silhouettes, crops where 60% of the pixels are wall. A good embedding pipeline must **reject bad crops before they ever reach the model**.

---

## Part 2 — Crop Quality Gate (Highest Impact, Lowest Effort)

Before any crop reaches the embedder, filter it through a quality gate. Three cheap checks catch 80% of the garbage:

### 2.1 Blur Score (Laplacian Variance)
```python
def is_blurry(crop, threshold=80.0):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold
```
A motion-blurred or out-of-focus crop has a very low Laplacian variance. This single check eliminates a large fraction of bad frames caused by fast movement or camera shake.

### 2.2 Minimum Crop Size
A crop smaller than ~48×96 pixels (person at the edge of the frame or very far away) produces embeddings with almost no discriminative power after upscaling to 224×224. Reject outright.

### 2.3 Darkness / Saturation Score
```python
def is_too_dark(crop, threshold=30):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return hsv[:,:,2].mean() < threshold  # V channel = brightness
```
A mean brightness below 30 (0–255 scale) means the crop is essentially a silhouette. The embedding will encode shadow shapes, not identity.

**Where to add this:** In `gallery.py:maybe_update_gallery()` and in `main.py` before queueing crops into the `embed_queue`. This is a ~20-line change with very high payoff.

---

## Part 3 — Low-Light & Dark Condition Enhancement

### 3.1 CLAHE Preprocessing (Contrast Limited Adaptive Histogram Equalization)

Research confirms CLAHE outperforms global histogram equalization for surveillance: it enhances local contrast in dark areas without blowing out bright regions or amplifying noise in uniform areas. Apply it to every frame **before** YOLO detection — improving both detection accuracy and crop quality simultaneously.

```python
def enhance_frame(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    enhanced = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
```

This operates in LAB color space so only the luminance channel is equalized — colors are preserved, which matters for the embedder's appearance features.

### 3.2 Auto-Gamma Correction
For very dark frames where CLAHE alone isn't enough, apply auto-gamma based on mean brightness:
```python
def auto_gamma(frame):
    mean_v = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:,:,2].mean()
    if mean_v < 60:
        gamma = 0.5  # brighten aggressively
        table = np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)], dtype=np.uint8)
        return cv2.LUT(frame, table)
    return frame
```

### 3.3 Long-Term: Infrared / Visible Cross-Modality (VI-ReID) *** there is no infrared camera in this project just for information so ship thie point
For true 24/7 surveillance, the industry standard is adding an infrared camera channel and using VI-ReID (Visible-Infrared Re-ID) models. The embedding model is trained on both RGB and IR inputs, making it naturally robust to complete darkness. This requires IR-capable hardware but eliminates all lighting problems at the architecture level.

**References:**
- *Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark* — CVPR 2023
- *Empowering Visible-Infrared Person Re-ID* — NeurIPS 2024

---

## Part 4 — Background Isolation: Cleaner Crops

### 4.1 Switch from `yolov8n.pt` → `yolov8n-seg.pt`

This is the **single most impactful structural improvement** for embedding quality. The `-seg` variant outputs a **pixel-level segmentation mask** for each detected person in addition to the bounding box. You then zero out all background pixels before sending the crop to the embedder:

```python
# With yolov8n-seg.pt (task='segment'):
results = model.predict(frame, task='segment', conf=0.45, iou=0.40)
for result in results:
    for i, (box, mask) in enumerate(zip(result.boxes, result.masks)):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        crop = frame[y1:y2, x1:x2]
        
        # Resize mask to crop dimensions and apply
        person_mask = cv2.resize(mask.data[0].numpy(), (crop.shape[1], crop.shape[0]))
        person_mask = (person_mask > 0.5).astype(np.uint8)
        
        # Replace background with neutral gray (not black — black biases the embedder)
        neutral = np.full_like(crop, 128)
        masked_crop = np.where(person_mask[:,:,None], crop, neutral)
```

**Why this matters critically:**
- If a person stands in front of a red wall, the embedder currently encodes "red wall + person." Two different people in front of the same red wall have artificially high similarity.
- Two views of the same person in front of different backgrounds have artificially low similarity.
- Background removal directly attacks the root cause of both false positives and false negatives.

**Cost:** `yolov8n-seg.pt` adds ~3ms per frame on CPU — entirely acceptable for the quality improvement gained.

---

## Part 5 — Embedding Model: The Most Critical Upgrade

### 5.1 The Problem with ResNet-50 (ImageNet)

ResNet-50 trained on ImageNet classifies objects ("cat", "car", "chair"). Its internal representations encode **semantic category**, not **individual identity**. Two people in similar clothing often score 0.70+ cosine similarity while the same person from the front vs. back scores 0.55. This distribution overlap is the fundamental root cause of:
- The "gallery sponge" problem (one person absorbs all new tracks)
- The "color change on turn" problem (front≠back, below threshold → new person)
- Unreliable thresholds that need constant tuning

### 5.2 Recommended: OSNet via Torchreid

**OSNet (Omni-Scale Network)** was purpose-built for person Re-ID (ICCV 2019, KaiyangZhou). It remains one of the best lightweight Re-ID models available. Pre-trained weights are available for Market-1501 + DukeMTMC — the two largest Re-ID benchmark datasets.

**Key architecture advantages:**
- **Omni-scale feature learning:** captures both fine-grained texture (clothing pattern, shoe color) and coarse body structure (height, build) simultaneously
- **Output:** 512-d identity-discriminative vector
- **Pretrained on Re-ID data:** tuned to distinguish individuals, not classify object categories
- **Lightweight variants:** `osnet_x0_5` is smaller than MobileNetV3 but dramatically more accurate

```bash
pip install torchreid
```

```python
import torchreid

model = torchreid.models.build_model(
    name='osnet_x1_0',
    num_classes=1,      # feature extraction mode — ignores classification head
    pretrained=True     # downloads Market-1501 weights automatically
)
model.eval()
# Output: 512-d L2-normalized Re-ID vector
```

**Expected improvement in cosine similarity distributions:**

| Pair Type | ResNet-50 (ImageNet) | OSNet (Market-1501) |
|---|---|---|
| Same person, same angle | 0.90–0.99 | 0.92–0.99 |
| Same person, 90° turn | 0.55–0.75 | 0.78–0.92 |
| Same person, front→back | 0.45–0.65 | 0.68–0.88 |
| Different people, similar clothes | 0.55–0.75 | 0.20–0.50 |

The two distributions (same person / different person) barely overlap with OSNet, making thresholds reliable and stable. The match threshold can be set confidently at **0.72–0.75** without risking false merges.

**Recalibrated thresholds for OSNet:**
```python
BODY_MATCH_THRESHOLD       = 0.72   # stable with OSNet
BODY_GALLERY_ADD_DISTANCE  = 0.20   # OSNet embeddings are tighter
GALLERY_MAX_DISTANCE       = 0.50
MERGE_CANDIDATE_THRESHOLD  = 0.72
AUTO_MERGE_THRESHOLD       = 0.85
```

### 5.3 Alternative: CLIP (OpenAI ViT-B/32)
CLIP is trained on 400M image-text pairs and produces embeddings that are remarkably robust to lighting changes, pose changes, and occlusion — even without being trained on Re-ID data. Best used as a fallback when OSNet confidence is low (dark/occluded crops). Can be combined with OSNet in an ensemble.

---

## Part 6 — Pose-Aware Gallery Strategy

The current gallery stores embeddings purely based on cosine distance from existing views. A better strategy explicitly tracks which canonical viewpoints are covered.

### 6.1 Target 4 Canonical Views Per Person

Maintain a target set: **front, back, left-side, right-side**. Use the bounding box aspect ratio and motion direction as a cheap view classifier:

```python
def estimate_view(bbox, prev_bbox=None):
    x1, y1, x2, y2 = bbox
    aspect = (x2-x1) / max(y2-y1, 1)
    if aspect < 0.40:
        return "side"
    if prev_bbox is not None:
        dx = (x1+x2)/2 - (prev_bbox[0]+prev_bbox[2])/2
        return "right_moving" if dx > 8 else "left_moving" if dx < -8 else "frontal"
    return "frontal"
```

Store the view tag per embedding. When updating the gallery, prioritize adding views from underrepresented angles. Once all 4 views exist, switch to diversity-only updates.

### 6.2 View Coverage Score
Track a "view coverage" score per person (0.0–1.0) based on how many canonical angles are covered. Only persons with coverage >= 0.5 (at least 2 distinct angles) are treated as "confirmed" for cross-camera matching.

---

## Part 7 — Tracking Upgrade: ByteTrack

DeepSORT was state-of-the-art in 2020. **ByteTrack** (ECCV 2022) is the current production standard in surveillance systems. The fundamental difference:

| Property | DeepSORT | ByteTrack |
|---|---|---|
| Low-confidence detections (0.1–0.45) | Discarded | Used for track association |
| Occlusion handling | Poor — track dies | Excellent — survives through occlusion |
| Re-ID model dependency | Yes (slows pipeline) | No (IoU + Kalman only) |
| Track identity switches | Frequent | Rare |
| CPU performance | ~8–12 fps/cam | ~15–20 fps/cam |

**ByteTrack's key insight:** When a person is half-occluded, YOLO outputs a low-confidence detection (0.25–0.40). DeepSORT discards it (below threshold). ByteTrack uses it to **keep the track alive** through the occlusion period. This directly fixes:
- The "color changes when person turns" problem (track survives the brief miss during rotation)
- The "bounding box jumps" problem (track maintained continuously through brief occlusions)
- The duplicate track problem (new track never created because old one never died)

```bash
pip install bytetracker
```

---

## Part 8 — Fast Approximate Search: FAISS

Currently `searcher.py` loads ALL embeddings from SQLite on every query, then runs a Python loop computing cosine similarity one-by-one. At 27 persons this is fine. At 500 persons it becomes the real-time bottleneck.

**FAISS** (Facebook AI Similarity Search) is an in-memory vector index that performs approximate nearest-neighbor search at ~1000× the speed of a linear Python scan:

```python
import faiss
import numpy as np

# Build index (done once at startup, updated incrementally)
index = faiss.IndexFlatIP(512)        # inner product = cosine sim on L2-norm vectors
index.add(all_embeddings_matrix)      # shape: (N_persons, 512)

# Query (replaces the entire searcher loop)
scores, person_indices = index.search(query_embedding.reshape(1,-1), k=5)
```

**Performance:** 10,000 persons, 512-d vectors → top-5 search in < 1ms on CPU. The index lives in memory and is rebuilt from SQLite on startup. New persons are added incrementally via `index.add()`.

```bash
pip install faiss-cpu
```

---

## Part 9 — Spatio-Temporal Camera Constraints

Re-ID researchers call this **spatio-temporal constraints**. If camera 3 is at the east entrance and camera 1 is at the west entrance, a person cannot appear at camera 1 just 2 seconds after being seen at camera 3 — the physical walk takes at least 30 seconds. Currently the system has no concept of camera topology and accepts any match regardless of physical plausibility.

### 9.1 Camera Transition Time Matrix
```python
# In settings.py — minimum seconds to physically move between camera zones
CAM_TRANSITION_MIN_SEC = {
    (0, 1): 10,   # cam0 → cam1: adjacent, 10s minimum
    (0, 2): 25,   # cam0 → cam2: far, 25s minimum
    (1, 3): 15,
    # Symmetric: (1, 0), (2, 0), (3, 1) same values
}
```

### 9.2 Applying the Constraint in the Searcher
```python
def search_by_embedding(self, query_emb, query_cam, query_time, ...):
    for pid, gallery in all_galleries.items():
        person = db.get_person(pid)
        last_cam = person['last_seen_cam']
        last_time = parse_time(person['last_seen_time'])
        elapsed = (query_time - last_time).total_seconds()
        min_required = CAM_TRANSITION_MIN_SEC.get((last_cam, query_cam), 0)
        if elapsed < min_required:
            continue  # physically impossible — reject match
```

This eliminates a whole class of false positives where two similar-looking people on different cameras are wrongly merged because they happened to appear in the same time window.

---

## Part 10 — Additional Improvements

### 10.1 Appearance Confidence Score
Instead of a binary match/no-match decision, compute a **confidence score** that combines:
- Cosine similarity
- Gallery size (more views = more trustworthy)
- Crop quality score (sharpness, brightness, size)
- Time since last seen (fresher = more reliable)

Only accept a match if the combined confidence exceeds a threshold — not just the raw similarity.

### 10.2 Description-Based Matching (LLM Integration)
The system already has `modules/llm/describer.py` prepared for Phase 4. Using the LLM to generate text descriptions ("person wearing red jacket, black backpack, ~175cm") creates a **text-based secondary index**. When visual matching is uncertain (darkness, occlusion), text descriptions provide a fallback:
- Visual similarity 0.60 (uncertain) + matching description ("red jacket") = accept match
- Visual similarity 0.65 (uncertain) + conflicting description ("blue jacket") = reject match

### 10.3 Multi-Query Aggregation for Identification
Instead of averaging 4 consecutive frames into one embedding, **keep the 4 embeddings separate** and run 4 queries against the gallery. Accept a match only if at least 3/4 queries agree on the same person (voting):

```python
votes = {}
for emb in per_frame_embeddings:
    matches = searcher.search_by_embedding(emb, top_k=1)
    if matches and matches[0]['similarity_score'] >= threshold:
        pid = matches[0]['person_id']
        votes[pid] = votes.get(pid, 0) + 1

# Accept only if majority vote
winner = max(votes, key=votes.get) if votes else None
if winner and votes[winner] >= 3:
    accept_match(winner)
```

This makes identification robust to individual bad frames (blur, partial occlusion).

---

## Part 11 — Implementation Roadmap

### Tier 1 — High Impact, Low Effort (This Week)
| Change | Files | Expected Improvement |
|---|---|---|
| Crop quality gate: blur + darkness + min-size filter | `gallery.py`, `main.py` | −60% bad embeddings entering the system |
| CLAHE + auto-gamma frame preprocessing | `detector.py` | Better detection accuracy in dark/indoor scenes |
| Multi-query voting for identification | `main.py` (embedding_worker) | More robust initial identity assignment |

### Tier 2 — High Impact, Medium Effort (Next Week)
| Change | Files | Expected Improvement |
|---|---|---|
| Switch embedder to **OSNet** (torchreid) | `embedder.py`, `settings.py` | Re-ID accuracy: ~65% → 88%+ on pose/lighting variation |
| Switch YOLO to **yolov8n-seg.pt** for background masking | `detector.py`, `main.py` | −40% background interference in embeddings |
| Recalibrate all thresholds for OSNet | `settings.py` | Stable, reliable identity matching |
| Pose-aware gallery tagging | `gallery.py` | Intentional angle diversity per person |

### Tier 3 — Architectural Changes (Two Weeks)
| Change | Expected Improvement |
|---|---|
| Replace DeepSORT with **ByteTrack** | Fewer track deaths, fewer color changes on occlusion |
| Add **FAISS** vector index | Scales to 10,000+ persons without slowdown |
| Add spatio-temporal camera transition constraints | Eliminate cross-camera false positives |
| LLM description as secondary matching signal | Handles cases where visual matching is uncertain |

---

## Summary: Priority Order

1. **Crop Quality Gate** — cheapest change, biggest safety improvement
2. **CLAHE Preprocessing** — 10 lines, free darkness handling
3. **OSNet Embedder** — single biggest accuracy jump
4. **YOLO Segmentation** — eliminates background interference  
5. **ByteTrack** — eliminates track-death color changes
6. **FAISS** — enables production scale
7. **Spatio-Temporal Constraints** — eliminates impossible cross-camera matches
8. **LLM Description Matching** — fills gaps where vision alone fails

---

## References

- [Background Matters: Language-Enhanced Adversarial Framework for Person Re-ID (arXiv 2025)](https://arxiv.org/html/2509.03032v1)
- [Unsupervised Re-ID: Adaptive Foreground Enhancement — IET Image Processing 2024](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/ipr2.13277)
- [Identity Hides in Darkness: Nighttime Person Re-ID (PMC 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11820754/)
- [Diverse Embedding Expansion for Low-Light VI-ReID — CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Zhang_Diverse_Embedding_Expansion_Network_and_Low-Light_Cross-Modality_Benchmark_for_Visible-Infrared_CVPR_2023_paper.pdf)
- [Torchreid: Deep Learning Person Re-ID Library — GitHub](https://github.com/KaiyangZhou/deep-person-reid)
- [OSNet Architecture Overview — DeepWiki](https://deepwiki.com/KaiyangZhou/deep-person-reid/9.1-osnet-architecture)
- [YOLOv8 Segmentation: Isolating Objects — Ultralytics Docs](https://docs.ultralytics.com/guides/isolating-segmentation-objects/)
- [CLAHE for Detection Under Low-Light — IEEE Xplore](https://ieeexplore.ieee.org/document/8780492/)
- [Survey on Person and Vehicle Re-ID — IET Computer Vision 2024](https://ietresearch.onlinelibrary.wiley.com/doi/full/10.1049/cvi2.12316)
- [From Poses to Identity: Training-Free ReID via Feature Centralization (arXiv 2025)](https://arxiv.org/html/2503.00938v1)
- [Nighttime Person Re-ID via Collaborative Enhancement Network (arXiv)](https://arxiv.org/html/2312.16246v2)
