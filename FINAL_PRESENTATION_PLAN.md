# SURVEILLANT — Final Defense Presentation (Semester 2)
### Slide-by-slide build plan — v2

> **For the AI slide generator:** Use the `THEME` block below for global styling. Each
> slide is delimited by `---`. Every slide lists: **Owner** (presenter), **Layout**,
> **Title**, **On-slide content**, **Figure(s) + position**, and **Notes/animation**.
> Speaker talking-points are under *Say:* — these are NOT printed on the slide.
>
> **Presenters:** **[Z] = Zeiad** (architecture, body Re-ID, problems, LLM, web app) ·
> **[F] = Face colleague** (Semester-1 recap + vision, face & attributes, violence) ·
> **[A] = Audio colleague** (audio intelligence). **[ALL] = shared / on-stage together.**
>
> **Format:** 45 min present + 15 min Q&A · **Live** demo · ~46 slides · 16:9.
>
> **v2 changes:** S1 recap corrected (S1 = 8 separate face/attribute models, NOT the
> body pipeline); "different viewing angles → multi-vector gallery" added as Problem #1;
> face section expanded to 11 slides incl. 3 face-specific problems. **System stack =
> the REAL built system** (ByteTrack, SQLite, CNN-LSTM, Qwen2.5-VL:3b) — NOT the older
> planning-deck stack (DeepSORT / PostgreSQL / SlowFast / Llama).

---

## THEME

```yaml
theme:
  name: "Emerald Sentinel"
  mood: "high-tech surveillance command-center; confident, precise, modern"
  aspect_ratio: "16:9"
  base_style: "dark glassmorphism with neon accents over a deep-charcoal dot-grid"

  palette:
    background:      "#0A0E0D"   # deep charcoal-green base
    surface:         "#111715"   # card / panel
    surface_glass:   "rgba(17,23,21,0.6)"  # glassmorphism cards (blur 14px, 1px emerald border)
    primary:         "#10B981"   # emerald (key accents, active state, "live")
    primary_bright:  "#34D399"   # emerald highlight
    accent:          "#22D3EE"   # cyan (data lines, secondary highlight)
    text:            "#E6F2EC"   # near-white
    text_muted:      "#8A9A92"   # captions, labels
    warn:            "#F59E0B"   # amber (alerts, "proposed action")
    danger:          "#EF4444"   # red (violence/threat, bounding-box hot state)
    grid_line:       "rgba(52,211,153,0.08)"

  fonts:
    heading: "Space Grotesk, Geist, sans-serif"   # bold, geometric
    body:    "Inter, sans-serif"
    mono:    "JetBrains Mono, monospace"           # for labels, metrics, code, [BRACKET_TAGS]

  motifs:
    - "mono bracketed section labels, e.g. [ 02 // RECAP ]"
    - "subtle scanline / dot-grid background texture (very low opacity)"
    - "targeting-reticle corner ticks on hero images"
    - "metrics shown as large mono numbers with small muted captions"

  transitions:
    default: "fade 250ms"
    section_breaks: "slide-left 300ms"
    build_lists: "fade-up each bullet 120ms stagger"
  rules:
    - "Never more than ~6 lines of text per slide; prefer one diagram + 3-4 short points"
    - "Every model name appears in mono; every metric appears as a big number"
    - "Owner tag [Z]/[F]/[A] shown as a tiny mono chip in the slide footer"
    - "OLD/NEW comparison slides share one consistent two-column layout"
```

---

## SLIDE 1 — Title
**Owner:** [ALL] · **Layout:** Hero / full-bleed

- **Title:** `SURVEILLANT`
- **Subtitle:** Intelligent Mall Surveillance System — Multi-Camera AI Re-Identification
- **Tagline (mono):** `[ GRADUATION PROJECT // SEMESTER 2 — FINAL DEFENSE ]`
- 3 presenter names · Supervised by Dr. Mohamed Salah · Examined by Dr. Mohamed ElShimy · date.

**Figure:** Full-bleed hero — darkened multi-camera grid with emerald bounding boxes + reticle corner ticks. *Position:* full background, dimmed 60%, title centered.
**Notes:** Logo top-left. Slow zoom-in (Ken Burns). Title → subtitle → names fade up in sequence.

---

## SLIDE 2 — Project Vision & Mission
**Owner:** [F] · **Layout:** Split (text left / visual right)

- **Core mission:** an AI surveillance platform for malls & large public spaces that turns every camera feed into a **searchable database of every person detected.**
- **Natural-language interface:** security staff query in plain English — no technical skill required.
- **Primary use cases:** Lost-person search · Real-time threat detection · Demographic analytics · Cross-camera tracking.
- *Say:* "We pitched this vision in Semester 1 — quick reminder, then how far it's come."

**Figure:** [GENERATE] icon trio (camera → person-ID → query/alert) connected by arrows. *Position:* right half.
**Notes:** Section label `[ 01 // VISION ]`. ~60–75s — a reminder, not a re-pitch.

---

## SLIDE 3 — Semester 1 Recap: What We Built  ⟵ CORRECTED
**Owner:** [F] · **Layout:** 2×3 card grid (model cards)

- **Header line:** "Semester 1 = separate, from-scratch models — one task each, no integration."
- Six cards (the S1 deliverables):
  1. **Face Detection** — custom CNN, Haar + deep learning, bounding-box regression.
  2. **Emotion Recognition** — 7 classes (FER2013).
  3. **Age Estimation** — ordinal regression (UTKFace).
  4. **Gender Classification** — binary (CelebA).
  5. **Violence Detection** — video CNN over temporal sequences (Hockey Fight dataset).
  6. **Architecture** — *8 separate models, trained from scratch, **no pipeline / no integration**.*
- *Say:* "Each model worked in isolation on a small dataset. There was no system tying them together — and that's the gap Semester 2 closes."

**Figure:** [GENERATE] 6 model cards, each with a tiny dataset tag in mono (FER2013 / UTKFace / CelebA / Hockey Fight). *Position:* full grid. The "Architecture / No Pipeline" card tinted muted/red.
**Notes:** Section label `[ 02 // S1 RECAP ]`. These are the **real S1 models** — do NOT show the body/tracking pipeline here (that's all S2).

---

## SLIDE 4 — Semester 1: Limitations That Drove Semester 2  ⟵ CORRECTED
**Owner:** [F] · **Layout:** Two-column — limitations (left) / root causes + pivot (right)

- **Limitations (the ✕ list):**
  - ❌ **Face-only** — ignores body, clothing, gait → fails when the face is occluded or turned away.
  - ❌ **Low accuracy** — 60–75% in real conditions; degrades badly with poor lighting, blur, crowds.
  - ❌ **No multi-angle support** — side profiles / partial occlusion → recognition fails.
  - ❌ **No integration** — 8 disconnected models, no shared pipeline.
- **Root causes:** tiny datasets (5K–50K vs. millions) · CPU-only, no training GPUs · first-time CV team.
- **The pivot (bridge to S2):** stop training from scratch → **stand on pre-trained SOTA models** (YOLOv8, OSNet, ByteTrack, InsightFace, local LLMs) + build one integrated pipeline.
- *Say:* "These weren't small bugs — they were structural. So Semester 2 was a redesign, not a patch. Over to Zeiad." → **handoff to [Z]**

**Figure:** [GENERATE] red "✕" checklist (left) that will visually flip to emerald ✓ later (callback at Slide 17). *Position:* left column; root-cause chips bottom-right.
**Notes:** [F]→[Z] handoff, slide-left transition. *(Optional extra slide if time allows: a dedicated "Lessons Learned" slide from your PPTX — insufficient CV theory / no research / coded before planning. Strong honesty moment; add as Slide 4b if [F] wants it.)*

---

## SLIDE 5 — What's New This Semester
**Owner:** [Z] · **Layout:** Grid of upgrade chips + S1↔S2 comparison strip

- "Semester 2 = a rebuilt brain + a product around it."
- **Comparison strip:** S1 (60–75% · 8 models · face-only · trained from scratch) → S2 (pre-trained SOTA · unified pipeline · full-body · instant deploy).
- Chips (mono): `OSNet Re-ID` · `ByteTrack` · `YOLO-seg masking` · `Pose-aware gallery` · `FAISS index` · `Overlap-aware matching` · `LLM description` · `Semantic search` · `Conversational assistant` · `Face + attributes` · `Violence` · `Audio` · `Web command-center`
- *Say:* "Architecture first, then component by component, then the problems we solved getting there."

**Figure:** [GENERATE] chip cloud — S1 items fade out, S2 items light emerald. *Position:* full center.
**Notes:** Section label `[ 03 // WHAT'S NEW ]`. Staggered fade-in. ~90s teaser — don't explain each yet.

---

## SLIDE 6 — System Architecture: The Four Layers
**Owner:** [Z] · **Layout:** Full-bleed diagram (centerpiece)

- **Your 4-layer architecture image goes here.** Layer labels (match to your diagram):
  1. **Ingestion** (24/7) — camera stream → detection → tracking → analysis → description.
  2. **Storage** (24/7) — identity DB + vectors + descriptions + FAISS index.
  3. **Query** (on-demand) — natural-language input → parsing → dual search → ranking.
  4. **Presentation** (on-demand) — React dashboard, live feeds, alerts, analytics, chat.
- *Say:* Walk top→bottom once: "A frame enters, becomes a tracked person, gets identified & analyzed, then surfaces to the operator."

**Figure:** [ASSET — YOUR 4-LAYER IMAGE] *Position:* full-bleed; thin emerald layer dividers + mono layer numbers on the left margin.
**Notes:** Section label `[ 04 // ARCHITECTURE ]`. Layers appear top→bottom (slide-up stagger). This is the map the whole talk hangs on — ~3 min.

---

## SLIDE 7 — How the Pieces Talk: End-to-End Data Flow
**Owner:** [Z] · **Layout:** Horizontal pipeline diagram

- Left→right flow:
  `Camera frame → CLAHE/γ enhance → YOLO-seg (detect+mask) → ByteTrack → Quality gate → OSNet embed → FAISS search → SQLite identity → [Face / Violence / LLM] → Web app / Alerts`
- Callout: **SQLite = source of truth; FAISS = in-memory cache** (a hard system invariant).
- *Say:* "Everything is async — detection never waits on embedding, embedding never waits on the LLM. That decoupling is why it runs on CPU."

**Figure:** [GENERATE] animated pipeline; a "frame token" travels the arrows, lighting each stage. *Position:* full width.
**Notes:** Moving dot along the path; stage boxes in mono. Sets up every deep-dive that follows.

---

## SLIDE 8 — Perception · Detection: YOLOv8-seg
**Owner:** [Z] · **Layout:** Split (concept left / before-after image right)

- **Model:** `YOLOv8n-seg` (segmentation variant).
- **Why segmentation, not plain detection:** outputs a **per-person pixel mask** → we zero out the background before embedding.
- Config (mono): `DETECTION_CONF = 0.10` — pass *all* detections to the tracker; gate later by IoU.
- *Say:* "The mask is the key — it stops two strangers in front of the same wall from looking identical. Detail in the Problems section."

**Figure:** [SCREENSHOT] same crop twice — (a) full box w/ background, (b) background masked to neutral gray. *Position:* right half, stacked before/after.
**Notes:** Emphasize the gray (value 128) masked silhouette.

---

## SLIDE 9 — Perception · Tracking: ByteTrack
**Owner:** [Z] · **Layout:** Concept + timeline diagram

- **Tracker:** `ByteTrack` (two-stage IoU association).
- **Core idea:** keep *low-confidence* detections (0.10–0.45) to coast a track through occlusion instead of killing it.
- Config (mono): `track=0.45 · low=0.10 · match IoU=0.80`.
- *Say:* "This killed the 'person changes ID/color when they turn' bug. Full before/after in Problems."

**Figure:** [GENERATE] two-row timeline: top = old tracker (track dies → new color), bottom = ByteTrack (same color survives occlusion). *Position:* center.
**Notes:** Red = dropped/duplicated ID; emerald = surviving ID.

---

## SLIDE 10 — Intelligence · Re-Identification: The Core Problem
**Owner:** [Z] · **Layout:** Concept

- **The question:** "Is the person on Camera 4 the same one I saw on Camera 1?"
- **Approach:** encode each body crop as a **vector (embedding)**; compare with **cosine similarity** — same person → high, different → low.
- The system's whole accuracy rests on **how good that vector is.**
- *Say:* "So the embedder is the most important single choice in the project — and it's where we made our biggest upgrade."

**Figure:** [GENERATE] two crops → two vectors → cosine-angle gauge (small angle = match). *Position:* center.
**Notes:** Sets up Slide 11 (OSNet) and the headline before/after in Problems.

---

## SLIDE 11 — Intelligence · The Embedder: OSNet x1.0
**Owner:** [Z] · **Layout:** Spec card + intuition

- **Model:** `OSNet x1.0` — **purpose-built for person Re-ID** (ICCV 2019), trained on **Market-1501** (1,501 identities, 6 cameras).
- **Output:** 512-d identity embedding · **Input:** 256×128 (body aspect ratio).
- **Robustness:** **median** aggregation over a 4-frame buffer (one bad frame can't skew identity).
- *Say:* "It learns *identity* (this specific person), not *category* ('a person'). That distinction is everything — numbers on the Problems slide."

**Figure:** [GENERATE] spec card with OSNet omni-scale block icon. *Position:* left. Caption (mono): `512-d · Market-1501 · 256×128`.
**Notes:** Save the before/after numbers for Slide 18.

---

## SLIDE 12 — Intelligence · Pose-Aware Gallery (Multiple Vectors per Person)
**Owner:** [Z] · **Layout:** 4-view diagram

- A person looks different from each side → store **multiple embedding vectors per identity**, one per canonical view: `frontal · left-moving · right-moving · side`.
- **Force-accept** an uncovered view so all angles are captured (guarded by a distance check so it can't accept garbage).
- *Say:* "This is *why* someone identified from the front is still matched later from behind. It's also the direct fix to S1's 'no multi-angle' limitation — more in the Problems section."

**Figure:** [GENERATE / SCREENSHOT] one identity with 4 stored crops labeled by view. *Position:* center row of 4.
**Notes:** Mono view tags under each crop. Cross-reference: this mechanism is the payoff of **Problem #1 (Slide 17)**.

---

## SLIDE 13 — Perception · Preprocessing Hardening
**Owner:** [Z] · **Layout:** Two stacked mini-sections

- **Frame enhancement** — `CLAHE` + auto-gamma, **only on dark frames** (skip when bright → no wasted compute / noise).
- **Quality gate** (gallery path only) — reject crops that are **blurry** (Laplacian var < 50), **tiny** (< 48×96), or **dark** (V-mean < 30).
- *Say:* "A blurry, dark, or tiny crop is noise, not identity. We let it *identify* a person but never let it *join* the gallery."

**Figure:** [GENERATE] before/after dark frame (left) + a "rejected crop" set with red ✕ stamps (right). *Position:* split.
**Notes:** Ties back to the S1 "poor lighting / blur" limitation from Slide 4.

---

## SLIDE 14 — Intelligence · Fast Search: FAISS
**Owner:** [Z] · **Layout:** Metric hero

- **FAISS** in-memory vector index over all gallery embeddings.
- **Big number:** `916× faster` search vs. the linear SQLite scan.
- Invariant: **SQLite = source of truth, FAISS = cache; if they diverge, trust SQLite.**
- *Say:* "Real-time cross-camera matching is only possible because search is near-instant."

**Figure:** [GENERATE] big mono `916×` + small bar chart (linear vs FAISS). *Position:* center hero number.
**Notes:** Single dominant metric. Minimal text.

---

## SLIDE 15 — Intelligence · Overlap-Aware Cross-Camera Matching
**Owner:** [Z] · **Layout:** Camera-topology diagram

- Some cameras **share a field of view** (cam3 ↔ cam4). Same person, opposite angle → similarity legitimately drops.
- Solution: **context-aware triple threshold** by where the candidate was last seen:
  - same-camera `0.75` · cross-camera `0.68` · overlap-pair `0.62`.
- *Say:* "One global threshold can't be right everywhere — we pick it from the camera relationship."

**Figure:** [GENERATE] top-down camera layout; cam3 & cam4 overlap zone highlighted; three threshold chips. *Position:* center.
**Notes:** Section label `[ 04b // CROSS-CAMERA ]`. This is your most advanced matching logic.

---

## SLIDE 16 — Engineering Journey: How I Hardened the System
**Owner:** [Z] · **Layout:** Section divider

- **Big statement:** "Every major decision was driven by a *measured* failure — here's the journey."
- Preview the 6 stories: **Viewing Angles · Embedder · Tracker · Background · Lighting/Quality · Camera Overlap & Calibration.**
- *Say:* "This is the part I'm most proud of — not the final system, but *why* each piece is what it is."

**Figure:** the Slide-4 red "✕" list now visibly **flipping to emerald ✓** (callback payoff). *Position:* center.
**Notes:** Section label `[ 05 // PROBLEMS → SOLUTIONS ]`. Slide-left transition. ~20s, pure momentum.

---

## SLIDE 17 — Problem → Solution #1: Different Viewing Angles → Multi-Vector Gallery  ⟵ NEW (first problem)
**Owner:** [Z] · **Layout:** Before/After two-column

- **Problem:** the same person looks drastically different from front, back, and side. A **single embedding** taken from one angle fails to match the same person seen from another → the system creates a **new identity every time the angle changes** (duplicate IDs, broken cross-camera tracking).
- **Solution:** store **multiple vectors per person — one per canonical view** (`frontal · left-moving · right-moving · side`); match a new sighting against **all** of an identity's stored views and take the best.
- **Impact:** a person enrolled from the front is correctly re-identified from behind or the side; far fewer duplicate identities; cross-camera continuity holds as people turn and walk away.
- *Say:* "This is the very first thing that broke real footage — people don't walk toward the camera politely. One vector per person was never going to work; one vector *per angle* does. The mechanism is the pose-aware gallery I showed earlier (Slide 12)."

**Figure:** [GENERATE] LEFT (old): one frontal vector ✕ fails to match a back-view crop → two different IDs. RIGHT (new): 4 view-vectors under one identity, the back-view crop matches the stored side/back vector → one ID. *Position:* full width, the hero of the slide.
**Notes:** Keep OLD/NEW header chips consistent with Slides 18–22. Animate: front-only fails → views fill in → match succeeds.

---

## SLIDE 18 — Problem → Solution #2: Embedder · ResNet-50 → OSNet  ⭐ KEY SLIDE
**Owner:** [Z] · **Layout:** Before/After two-column with metrics

- **OLD:** `ResNet-50` (ImageNet, 2048-d).
- **Problem:** ImageNet weights encode *semantic category* ("a person"), not *identity*. Same-person & different-person similarities **overlapped** → **no usable threshold**.
  - Same person (front→back): `0.45–0.65`
  - Different people, similar clothes: `0.55–0.75`  ← overlap!
- **NEW:** `OSNet x1.0` (Market-1501 Re-ID, 512-d).
- **Impact:**
  - Same person: `0.78–0.92`
  - Different people: `0.20–0.50`  ← **clean separation → a reliable threshold exists.**
- *Say:* "Single most important fix. Gotcha: torchreid's `pretrained=True` only loads ImageNet — the Market-1501 weights must be fetched separately; without them OSNet is barely better. Cost me a day to discover."

**Figure:** [GENERATE] two overlapping distribution curves (OLD, red) vs. two cleanly separated curves (NEW, emerald). *Position:* full width — slide hero.
**Notes:** Slow down here. Animate overlapping → "click" → separated.

---

## SLIDE 19 — Problem → Solution #3: Tracker · DeepSORT → ByteTrack
**Owner:** [Z] · **Layout:** Before/After + timeline

- **OLD:** `DeepSORT` — discards low-confidence detections.
- **Problem:** when a person is half-occluded or turning, YOLO emits a low-conf box (0.25–0.40). DeepSORT drops it → **track dies → new ID + new box color on reappearance** ("color change on turn").
- **NEW:** `ByteTrack` — two-stage IoU keeps low-conf detections to coast the track through occlusion.
- **Impact:** stable identities through turns/occlusion; fewer fragmented tracks → fewer duplicate person IDs downstream; also dropped DeepSORT's heavy appearance model.
- *Say:* "ByteTrack keeps the track alive through the exact moments DeepSORT gave up on."

**Figure:** refine the Slide-9 two-row timeline (old breaks vs ByteTrack survives). *Position:* bottom half, full width.
**Notes:** OLD/NEW chips consistent with the rest of the section.

---

## SLIDE 20 — Problem → Solution #4: Background · YOLO → YOLO-seg
**Owner:** [Z] · **Layout:** Before/After image pair

- **OLD:** `YOLOv8n` detection only → embedder saw the **whole box**, background included.
- **Problem:** two different people in front of the **same wall** scored artificially **high**; the same person in **different backgrounds** scored artificially **low**.
- **NEW:** `YOLOv8n-seg` → mask background to neutral gray before embedding (~3 ms/frame).
- **Impact:** embedding encodes **the person, not the room**; both failure modes removed.
- *Say:* "Neutral gray, not black — black biases the embedder toward 'absence of color.' Small detail, real effect."

**Figure:** [SCREENSHOT] same two crops with/without masking + similarity scores annotated. *Position:* center pair.
**Notes:** Reuse the masked-silhouette asset from Slide 8.

---

## SLIDE 21 — Problem → Solution #5: Lighting & Crop Quality
**Owner:** [Z] · **Layout:** Two mini before/afters

- **Problem A — dark scenes:** YOLO misses detections; embeddings encode shadow, not identity. → **Fix:** conditional `CLAHE` + auto-gamma (dark frames only).
- **Problem B — bad crops poisoning the gallery:** blur/tiny/dark crops stored as "identity." → **Fix:** 3-check **quality gate** on the gallery path only.
- **Impact:** cleaner galleries, fewer false non-matches, no compute wasted on already-bright frames.
- *Say:* "Both are 'garbage-in' defenses — the gallery is only as good as what we let into it."

**Figure:** [GENERATE] 2×2: dark→enhanced (top), accepted vs rejected crop (bottom). *Position:* center grid.
**Notes:** Callback to Slide 13, here framed problem→fix.

---

## SLIDE 22 — Problem → Solution #6: Overlapping Cameras & Threshold Calibration
**Owner:** [Z] · **Layout:** Problem story + fix

- **Problem A — "gallery sponge":** one global threshold (0.65) sat inside the different-person score range → **one identity absorbed 3 different people**, "teleporting" across cameras in seconds. → **Fix:** raise + **split** thresholds: same-cam `0.75` / cross-cam `0.68`.
- **Problem B — overlapping cameras (cam3↔cam4):** legit same-person drops below cross-cam threshold. → **Fix:** dedicated overlap threshold `0.62` + camera-topology awareness.
- **Impact:** no more identity merging; correct matches across shared-FOV cameras.
- *Say:* "Real debugging — I found it in the dashboard: one person 'moving' between four cameras in seconds, which is physically impossible."

**Figure:** [SCREENSHOT] dashboard — the sponge (one ID, many people) BEFORE → clean separate IDs AFTER. *Position:* before/after split.
**Notes:** Strong engineering-maturity moment — don't rush. Optional threshold table.

---

## SLIDE 23 — Intelligence · Describing People in Natural Language (LLM)
**Owner:** [Z] · **Layout:** Pipeline + example

- **Model:** `Qwen2.5-VL:3b` (vision-language) via local **Ollama**, CPU, ~20–40 s/person.
- Produces a **description of only what's visible** (anti-hallucination): a `long_description` + structured attributes **only when confident** (no guessing "brown pants" from a face-only crop).
- *Say:* "Lesson learned: I first tried a *reasoning* VLM (qwen3-vl) — ~1,800 thinking tokens per image, 400 s+ each on CPU. Switched to a non-reasoning model: same quality, 10× faster, fully offline."

**Figure:** [SCREENSHOT] person crop → generated description card (long text + a few structured tags). *Position:* right half.
**Notes:** Mono caption: `Qwen2.5-VL:3b · local · CPU`. Stress on-prem privacy.

---

## SLIDE 24 — Intelligence · Semantic Natural-Language Search
**Owner:** [Z] · **Layout:** Query → results

- **OLD:** keyword SQL `LIKE` + synonym tables — brittle, missed paraphrases.
- **NEW:** **semantic** search — embed each description with `all-MiniLM-L6-v2`, embed the query, **cosine-rank**.
- **Impact:** "a man in a dark jacket carrying a bag" matches even if those exact words were never stored.
- *Say:* "Search now understands meaning, not just string matches."

**Figure:** [SCREENSHOT] search box with a natural query → ranked person cards. *Position:* center.
**Notes:** Show a real query. OLD/NEW chips consistent with the Problems section.

---

## SLIDE 25 — Intelligence · Conversational Assistant
**Owner:** [Z] · **Layout:** Chat mockup + capability chips

- A **chatbot over the whole system** — ask in plain language; it searches, answers stats, inspects alerts, and can **safely edit the database.**
- **Tiered brain:** Tier 1 = instant offline rules; Tier 2 = LLM fallback for ambiguous phrasing.
- **Safety:** write actions are **proposed, then require confirmation + admin/operator role + audit log** — the bot never edits on its own. Multi-turn memory ("show me the second one").
- *Say:* "This is the human interface to everything we built — natural language in, real surveillance answers out."

**Figure:** [SCREENSHOT] Assistant page: a query, inline result cards, an amber 'Confirm action?' bar. *Position:* center, panel framing.
**Notes:** A "wow" slide — give it room. Reappears in the live demo.

---

## SLIDE 26 — Data Layer · How It's All Stored (and Kept Safe)
**Owner:** [Z] · **Layout:** Schema diagram

- **SQLite (WAL)** = source of truth · **FAISS** = in-memory cache.
- Tables: `persons` · `person_embeddings` · `camera_history` · `merge_proposals` · `face_embeddings` (**isolated**) · `person_descriptions` (**append-only history**).
- **Critical isolation:** face vectors live in their **own table** — face & body are both 512-d and the body searcher pools by *dimension*; mixing them would silently corrupt identity.
- Chat data lives in a **separate** `auth.db`, never polluting identity data.
- *Say:* "Separation of concerns isn't just tidy here — it's a correctness requirement. Over to [F] for the face side." → **handoff to [F]**

**Figure:** [GENERATE] schema — body tables (emerald) vs face table (cyan) vs auth.db (dashed, separate). *Position:* center.
**Notes:** [Z]→[F] handoff. Section label `[ 06 // DATA ]`.

---

# ───────── FACE & VIOLENCE BLOCK ([F], 11 slides) ─────────

## SLIDE 27 — Face Intelligence: From 8 Models to One Pipeline
**Owner:** [F] · **Layout:** Evolution split (S1 ✕ left / S2 ✓ right)

- **S1:** 8 separate from-scratch models (detection, emotion, age, gender, …) — face-only, 60–75%, no integration.
- **S2:** one integrated **face-intelligence layer** built on pre-trained models — `InsightFace (buffalo_l)` + `ethnicity (ResNet18)` + `glasses (CNN)` — feeding the same identity database as the body pipeline.
- *Say:* "Instead of eight brittle models, one robust pipeline. Here's each piece, then the problems I hit making them work on real footage."

**Figure:** [GENERATE] left: 8 scattered boxes (muted); right: one clean stacked pipeline (emerald). *Position:* split.
**Notes:** Section label `[ 07 // FACE ]`. Sets the narrative for the next 10 slides.

---

## SLIDE 28 — Face · Detection & Alignment
**Owner:** [F] · **Layout:** Concept + image

- **Model:** `InsightFace (buffalo_l)` detector — finds faces, returns **5-point landmarks**, and **aligns** each face before analysis.
- Runs **once per new track** (not every frame) — alignment normalizes pose/scale so downstream attribute models see a consistent input.
- *Say:* "Good detection + alignment is the foundation — every attribute below depends on a clean, aligned face crop."

**Figure:** [SCREENSHOT] face with detected box + 5 landmark dots → aligned/cropped face. *Position:* right half.
**Notes:** Mono caption: `buffalo_l · SCRFD detector · 5-pt align`.

---

## SLIDE 29 — Face · Recognition & Embedding
**Owner:** [F] · **Layout:** Spec + matching diagram

- **Model:** `InsightFace` → **512-d face embedding** per face.
- **Identity uses:** cosine match against a **named watchlist** (`known_faces/`) + a **"returning face"** badge for faces seen before.
- Face is an **additive** identity signal — it sharpens identity when a face is visible but **never overrides** body Re-ID.
- *Say:* "Body Re-ID works with no face at all; when a face *is* visible, this adds a second, stronger identity cue."

**Figure:** [GENERATE] face crop → 512-d vector → cosine match to a watchlist entry (name + score). *Position:* center.
**Notes:** Mono caption: `512-d · cosine match · watchlist + returning badge`.

---

## SLIDE 30 — Face · Age & Gender Estimation
**Owner:** [F] · **Layout:** Two readout cards

- From the **same InsightFace pass**: **age** estimate (range) + **gender** classification with confidence.
- One model call yields detection + embedding + age + gender — replacing four separate S1 models.
- *Say:* "What took four from-scratch S1 models now comes free from one pre-trained pass."

**Figure:** [SCREENSHOT] aligned face → age + gender readout cards. *Position:* center, two cards.
**Notes:** Tie back to Slide 27 ("one pipeline, not eight").

---

## SLIDE 31 — Face · Ethnicity Classification
**Owner:** [F] · **Layout:** Spec card + class wheel

- **Model:** `ResNet18` classifier on a 224×224 aligned face crop.
- **6 classes:** Asian · Black · Indian · Latino/Hispanic · Middle-Eastern · White.
- Treated as a **soft demographic descriptor for search**, never as ground-truth identity (see Ethics slide).
- *Say:* "Useful for narrowing a search — 'a Middle-Eastern man in his 30s' — but always a descriptor, never a verdict."

**Figure:** [GENERATE] 6-class wheel/bar with the predicted class highlighted. *Position:* right half.
**Notes:** Mono caption: `ResNet18 · 224×224 · 6 classes`.

---

## SLIDE 32 — Face · Glasses Detection
**Owner:** [F] · **Layout:** Spec card + binary toggle + anecdote callout

- **Model:** CNN classifier (`glasses_classifier.pth`) — binary **Glasses / No-glasses**.
- **Engineering note (callout):** the checkpoint turned out to be a **ResNet-50** nested under a `base.` prefix — not a ResNet-18. The loader now **auto-detects backbone depth**; before the fix it silently matched **0 weights and ran on random init**.
- *Say:* "A great lesson in *verifying* a model loaded correctly — it was 'working' on pure noise until we checked the weight-match count."

**Figure:** [GENERATE] glasses on/off toggle + a small amber callout box for the "0 weights / random init" catch. *Position:* center.
**Notes:** Worth saying aloud — examiners love an attention-to-detail debugging story.

---

## SLIDE 33 — Face · Problem #1: Detection Latency / Frame Buffering
**Owner:** [F] · **Layout:** Before/After (problem→fix)

- **Problem:** face analysis **buffered too many frames** before committing to a decision → high latency, sluggish decisions, backlog under load.
- **Fix:** **reduce frames-per-decision** — collect just **3–5 well-chosen frames** per track and run InsightFace **once per new track ID** (not per frame).
- **Impact:** real-time face analysis with no decision backlog; cost drops dramatically while accuracy holds (multi-pose aggregation over the few frames keeps it robust).
- *Say:* "Buffering more frames felt safer but actually hurt — fewer, better frames per track made it both faster and reliable."

**Figure:** [GENERATE] before: tall stack of buffered frames → slow ⏳; after: 3–5 frames → fast ⚡, once per track. *Position:* split before/after.
**Notes:** OLD/NEW chips consistent with [Z]'s problem slides for visual rhythm.

---

## SLIDE 34 — Face · Problem #2: Pose & Lighting Variation
**Owner:** [F] · **Layout:** Before/After + augmentation strip

- **Problem:** real faces appear at **varied poses and lighting** — side angles, tilt, dim/harsh light — and recognition / age / gender / ethnicity all degraded on non-frontal, poorly-lit faces (the exact S1 "no multi-angle" weakness).
- **Fix:** **data augmentation** applied to the training set — **contrast adjustment, rotation, horizontal flipping** (and more) — so the models generalize beyond clean frontal faces.
- **Impact:** stable predictions across poses and lighting; far fewer misses on profile / off-angle / low-light faces.
- *Say:* "We couldn't control how people face the camera, so we taught the models to expect the variation — augment the data, not constrain the world."

**Figure:** [GENERATE] one face → augmented variants strip (rotated, flipped, contrast-shifted, dimmed). *Position:* center strip.
**Notes:** Connect back to S1 Slide 4's "no multi-angle" ✕ → now ✓.

---

## SLIDE 35 — Face · Integration & Database Isolation
**Owner:** [F] · **Layout:** Mini-architecture diagram

- Face attributes attach to the **same person record** as the body identity — one unified profile per person.
- **Isolation invariant:** face embeddings live in a **separate `face_embeddings` table**, never mixed with body vectors (both 512-d; mixing would corrupt body identity).
- Face **enriches** identity (name, demographics, "returning") but **never drives** cross-camera body matching.
- *Say:* "Face and body cooperate but stay cleanly separated — that boundary is what keeps identity correct."

**Figure:** [GENERATE] person record card pulling from body-vectors (emerald) + face-attributes (cyan), with the isolation boundary drawn. *Position:* center.
**Notes:** Reinforces [Z]'s Slide 26 from the face side.

---

## SLIDE 36 — Behavior · Violence Detection (Model)
**Owner:** [F] · **Layout:** Temporal-model diagram + alert

- **Model:** `CNN-LSTM` — **ResNet-50** backbone + **Bidirectional LSTM** over **16-frame** clips → violence probability.
- **Why temporal:** violence is *motion over time*, not a single frame — the LSTM reads the sequence.
- Runs in its **own daemon thread**; on a hit → JSON log + saved alert clip + optional email.
- *Say:* "This watches *how* people move, not just where they are."

**Figure:** [GENERATE] 16 frames → CNN → BiLSTM → probability gauge; red alert banner above threshold. *Position:* center flow.
**Notes:** Use danger-red accent. Sets up the violence problem next.

---

## SLIDE 37 — Behavior · Violence Problem: Domain Mismatch + Threshold
**Owner:** [F] · **Layout:** Before/After (problem→fix) + threshold dial

- **Problem A — domain mismatch:** trained on **RWF-2000** (street/movie fights) but deployed on **office CCTV** → it gave **wrong decisions on any footage outside the training set**; the two domains simply didn't match.
- **Fix A:** collected real **office surveillance footage**, added it to the training set, and retrained (`violence_detector_v3_office`) → reliable on the actual deployment domain.
- **Problem B — threshold:** picking the **decision threshold** that best separates violence vs. no-violence (too low → false alarms; too high → missed events).
- **Fix B:** tuned the threshold on the office data to balance false alarms against misses.
- *Say:* "A model is only as relevant as its training domain — RWF-2000 looked nothing like an office camera, so we brought the domain to the model." → **handoff to [A]**

**Figure:** [GENERATE] left: RWF-2000 fight clip ✕ wrong on office clip; right: office-augmented training → ✓ correct; a threshold dial below. *Position:* split + dial.
**Notes:** [F]→[A] handoff. Danger-red accent. OLD/NEW chips consistent.

---

# ───────── AUDIO BLOCK ([A]) ─────────

## SLIDE 38 — Audio Intelligence · Overview
**Owner:** [A] · **Layout:** Pipeline diagram

- A fully **isolated, flag-gated** audio modality (never touches body/face identity).
- Pipeline: `audio → STT → language ID → translation → speaker ID → spoken-threat scan`.
- *Say:* "Same philosophy as the rest of the system: additive, isolated, safe to turn off."

**Figure:** [GENERATE] left→right audio pipeline. *Position:* full width.
**Notes:** Section label `[ 08 // AUDIO ]`. Audio colleague tailors the next two slides.

---

## SLIDE 39 — Audio · Speech, Language & Translation
**Owner:** [A] · **Layout:** Demo-result mockup

- **STT:** `Whisper (medium)` — transcribes speech.
- **Language ID** with automatic disambiguation (keeps the more-confident decode for accented/bilingual speech).
- **Translation:** MarianMT / Google fallback → target language.
- *Say:* "It transcribes, detects the language, and translates — even on accented or mixed-language clips."

**Figure:** [SCREENSHOT] Audio page: sample clip → transcript + detected language + translation. *Position:* center.
**Notes:** Audio colleague swaps in their best demo clip result.

---

## SLIDE 40 — Audio · Speaker ID & Spoken Threats
**Owner:** [A] · **Layout:** Two-up

- **Speaker recognition:** `ECAPA-TDNN` embeddings vs. enrolled profiles (cosine match) → "who is speaking."
- **Spoken-threat scan:** multilingual keyword detection (~30 languages) → raises an alert into the shared alerts feed.
- *Say:* "It can say *who* spoke and flag *what* was said if it's threatening." → **handoff to [Z]/[ALL]**

**Figure:** [SCREENSHOT] speaker match (name + score) + a flagged-threat example. *Position:* split.
**Notes:** [A]→[ALL] handoff into Results/Demo.

---

# ───────── CLOSE ([ALL]) ─────────

## SLIDE 41 — Results & Evaluation
**Owner:** [Z] / [ALL] · **Layout:** Metric dashboard

- Headline numbers as big mono tiles:
  - Re-ID separation: same `0.78–0.92` vs different `0.20–0.50` (vs. overlapping ResNet-50 baseline).
  - Search speed: `916×` faster with FAISS.
  - LLM describe: `~20–40 s`/person on CPU (offline).
  - Face: age/gender/ethnicity/glasses + 512-d embedding verified end-to-end.
  - Tracking: stable IDs through occlusion (qualitative before/after).
- *Say:* "Not just features — measured evidence each one works."
- ⚠️ **TODO (fill with your REAL test numbers):** detection rate, # identities tracked across the 5-cam set, violence accuracy on office footage, any precision/recall you logged. Replace estimates with measured numbers before the defense.

**Figure:** [GENERATE] KPI tile grid + the Re-ID separation chart reused from Slide 18. *Position:* full grid.
**Notes:** Section label `[ 09 // RESULTS ]`. **Most important slide to back with real data** — flag for the team to fill.

---

## SLIDE 42 — Live Demo  *(SPEAKER SCRIPT — minimal on-screen)*
**Owner:** [ALL] (Zeiad drives) · **Layout:** Near-empty "DEMO" slide

- On screen: just `LIVE DEMO` + logo (clean projector while switching to the app).
- **Speaker script (NOT on slide):**
  1. Open **Live Cams** → toggle **Overlays** → emerald boxes + IDs.
  2. Show a **cross-camera re-ID** (ideally a cam3↔cam4 overlap match).
  3. Open the **Conversational Assistant** → type a natural query → ranked results.
  4. Open a **person detail** → LLM description + face attributes + embeddings.
  5. Trigger / open an **alert** (violence or spoken-threat) → show it in the feed.

**Figure:** none (live app). **App running + DB seeded before you start.**
**Notes:** Section label `[ 10 // DEMO ]`. Forward = the fallback slide. **Pre-flight:** app running, overlays recorded, Ollama warm, a known-good query + a known alert ready. Keep under ~7 min.

---

## SLIDE 43 — Demo Fallback (if anything hiccups)
**Owner:** [ALL] · **Layout:** Screenshot gallery / GIF

- Pre-captured proof of every demo beat: overlay frame · cross-cam match · assistant answer · person detail · alert.
- *Say:* only if needed — "Here's the same flow captured earlier." Keep talking, no dead air.

**Figure:** [SCREENSHOT / GIF ×5] one per demo step. *Position:* 2–3 per row, or an auto-playing GIF.
**Notes:** **Insurance slide** — never shown if live works. Build it anyway; juries forgive a hiccup, not a freeze.

---

## SLIDE 44 — Ethics, Privacy & Limitations
**Owner:** [ALL] · **Layout:** Two-column (honest)

- **Privacy:** runs **fully on-prem / offline** (local Ollama, local DB) — no cloud upload of footage.
- **Bias:** demographic classifiers (ethnicity, age, gender) carry dataset bias → used as **soft search descriptors, never ground-truth identity or automated decisions.**
- **Consent & scope:** intended for **authorized** security/forensic use on owned premises.
- **Limitations (own them):** CPU-bound throughput; validated on WiseNet + office footage; spatio-temporal camera constraints (Part 9) not yet implemented.
- *Say:* "Surveillance tech demands responsibility — here's where we're careful and where the boundaries are."

**Figure:** [GENERATE] split icon panel: shield/lock (privacy) vs scale/balance (bias). *Position:* center.
**Notes:** Section label `[ 11 // ETHICS ]`. Pre-empts the toughest Q&A — say it confidently.

---

## SLIDE 45 — Conclusion & Future Work
**Owner:** [Z] · **Layout:** Recap + roadmap

- **Delivered:** a rebuilt Re-ID brain (OSNet / ByteTrack / YOLO-seg / FAISS), an integrated face + violence + audio intelligence layer, an LLM description+search layer, a conversational assistant, and a full web command-center.
- **Future work:**
  - **Part 9 — spatio-temporal camera constraints** (use transit-time between cameras to reject impossible matches). *Next.*
  - Multi-snapshot consensus + hybrid photo+text re-rank.
  - GPU deployment for real-time LLM description.
- *Say:* "The architecture is built so each of these drops in without touching what already works."

**Figure:** [GENERATE] roadmap timeline: ✅ S1+S2 → 🔲 Part 9 → 🔲 consensus/rerank → 🔲 GPU. *Position:* bottom timeline.
**Notes:** Section label `[ 12 // CONCLUSION ]`. End forward-looking.

---

## SLIDE 46 — Thank You / Q&A
**Owner:** [ALL] · **Layout:** Hero close

- `SURVEILLANT` · "Thank you — Questions?"
- 3 presenter names + supervisor + acknowledgements + (optional) repo link.
- *Say:* invite questions; keep the demo app open in case of "can you show…".

**Figure:** reuse the Slide-1 hero (visual bookend), dimmed. *Position:* full background.
**Notes:** Keep the live app one Alt-Tab away during Q&A. Slow fade in.

---

## APPENDIX (optional backup slides — only if asked in Q&A)

> Not presented; keep at the end for tough questions.

- **A1 — Full threshold table:** same-cam 0.75 / cross-cam 0.68 / overlap 0.62 / force-accept guard 0.35 / auto-merge 0.82 (+ rationale).
- **A2 — Async architecture:** detection / embedding / description as decoupled queues; why it survives on CPU.
- **A3 — Reconciliation worker:** mean-pool similarity, view-coverage gate, merge proposals & audit trail.
- **A4 — WiseNet dataset details:** 5 cameras, synchronization, overlap topology {3,4}.
- **A5 — Why local LLM:** privacy + cost vs. cloud; reasoning-VLM CPU failure (qwen3-vl) and the fix.
- **A6 — Violence training data:** RWF-2000 → office-footage augmentation; threshold tuning curve.

---

### BUILD CHECKLIST (before handing this to the slide generator)
- [ ] Drop in **your real 4-layer architecture image** (Slide 6).
- [ ] Capture screenshots called out as `[SCREENSHOT]` from your running app.
- [ ] Fill **Slide 41 Results** with your **measured** numbers (don't ship placeholders).
- [ ] Generate the `[GENERATE]` diagrams (or let the AI generator create them from the descriptions).
- [ ] Record the **Slide 43 fallback** GIF/screenshots from a known-good run.
- [ ] Rehearse the **Slide 42 demo script** end-to-end on the actual machine.
- [ ] Confirm the stack story is the **built** system (ByteTrack/SQLite/CNN-LSTM/Qwen2.5-VL) — NOT the older planning-deck stack (DeepSORT/PostgreSQL/SlowFast/Llama).

### PACING NOTE
v2 is ~46 slides; with a ~7-min live demo this runs ~48–50 min of content. To hit 45 exactly: keep the face attribute slides (30–32) at ~30s each, and/or merge two [Z] deep-dive slides (e.g. 10+11, or 13 into 21). The face block is intentionally large per the team's request.
