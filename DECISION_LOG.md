# SURVEILLANT — Decision Log
## Full Record of Ideas, Implementations, Replacements, and Reasoning

> Every significant change goes here. This file is the primary source for writing
> project documentation, reports, and academic write-ups.
> Format: newest entries go at the top within each part.

---

## How to Read This File

- Each entry has: **What**, **Why**, **Old approach (if replaced)**, **New approach**, **Result/Status**.
- Part numbers match `ENHANCEMENT_PROPOSAL.md`.
- Phase numbers match `SURVEILLANT_BUILD_SPEC.md`.

---

## Current Implementation Status

| Part | Title | Status |
|---|---|---|
| Phase 1 | Camera + Detection + Tracking + Display | ✅ Complete |
| Phase 2 | Embedding + Database + Cross-Camera Matching | ✅ Complete |
| Phase 3 | Photo Search | ✅ Complete |
| 3.5 | Preprocessing Hardening | ✅ Complete |
| Part 1 | Crop Quality Gate | ✅ Complete |
| Part 2 | CLAHE + Auto-Gamma Frame Enhancement | ✅ Complete |
| Part 3 | (IR/VI-ReID — out of scope, no IR hardware) | ⛔ Skipped |
| Part 4 | Background Isolation via YOLO-seg masking | ✅ Complete |
| Part 5 | OSNet x1.0 Embedder (replaces ResNet-50) | ✅ Complete |
| Part 6 | Pose-Aware Gallery (canonical views) | ✅ Complete |
| Part 7 | ByteTrack (replaces DeepSORT) | ✅ Complete |
| Part 8 | FAISS In-Memory Vector Index | ✅ Complete |
| Part 8.5 | Overlap-Aware Matching (camera topology, prelude to Part 9) | ✅ Complete |
| Part 9 | Spatio-Temporal Camera Constraints | 🔲 Not yet implemented |
| Part 10 (Phase 4A+4B) | LLM Body Description + Natural-Language Search | ✅ Complete |
| Part 10 (Phase 4C/4D) | Multi-snapshot consensus, color sanity-check, re-description triggers, hybrid photo+text rerank | 🔲 Deferred |
| Part 11 | Face & Violence integration (InsightFace + glasses + ethnicity + CNN-LSTM violence) merged from team branch — ADDITIVE | ✅ Code complete (awaiting team `.pth` weights) |
| Part 13 | Audio intelligence (Whisper STT + language ID + translation + ECAPA speaker-ID + spoken-threat) — ADDITIVE | ✅ Integrated (file-based webapp + CLI; needs audio deps installed) |
| Track A | Live-Cams Overlay Recorder (server-side burned-in detection boxes) | ✅ Complete |
| Track B | Conversational Assistant (tiered brain, multi-turn memory, read+write) | ✅ Complete |
| Track C | Public marketing / academic-showcase site (no-login landing pages) | ✅ Complete |

---

## Description richness + Assistant accuracy/perf pass — 2026-06-21

> Two operator-facing quality fixes: (1) the VLM body descriptions were too
> short (one line); (2) the chat assistant misrouted some queries and felt slow.

### D1: Longer, still-strict body descriptions (`modules/llm/describer.py`)
**What:** `USER_PROMPT_DESCRIBE` now demands a RICH paragraph of **≥4–6
sentences** (was "1–3 sentences"), with explicit instruction on HOW to be long
WITHOUT guessing — elaborate on the exact shade / cut / fit / layering / pose /
orientation / frame position of what is *actually visible*, and note when parts
are out of frame. The 100%-certainty / anti-hallucination rule is restated and
strengthened ("being long is required, but being WRONG is far worse than being
short"). Example replaced with a 5-sentence chest-up paragraph that models the
"hands/legs out of frame → not mentioned" behaviour. The quality-mode self-verify
prompt (`USER_PROMPT_VERIFY`) now says keep it rich, only strip the unverifiable.
**Why:** A one-sentence description under-extracts the genuine, certain detail in
the crop — poor for both the operator and the semantic-search embedding.
**Also:** added `num_predict: 768` to the describe/verify Ollama options so the
longer JSON is never truncated mid-object (which would fail to parse).

### D2: Assistant — fixed the search→edit misroute (`chatbot/brain.py`)
**Bug:** "find anyone with age 55" / "find a man with age range from 40 to 55"
were hijacked into an **attribute-EDIT proposal** ("Set age_range=45-55 on person
…?"). `_parse_attr_edits` matched the bare phrase "age 55" and `_resolve_one`'s
sole-result fallback bound it to whatever person was left over from the prior turn.
**Fix:** attribute edits now require an explicit mutation verb (`_EDIT_VERB`:
set/change/rename/mark/update/edit/correct/assign/"call it"/"make it"). Without
one, age/gender/ethnicity phrases are treated as SEARCH terms, not edits.

### D3: Assistant — age-RANGE search (`chatbot/tools.py`)
"age range from 40 to 55" / "between 18 and 25" / "40-55 years" previously
extracted **no** age filter (only `age <number>` was parsed). Added range
detection + `buckets_for_age_range(lo,hi)` returning every bucket that OVERLAPS
the span (correct, since the DB stores coarse buckets not exact ages).

### D4: Assistant — smarter empty results (`chatbot/tools.py` `facet_hint`)
When a demographic search matches nobody (e.g. "show all White people" but the
data has no `White` class), the bot now reports which values DO exist
("Ethnicities on record: Middle Eastern (4), Asian (3)…") instead of a flat
"Sorry, I don't have any." Note: ethnicity/gender matching was already correct —
the classifier stores `White`/`Middle_Eastern` and `extract_filters` maps to those;
the miss was a real data gap, now made transparent.

### D5: Assistant — stronger role/prompt + perf guardrails (`chatbot/brain.py`)
- `_ASSISTANT_ROLE`: a real command-center operator persona; the reply-polish and
  Tier-2 prompts now carry ABSOLUTE rules — preserve every fact/number/ID/camera
  exactly, keep the suggestion, never invent data, never claim an untaken action.
- Reply polish is now gated by `CHAT_LLM_POLISH` (settings, default True) and uses
  `CHAT_LLM_TIMEOUT_SEC` (default 15 s, was a hard 25 s) so a slow/stalled Ollama
  never freezes the assistant — it falls back to the (well-formed) deterministic
  reply. Polish temp lowered 0.3→0.2 for steadier phrasing.
- Hardened `_detect_lookup`: a message with search filters or a search verb (and
  no explicit `P:id`) can no longer be hijacked into "open the sole leftover
  result" — it routes to search.
**Verified:** all four files byte-compile; regex/bucket logic unit-checked
(age-55 → search not edit; "40 to 55" → [30-45,45-55,55-70]; "set age to 55" →
edit; "rename it to Ali" → edit).

---

## Part 13 — Audio Intelligence Integration — 2026-06-19

> Integrates the team's audio part (`Grad_Project_Audio_Part`) — a single
> `audio_processor.py` pipeline — as a new ADDITIVE, ISOLATED, flag-gated
> modality, mirroring the Part 11 face/violence pattern. Audio NEVER influences
> body/face identity or tracking; it produces transcript + language + translation
> + speaker + spoken-threat only.

### P13-A: Engine module (`surveillant/modules/audio/`)
**What:** Vendored `audio_processor.py` (ECAPA-TDNN speaker recognition, Whisper
large-v3 STT, multi-engine language ID, MarianMT/Google translation, multilingual
spoken-threat keyword scan, pyaudio mic) and made it **import-safe**:
- Hard top-level `sys.exit` imports (pyaudio / langdetect / deep-translator) →
  soft optional imports with `_HAS_*` flags; `FORMAT`/`AudioCapture` guard pyaudio.
- All reachable `sys.exit(...)` (whisper/speechbrain/azure dep checks) →
  `raise RuntimeError(...)` so they can't kill a web-server worker.
- `SpeakerRecognizer._check_deps()` deferred from `__init__` to first embed, and
  `AudioProcessingPipeline` builds `AudioCapture` lazily — so file-based analysis
  constructs/runs with **no microphone and no speechbrain**.
- Email recipient is no longer hard-coded; aligned to `SURVEILLANT_ALERT_*` env.
Added `analyzer.AudioAnalyzer.analyze_file(wav) -> dict` (clean normalised result)
+ `availability()` (reports missing packages). `__init__.py` exports both.
**Verified:** imports cleanly with ZERO heavy deps present; `availability()`
correctly reports `missing_core=["whisper"]`, `missing_speaker=[...]`.

### P13-B: Settings (`config/settings.py`)
`ENABLE_AUDIO_ANALYSIS=False` (default OFF, graceful) + `AUDIO_WHISPER_MODEL`
(default "small" for CPU responsiveness; large-v3 selectable), `AUDIO_TARGET_LANG`,
`AUDIO_TRANSLATE`, `AUDIO_SPEAKER_ID`, `AUDIO_VIOLENCE_CHECK`,
`AUDIO_SIMILARITY_THRESHOLD`, `AUDIO_SPEAKER_PROFILES` (→ bundled ECAPA pkl).
Assets vendored: `models/audio/speaker_profiles_ecapa.pkl` +
`webapp/api/data/audio_samples/*.wav` (12 demo clips).

### P13-C: Webapp API (`webapp/api/routers/audio.py`)
`get_audio_analyzer()`/`audio_availability()` added to `engine.py` (lazy, cached).
Endpoints (auth-gated except sample playback): `GET /audio/status`,
`GET /audio/samples`, `GET /audio/samples/{name}` (public, path-guarded, for
`<audio>`), `POST /audio/analyze` (upload OR bundled sample → normalised result).
Unavailable → **503 with the missing-package list**, never a 500. On a spoken
threat, appends to the shared `DEMO_ALERTS_LOG` (`source:"audio"`) so it surfaces
on the Alerts page. Sync endpoint (FastAPI threadpool) since Whisper blocks.
**Verified:** app builds with all 4 routes; availability degrades gracefully.

### P13-D: Frontend + public site
New console page `pages/Audio.tsx` (`/app/audio`, nav "Audio"): demo-sample list
with inline players, file upload, and a result panel (transcript / language /
speaker / translation / threat). Public `site/Modules.tsx`: "Speech & Translation"
and "Speaker Recognition" flipped to **live**; "Threat Detection" notes spoken
threats. **Verified:** all modules transform cleanly in Vite.

### P13-E: Windows / modern-stack runtime fixes (2026-06-19)
Verified end-to-end (sample j2 → speaker **Joe** @0.99, transcript, `pl`→en
translation, no-threat). Fixed five real bugs that only surface once the heavy
deps are installed on a plain Windows box with bleeding-edge versions
(torch 2.11, speechbrain 1.1, transformers 5.9, whisper 2025):
1. **Emoji prints crash on cp1252** → force UTF-8 on stdout/stderr at module load.
2. **No ffmpeg + torchaudio 2.9 needs TorchCodec** → load all audio via
   `soundfile` (`_load_audio_mono`, linear resample). No ffmpeg/torchcodec needed
   for the bundled WAVs; whisper gets a numpy array, ECAPA a soundfile tensor.
3. **librosa unusable after speechbrain import** → speechbrain 1.x registers a
   broken `k2_fsa` lazy module (no k2 wheels on Windows) whose attribute access
   raises ImportError, poisoning `inspect.stack()` which librosa's lazy_loader
   calls. Dropped librosa from the load paths (soundfile instead) AND neutralise
   the `k2_fsa` lazy module with a benign stub before loading ECAPA.
4. **WinError 1314 (symlink privilege)** → `HF_HUB_DISABLE_SYMLINKS=1` +
   `LocalStrategy.COPY` so HF/speechbrain copy model files instead of symlinking.
5. **Translation**: transformers 5.x moved MarianTokenizer → MarianMT import
   fails, falls back to Google Translate (works online). Acceptable; MarianMT can
   be restored by pinning transformers if offline translation is required.
Also added `AUDIO_FAST_DECODE` (greedy, no word-timestamps) — ~7 s/clip on CPU.

### P13-F: Accuracy fix — anti-aliased resampling + model upgrade (2026-06-19)
**Symptom:** transcripts were wrong/hallucinated (e.g. j2 → Polish "Dobry
morning…"; language flip-flopped pl↔ar across runs).
**Root cause (two compounding bugs):**
1. **Resampling.** The demo clips are **48 kHz**; Whisper/ECAPA need 16 kHz. The
   loader downsampled with naive linear `np.interp` — at a 3:1 ratio that aliases
   and turns clean speech into mush. Fixed: `_load_audio_mono` now uses
   `scipy.signal.resample_poly` (polyphase, anti-aliased); linear only as a
   no-scipy fallback.
2. **Model too small.** Even with correct resampling, `small` mis-detects the
   language on short/accented clips. `medium` transcribes them perfectly.
   Default `AUDIO_WHISPER_MODEL` changed `small` → `medium` (~15-35 s/clip greedy
   on CPU; large-v3 selectable for max accuracy).
**Verified (medium, full analyzer, greedy):** j1 "What a weather today the Sun is
shining"; j2 "Good morning, good morning. How are you my friend?"; m1 "Arsenal is
my favourite team in the Premier League."; m3 (ar) "مرحباً، اسمي مو … جميل جداً
اليوم". Speaker ID Joe/Mohamed @0.99 throughout. (NB: the team's earlier attempts
to "improve it" were blocked by the resampling bug — no model size could fix it.)

### P13-G: Language override (2026-06-19)
Whisper's audio language detection is inherently ambiguous for accented/bilingual
speech — e.g. m3 is spoken English but auto-detected Arabic (ar 55% vs en 31%),
so it was transcribed into Arabic script (then Google-translated back to English).
Added a per-request **force-language** path: `AudioAnalyzer.analyze_file(..., force_lang)`
sets `stt.force_lang` before the run; `POST /audio/analyze` takes a `language` form
field (""/"auto" = detect); the Audio page has a **Language** dropdown (Auto · EN ·
AR · FR · DE · ES · IT · RU · TR · ZH). Verified: m3 forced `en` →
"Hello, my name is Mo. The sun is very beautiful…".

### P13-H: Automatic language disambiguation (2026-06-19)
The manual override (P13-G) needed a click per clip. Made it automatic:
`SpeechToText.transcribe` now, when **not** force-lang'd, decodes the detected
language AND a preferred language (`AUDIO_DISAMBIGUATE_LANG="en"`, when it differs)
and keeps whichever Whisper is **more confident** about (higher avg_logprob). The
correct language reliably wins; a genuinely Arabic/French/… clip still wins in its
own language (safe), at the cost of one extra decode only for mis-detected clips.
Also corrects the reported `language_probs` to the winning language so the UI label
matches the text. **Verified on the team's whole `m` set** (all accented English
mis-detected as ar/de): m3/m4/m5/m6 now auto-resolve to correct English transcripts
AND `en` labels with no user action. The dropdown remains as an explicit override.

### Status / next
Code complete + import-safe + **verified accurate end-to-end** (medium), with
automatic language disambiguation + a manual override. To run: `pip install -r
surveillant/modules/audio/requirements.txt`, set `ENABLE_AUDIO_ANALYSIS=True`,
restart the API. (Live-mic "Both" mode: the original CLI in `audio_processor.py`
already records from the mic — `python -m surveillant.modules.audio.audio_processor`;
a webapp live path is the next step.)

---

## Track C — Public Marketing / Academic-Showcase Site — 2026-06-18

> A public, **no-login** front for the website that presents the project's idea,
> capabilities and modules to visitors (professors / general audience). The
> authenticated dashboard is unchanged — only *remounted* under `/app`.

### TC-A: Routing split (public site at `/`, console at `/app`)
**What:** `App.tsx` now serves a public `SiteLayout` at `/`, `/modules`,
`/architecture`, `/ethics`, `/team`; the existing console moved from `/`, `/cameras`…
to `/app`, `/app/cameras`…. Only mechanical path edits to the console plumbing:
`Layout.tsx` nav hrefs, `Login.tsx` post-login redirect (`/`→`/app`),
`AuthContext.logout` (→ `/`). **No dashboard design/behavior change.**
**Why:** Standard product structure — visitors land on the marketing site; signing
in opens the console.

### TC-B: Emerald-Sentinel design system, fully isolated
**What:** New `src/site/` module (`SiteLayout`, `Home`, `Modules`, `Architecture`,
`Ethics`, `Team`, shared `ui.tsx`) styled by a **scoped** `site.css` — every class is
prefixed `site-*` and the whole tree is wrapped in `.site-shell`, so it cannot affect
the dashboard's `index.css` tokens. Fonts (Geist + JetBrains Mono) load via `<link>`
in `index.html` (unused by the dashboard, so harmless). Theme matches the Stitch
"Emerald Sentinel" designs: deep-charcoal dot-grid base, emerald/cyan neon accents,
glassmorphism cards, mono bracketed labels, targeting reticles, scanlines.
**Content:** Drawn from real project facts (OSNet/ByteTrack/FAISS Re-ID, InsightFace,
CNN-LSTM violence, Qwen2.5-VL description + MiniLM semantic search). Modules page marks
each of the 10 modules **live vs planned** honestly. Team page uses editable placeholders.
**Verified:** all 5 public pages render (computed styles confirm Geist/charcoal/glass/
emerald), `/app` correctly redirects to `/login` when unauthenticated, 0 console/server errors.

---

## Track B — Conversational Assistant — 2026-06-18

### TB-A: Chatbot tool layer (`webapp/api/chatbot/tools.py`)
**What:** Thin wrapper module exposing the engine's DB + services as named "tools": `extract_filters`, `search`, `get_person`, `stats`, `alerts`, `resolve_person_id`. Filter extraction combines the engine's own `QueryParser._rule_based_fallback()` (Tier-1, instant, offline) with extra chat-specific patterns for age buckets (`"9 years old"` → `"7-10"`), child/teen/young/elderly word sets, camera number, ethnicity, status, and `named X`.
**Why:** Centralising filter logic here means the chatbot reuses the exact same age-bucket table, status words, and appearance keys the REST search routes use. No parallel logic that can drift.
**Result:** ✅ All read tools validated; `extract_filters("find a child about 9 wearing a red t-shirt")` correctly produces `face.age_range=["0-3","4-6","7-10"]` + appearance clothing colour.

### TB-B: Shared corrections service (`webapp/api/corrections_service.py`)
**What:** Extracted `merge`, `delete_person`, `edit_attributes`, `redescribe`, `split`, `run_descriptions` from `routers/corrections.py` into a standalone module. Both the REST router and the chatbot call these functions. Each goes through the engine's invariant-safe `Database` methods + `audit_log` + `engine.invalidate_search_caches`.
**Why (invariant compliance):** The chatbot's write actions must go through the same audit trail and FAISS cache-invalidation path as the REST corrections endpoint. Sharing one code path makes this automatic — no risk of the chatbot bypassing audit or leaving a stale FAISS index.
**Old approach:** All correction logic was inlined in `corrections.py`. Chatbot would have needed its own parallel copy.
**New approach:** `corrections_service.py` (pure functions on `TenantCtx`); `corrections.py` routes delegate to it.
**Result:** ✅ REST routes unchanged (regression-safe); chatbot confirmed to use same path.

### TB-C: Tiered NLU brain (`webapp/api/chatbot/brain.py`)
**What:** Two-tier intent router. **Tier 1** (default, instant, offline): regex intent classifier (greeting / help / thanks / stats / alerts / write-detection / lookup / search/refine / unknown) + `tools.extract_filters` for structured filter extraction. **Tier 2** (optional, Ollama): single `qwen2.5vl:3b` JSON call for messages Tier 1 can't classify; times out → falls back to Tier-1 clarification.
**Design decisions:**
- **Greeting has no DB side-effect** — replies immediately with no search (verified: "hi" → reply, no results).
- **Write actions are PROPOSED, never executed by the brain** — returns `proposed_action` + `pending_action` dict; execution only on a follow-up affirmative turn after a role check.
- **Refinement detection**: if `active_filters` is non-empty and the message starts with `only/just/also/and/make it/on camera/…` (or has filters but no explicit search verb), the filters are MERGED into the running state rather than replaced.
- **Reference resolution**: `_resolve_one`/`_resolve_two` understand ordinals (`"first"`, `"second"`), explicit IDs (`"P:3601b2"`), and pronoun-like references (`"them"`/`"both"` for the last results list).
**Why tiered:** Tier-1 handles greetings, searches, and stats with zero latency and zero Ollama dependency — safe on CPU-only setups where Ollama may be slow or down. Tier-2 only fires for genuinely ambiguous phrasing, with a hard 25 s timeout.
**Result:** ✅ All intent paths verified via smoke tests and regression.

### TB-D: Conversation memory store (`webapp/api/chatbot/store.py`)
**What:** SQLite-backed session store in `AUTH_DB_PATH` (the webapp's own `auth.db`, never the engine DB). Two tables: `chat_sessions` (per session: `active_filters`, `last_results`, `pending_action`) and `chat_messages` (full transcript). In-memory ephemeral by default; SQLite ensures restart-survival.
**Invariant:** Chat data is stored in the AUTH db (not the engine's `surveillant.db`) so identity data is never polluted by conversation metadata. This mirrors the audit_log and user tables that already live in `auth.db`.
**Result:** ✅ Session persistence verified — history survives across three round-trips; 6-message history confirmed by `GET /chat/sessions/{id}/messages`.

### TB-E: Write-action confirmation guard (`webapp/api/chatbot/router.py`)
**What:** Any write action proposed by the brain (`pending_action` in session) is only executed when the user's NEXT message passes `brain.is_affirmative()` AND the caller's role is in `{admin, operator}`. A negative turn clears the pending action without executing. The brain never executes writes directly.
**Why:** Write via chat is powerful; the confirm step + role gate + audit log are the safety guardrails. A viewer-role user gets a clear "requires admin or operator role" message instead of an error.
**Result:** ✅ Router implemented; role gate enforced on execution path.

### TB-F: Chat endpoint (`POST /chat`, `GET /chat/sessions/{id}/messages`)
**What:** Two endpoints mounted at `/chat` in `main.py`. `POST /chat` accepts `{session_id?, message}`, routes through the brain, persists state, executes confirmed writes. `GET /chat/sessions/{id}/messages` returns the transcript for page reloads.
**Result:** ✅ Endpoints smoke-tested end-to-end (greeting, search, stats, write-propose, history).

### TB-G: Assistant frontend page (`webapp/web/src/pages/Assistant.tsx`)
**What:** New React page at `/assistant`. Features: bot/user message bubbles with markdown-lite rendering (bold, bullet lists); loading spinner while waiting; inline `PersonCard` grid when results are returned; amber action-proposal bar with Confirm/Cancel buttons when a `proposed_action` is present; auto-open `PersonDrawer` when the brain resolves a lookup; suggestion chips on the empty state; "New chat" button to clear session; session persistence via `sessionStorage`.
**Design:** Uses the existing `PersonCard`, `PersonDrawer`, `useAuth` and design tokens (ink-700/800, emerald, etc.) — no new design-system tokens introduced.
**Result:** ✅ TypeScript compiles clean (0 errors after two minor JSX fixes: escaped curly-quote in placeholder, named vs default PersonDrawer import). Sidebar nav entry added; route registered in App.tsx.

---

## Track A — Live-Cams Overlay Recorder — 2026-06-18

### TA-A: Offline recorder script (`webapp/api/tools/record_overlays.py`)
**What:** One-time offline script that runs the existing `PersonDetector` + `PersonTracker` + `PersonSearcher` over every demo camera video in sequence, and for each frame emits a JSON sidecar (`webapp/api/data/demo/overlays/cam{i}.json`) keyed by frame index. Each frame entry contains `[{bbox, track_id, person_id, state, status, gallery_size, name, gender, age_range}]`. Track→person_id mapping is cached once per track (searched against the seeded gallery).
**Why sidecar, not engine DB tables:** Invariant #4 — never mutate the engine's SQLite schema. Overlay metadata is an artifact of the webapp's presentation layer, not a core identity signal. JSON sidecar in a webapp-owned directory respects the boundary.
**Sync guarantee:** The recorder reads frames `0..N` in order; the MJPEG endpoint reads in order too and uses `frame_idx % total_frames` for the modulo — boxes line up exactly with playback.
**Fallback labelling:** Tracks with no confident gallery match (sim < threshold) are labelled `T{track_id}` (shown in collecting/unverified style), not dropped.
**Result:** ✅ All 5 cameras recorded. Cams 0–2 have moderate traffic; cams 3–4 heavy (517 kB / 568 kB sidecars). Verified: 2 of 3 tracks on cam0 resolved to real `person_id`s from the demo DB.

### TA-B: Shared draw helper (`surveillant/display/overlay_draw.py`)
**What:** Factored the per-track box-drawing logic (lines ~207–297 of `visualizer.py`) into a standalone `draw_tracks(frame, tracks, cam_id, color_registry) → np.ndarray`. `GridDisplay.update()` now calls this function (no behavior change for the desktop app). The MJPEG endpoint also calls the same function — single source of truth for box style, label format, state colors, and Part-11 attribute bits.
**Why:** Ensures the webapp's live view renders EXACTLY the same box style as the desktop display. Any future style change (label format, color scheme, border) needs to be made in one place.
**Result:** ✅ Visually verified — rendered an overlay frame (cam0 frame 277) with red box + `P:3601b2 [*][G:1] | Male | 20-30 | Mi…` label; style matches the desktop display.

### TA-C: Overlay-aware MJPEG endpoint (`webapp/api/routers/cameras.py` + `webapp/api/overlays.py`)
**What:** New `webapp/api/overlays.py` loads and memo-caches each camera's sidecar. Modified `_mjpeg(...)`: after `cap.read()`, checks `?overlay=1` query param; if set, looks up `frames[str(idx % total)]` from the sidecar and calls `draw_tracks()` before JPEG encode. Process-wide singleton `ColorRegistry` ensures stable colors across cameras. Default `overlay=False` keeps current behavior when the toggle is off.
**Result:** ✅ Backend serving overlays; sidecar cache working.

### TA-D: Frontend overlay toggle (`webapp/web/src/pages/LiveCams.tsx`)
**What:** Added an "Overlays" toggle (default on) to the Live Cameras page. `streamUrl(cam_id, overlay)` appends `&overlay=1` when on. Replaced the "placeholder mode" banner with a short legend (🔲 Collecting, 🟩 Identified, ⬛ Returning). Box style matches the desktop display (deterministic colors from `ColorRegistry`).
**Result:** ✅ Toggle live; overlay URLs confirmed correct.

---



## Part 11 — Face & Violence Integration (merged from team branch) — 2026-06-16

> Merge of the team's last-semester face-side models into the current system.
> Their copy had branched from an OLDER version (DeepSORT + MobileNetV3 body
> embeddings), so ONLY the face/violence work was lifted; their regressed body
> stack was discarded. The current ByteTrack + OSNet body pipeline is unchanged.

### What was added
- **Face analysis** (`modules/face/face_analyzer.py`): InsightFace (buffalo_l)
  face detection + 512-d embedding + age/gender, a named watchlist
  (`data/known_faces/`), and a "returning face" badge (`data/facesfromvid/`).
- **Glasses** (`modules/detection/glasses_detector.py`) and **ethnicity**
  (`modules/face/ethnicity_classifier.py`) ResNet18 classifiers.
- **Violence detection** (`modules/violence/violence_detector.py` + `alerts.py`):
  CNN-LSTM (ResNet50 + BiLSTM) over 16-frame clips, in its own daemon thread,
  with JSON log + alert clip + optional email.
- **Face-image search**: `modules/face/face_searcher.py` + `main.py --phase 3
  --face-photo IMG` — "find this person by their face".
- **DB**: new ISOLATED `face_embeddings` table; new `persons` columns
  `name`/`ethnicity`/`glasses`; methods `add_face_embedding`,
  `get_all_face_embeddings`, `update_person_attributes`.
- **Config**: `ENABLE_FACE_ANALYSIS`, `ENABLE_VIOLENCE_DETECTION` (+ model paths
  under `models/`, thresholds, SMTP-from-env). All default OFF.

### Why (key design decisions)
- **Additive only** (user's explicit choice): face/violence contribute
  attributes, watchlist names, "returning" badges, alerts, and a face-search
  path — but cross-camera BODY identity stays 100% OSNet/ByteTrack driven.
  Nothing in this layer binds, merges, or re-scores a `person_id`.
- **Face embeddings are ISOLATED in `face_embeddings`, never in
  `person_embeddings`.** Critical reason: InsightFace vectors are 512-d — the
  SAME dimension as OSNet body — and the body searcher / FAISS / reconciliation
  pool by DIMENSION, not by `embedding_type`. Mixing them would silently corrupt
  body identity. `add_face_embedding` also does NOT fire the `on_embedding_added`
  FAISS hook, so face vectors never reach the body index.
- **Default OFF + graceful disable**, mirroring `ENABLE_DESCRIPTION_WORKER`:
  with flags off the system behaves byte-for-byte as before. Each model
  auto-disables (warn + no-op) when its `.pth` is missing.
- **No committed secrets**: the team's copy hardcoded a live Gmail app password;
  email creds now come from environment variables only (empty → email disabled).

### Old approach (discarded from the team copy)
- DeepSORT tracker + MobileNetV3 576-d body embeddings (older than current).
- A single `PersonEmbedder` mixing body + face + ethnicity; hardcoded Windows
  model paths (`C:\Users\Zyad\...`); face embeddings written into the shared
  gallery.

### Result / Status
- ✅ Code integrated, flag-gated, isolated. Body pipeline untouched.
- ⏳ Pending team `.pth` files (glasses / ethnicity / violence) — those features
  stay disabled until the weights are dropped into `surveillant/models/`.
- Verification: regression run with flags OFF + Phase 1–3 tests; face core
  verified with `ENABLE_FACE_ANALYSIS=True` (InsightFace auto-downloads).

### Dashboard upgrade (Part 11 tooling) — 2026-06-16

Reworked the debug dashboard (`debug_dashboard.py` + `debug_dashboard_ui.html`)
into a richer "command center" that surfaces the new data:
- **Backend (read-only, unchanged safety model):** enriched `/api/stats`
  (face embeddings, descriptions, named persons, cameras, violence counts);
  `/api/persons` now supports attribute filters (gender/age/ethnicity/glasses),
  keyword search across id/name/description, sort, and returns the latest
  description summary + face-embedding count. New endpoints: `/api/analytics`
  (aggregates for charts), `/api/persons/{id}/description` (LLM history),
  `/api/persons/{id}/face_embeddings`, `/api/violence` (reads violence_log.json),
  `/api/alert_media` (serves alert snapshots/clips from the alerts dir, path-guarded).
  New tables guarded via `_table_exists` for backward compatibility.
- **Frontend:** new **Overview** tab (KPI cards + dependency-free distribution
  charts: status, per-camera, gender/age/ethnicity/glasses, gallery histogram,
  body-vs-face embeddings, description coverage, persons-over-time); enriched
  **Persons** table (name, profile badges, description snippet, face count) with
  rich filters + CSV/JSON export; expanded **detail panel** (profile, LLM
  description + structured attributes + history, body + face embeddings, camera
  timeline); new **Violence** tab with severity feed + snapshot/clip media; and a
  fuller header stat bar.
- Verified by seeding a temp DB and exercising every endpoint (stats, filtered
  persons, analytics, description, face, violence) — all green.

### Model deployment & live verification (Part 11) — 2026-06-16

The team delivered the trained weights (`models (2).zip`). Deployed and verified:
- Placed `best_race_classifier_resnet18_new.pth`, `glasses_classifier.pth`,
  `violence_detector_v3_office.pth` into `surveillant/models/`. (The zip's extra
  `age_mobilenetv2_balanced.h5` and `fairface.onnx` are not used — InsightFace
  provides age, and ethnicity uses the ResNet18 — left unwired.)
- **Glasses loader fix:** the glasses checkpoint turned out to be a **ResNet-50**
  (Bottleneck blocks) nested under a `base.` prefix with a single sigmoid head —
  NOT a ResNet-18. The original loader silently matched 0 weights (ran on random
  init). `GlassesDetector` now auto-detects backbone depth (18/34/50 via bn3/layer
  probes) and strips wrapper prefixes; reloads with `missing=0 unexpected=0`.
- Installed `insightface==1.0.1` + `onnxruntime==1.27.0` (0.7.3 has no Python-3.12
  wheel); updated `requirements.txt`.
- Enabled `ENABLE_FACE_ANALYSIS=True` and `ENABLE_VIOLENCE_DETECTION=True`.
- **End-to-end live check:** `buffalo_l` downloaded; analyzing a real person
  snapshot produced age `30-45`, gender `Male`, ethnicity `Asian`, glasses
  `Glasses`, plus a 512-d face embedding. Violence model scores on a frame seq.
  All four model groups confirmed working on CPU.
- Remaining: the full multi-camera live run (`python main.py --phase 2 --set
  set_1`) is run on the user's machine (needs the WiseNet videos + a display).
- Dashboard fix: the read-only dashboard crashed (`no such column: ethnicity`)
  on a pre-Part-11 DB because it can't run migrations. Made it column-resilient
  (`_columns()` guards in `/api/stats`, `/api/persons`, `/api/analytics`) AND
  migrated the existing `surveillant.db` (added name/ethnicity/glasses +
  face_embeddings). Attribute columns populate after a Phase-2 run with
  ENABLE_FACE_ANALYSIS=True.

---

## Phase 1 — Camera Simulator + Detection + Tracking + Display

### P1-A: Multi-threaded display architecture
**What:** Split display and detection into two threads: main thread renders at native FPS, worker thread does round-robin YOLO inference.
**Why:** Running YOLO detection synchronously on every frame made the display freeze. Round-robin (one camera per detection cycle) ensures smooth display while maximising throughput on CPU.
**Result:** Display stays smooth regardless of how long YOLO takes per frame.

### P1-B: Initial tracker — DeepSORT
**What:** Used `deep-sort-realtime` library for per-camera tracking.
**Why:** DeepSORT was the standard multi-object tracker at the time. Requires a Re-ID appearance model (embedding overhead).
**Status:** **REPLACED by ByteTrack** (see Part 7).

---

## Phase 2 — Embedding + Database + Cross-Camera Matching

### P2-A: Initial embedder — ResNet-50 (ImageNet, 2048-d)
**What:** Used torchvision ResNet-50 pretrained on ImageNet as the body embedding backbone.
**Why:** Available out-of-the-box, fast, well-known.
**Problem:** ImageNet weights encode *semantic category* (chair, cat, car) — not *person identity*. Same-person/different-person similarity distributions overlap heavily:
- Same person front→back: 0.45–0.65
- Different people, similar clothing: 0.55–0.75
- Distributions overlap completely — no reliable threshold possible.
**Status:** **REPLACED by OSNet x1.0 + Market-1501** (see Part 5).

### P2-B: Async embedding worker
**What:** A dedicated embedding thread consumes an `embed_queue` of `(cam_id, track_id, crops, ...)` items. Detection thread never blocks on embedding.
**Why:** OSNet inference takes ~20-50ms on CPU. Running it in the detection loop would drop frames. The queue decouples the two.
**Result:** Detection and display run unimpeded; embedding runs as fast as possible in the background.

### P2-C: Permanent track registry
**What:** `track_registry[(cam_id, track_id)] = person_id` — once a track is bound to a person, the binding is permanent for the session.
**Why:** Prevents repeated embedding calls for the same track. The registry is saved to `database/track_registry_session.json` so sessions survive restarts.
**Rule 1:** Intra-camera — once bound, never re-checked.
**Rule 2:** Inter-camera — new track does one embedding search, then Rule 1 applies.

### P2-D: SQLite with WAL
**What:** SQLite in WAL (Write-Ahead Logging) mode.
**Why:** The detection thread (reads) and embedding thread (writes) access the DB concurrently. WAL allows one writer + multiple readers simultaneously without deadlocks.
**Schema tables:** `persons`, `person_embeddings`, `camera_history`, `merge_proposals`.

### P2-E: Database callback hooks
**What:** `Database.on_embedding_added` and `Database.on_merge` — optional callbacks fired AFTER SQLite transaction commits.
**Why:** Allows downstream caches (FAISS index) to stay in sync without the Database knowing about them directly. Failures in hooks are caught and logged — they never take down the embedding worker because SQLite is the source of truth.

---

## Part 1 / Part 2 — Crop Quality Gate

### QG-A: Quality gate implementation
**File:** `modules/preprocessing/quality_gate.py`
**What:** Three-check filter applied before extracting gallery embeddings:
1. **Laplacian variance ≥ 50.0** — rejects motion-blurred / out-of-focus crops.
2. **Minimum size ≥ 48×96 px** — rejects far/edge crops that upscale badly.
3. **HSV V mean ≥ 30** — rejects dark silhouettes with no color information.
**Why:** A blurry, tiny, or dark crop produces an embedding that represents noise, not identity. Storing it pollutes the gallery and causes false non-matches.
**Applied to:** Gallery-update path only (both in `main.py` and defensively inside `GalleryManager`). **NOT** applied to identification crops — even a blurry crop carries enough signal for a one-time "does this person exist?" query.

---

## Part 2 — CLAHE + Auto-Gamma Frame Enhancement

### FE-A: FrameEnhancer implementation
**File:** `modules/preprocessing/enhancement.py`
**What:** Two-stage frame preprocessing before YOLO detection:
1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) in LAB color space — equalizes luminance channel only, preserving color hues for the embedder.
2. **Auto-gamma** (γ = 0.5) — applied only when mean frame brightness < 60 (very dark scenes).
**Why:** Indoor surveillance scenes have variable lighting. Dark frames cause YOLO to miss detections and produce embeddings that encode shadow shapes rather than identity.

### FE-B: Conditional CLAHE (Bug fix — Part 9 audit)
**What:** CLAHE was originally applied to every frame unconditionally.
**Problem:** On well-lit frames, CLAHE amplifies fine-grained noise and slightly degrades color accuracy — wasted ~10ms/frame.
**Fix:** Skip both stages when mean V-channel brightness > 120. Only trigger enhancement when the frame is actually dark.
**File:** `modules/preprocessing/enhancement.py`

---

## Part 4 — Background Isolation via YOLO Segmentation

### BG-A: Switch from `yolov8n.pt` to `yolov8n-seg.pt`
**What:** Upgraded YOLO model from detection-only to segmentation variant.
**Why:** The `-seg` variant outputs per-person pixel masks alongside bounding boxes. Zeroing out background pixels before sending the crop to the embedder means the embedding encodes *the person*, not the background wall/floor.
**Old problem:** Two different people in front of the same red wall had artificially high similarity. Same person in front of different backgrounds had artificially low similarity.
**Cost:** ~3ms extra per frame. Acceptable for the improvement in embedding quality.
**Config:** `YOLO_MODEL = "yolov8n-seg.pt"`, `USE_SEGMENTATION = True`
**Background replacement:** Neutral gray (value 128) — not black, because black biases the embedder toward encoding an absence of color.

### BG-B: IoU-based mask-to-track association
**File:** `modules/preprocessing/masking.py` — `associate_masks_to_tracks()`
**What:** After ByteTrack reorders and Kalman-corrects track bboxes, a simple index-match between YOLO detections and ByteTrack outputs fails. This function recovers the mask for each track via IoU matching (threshold 0.4).
**Why:** ByteTrack returns Kalman-corrected bboxes that don't exactly match YOLO's original detections. Direct equality comparison (`d["bbox"] == t["bbox"]`) always fails (Bug B fix).

---

## Part 5 — OSNet x1.0 Embedder

### EM-A: Replace ResNet-50 with OSNet x1.0
**File:** `modules/embedding/embedder.py`
**What:** Replaced ResNet-50 (ImageNet, 2048-d) with OSNet x1.0 (Market-1501 Re-ID weights, 512-d).
**Why:** OSNet was purpose-built for person Re-ID (ICCV 2019). Market-1501 is a Re-ID benchmark dataset of 1,501 identities across 6 cameras — OSNet trained on it encodes *individual identity*, not semantic category.

**Old (ResNet-50 / ImageNet):**
- Same person, 90° turn: 0.55–0.75 (overlaps different-person range)
- Different people, similar clothes: 0.55–0.75

**New (OSNet / Market-1501):**
- Same person, 90° turn: 0.78–0.92 (clean separation)
- Different people, similar clothes: 0.20–0.50

**Critical note:** `torchreid.build_model(pretrained=True)` loads **ImageNet weights only** — NOT Re-ID weights. Market-1501 weights must be downloaded separately via `gdown` from the torchreid model zoo. This step is mandatory; without it OSNet performs barely better than ResNet-50.
**Cache path:** `~/.cache/torchreid/checkpoints/osnet_x1_0_market1501.pth`

### EM-B: Input size — 256×128 (H×W)
**What:** OSNet input is resized to 256×128, not 224×224.
**Why:** OSNet was trained on this aspect ratio which matches typical person crops (taller than wide). Using 224×224 would distort body proportions.

### EM-C: Median aggregation (Bug H fix)
**What:** Changed `aggregate_embeddings()` from mean to median pooling over the 4-frame identification buffer.
**Why:** Mean is skewed by outlier frames (blur flash, partial occlusion). Median is robust — one bad frame out of 4 doesn't move the aggregate.

### EM-D: Database incompatibility after backbone change
**Important:** Old 2048-d ResNet-50 embeddings are byte-incompatible with new 512-d OSNet embeddings. The dimension check in `searcher.py` and the reconciliation worker auto-skip stale embeddings, but the database should be deleted and rebuilt from scratch after the upgrade.

---

## Part 6 — Pose-Aware Gallery

### PG-A: Canonical view classification
**File:** `modules/embedding/gallery.py` — `estimate_view()`
**What:** Classifies a bounding box into one of 4 canonical views: `frontal`, `right_moving`, `left_moving`, `side`.
**Logic:**
- `aspect = width/height < 0.40` → `"side"` (person sideways)
- Horizontal Δcenter > 8px → `"right_moving"` / `"left_moving"`
- Otherwise → `"frontal"`

### PG-B: Force-accept for uncovered canonical slots
**What:** If a canonical view slot (e.g. `right_moving`) is not yet covered for a person, the new embedding is force-accepted regardless of cosine distance, bypassing the novelty/diversity gate.
**Why:** Without this, the gallery accumulates near-duplicate frontal views because the diversity gate only accepts novelty — and the person never happens to walk sideways when the diversity gate is satisfied.
**Guard:** Force-accept still checks `GALLERY_MAX_DISTANCE` (garbage check) and `MAX_GALLERY_SIZE`.

### PG-C: View coverage score
**What:** `get_view_coverage()` returns `covered_canonical_slots / 4` (0.0–1.0).
**Used in:** Reconciliation worker only — both persons must have coverage ≥ 0.5 before being considered merge candidates.
**NOT used in:** The real-time searcher (Bug F fix — see below).

### PG-D: Bug F — view-coverage gate in searcher caused duplicate IDs
**What:** The view-coverage gate was originally also in `searcher.py` to skip persons with < 2 canonical views.
**Problem:** When a person re-entered the scene, their gallery was already built but the view-coverage check caused the searcher to skip them → new person_id created every re-entry → unbounded duplicate proliferation.
**Fix:** Removed the view-coverage gate from the searcher entirely. The gate only makes sense for reconciliation (batch, high-confidence decisions), not real-time re-entry matching.

### PG-E: Bug C — initial embedding tagged "initial" blocked reconciliation
**What:** The first embedding for a new person was tagged `"initial"` (not in `CANONICAL_VIEWS`).
**Problem:** `get_view_coverage()` only counts canonical views — `"initial"` doesn't count. So every person started with view-coverage 0.0 and was never eligible for reconciliation.
**Fix:** When creating a new person in `main.py`, the canonical view is estimated from the first crop's bbox and passed as `angle_tag` to `insert_person()`. The initial embedding is now tagged with a canonical view from day one.

---

## Part 7 — ByteTrack (Replaces DeepSORT)

### BT-A: Replace DeepSORT with ByteTrack
**File:** `modules/tracking/tracker.py`
**What:** Switched from `deep-sort-realtime` to `ultralytics` built-in ByteTrack.
**Why:**
- DeepSORT requires an appearance model (heavy) and discards low-confidence detections.
- ByteTrack uses **two-stage IoU association** — low-confidence detections (0.10–0.45) keep coasting tracks alive through occlusion instead of killing them.
- When a person is half-occluded, YOLO outputs a low-conf detection (0.25–0.40). DeepSORT discards it → track dies → new ID on reappearance → "color change on turn" bug. ByteTrack uses it → track survives → same color.

**Two-stage association:**
- Stage 1 (≥ 0.45 conf): IoU match against active tracks.
- Stage 2 (0.10–0.45 conf): matches against unmatched tracks from Stage 1 only.

**Config:**
```
BYTETRACK_TRACK_THRESH = 0.45   (high-conf gate)
BYTETRACK_LOW_THRESH   = 0.10   (low-conf gate — keeps tracks alive through occlusion)
BYTETRACK_MATCH_THRESH = 0.80   (IoU threshold for association)
DETECTION_CONF         = 0.10   (pass all detections to ByteTrack)
```

### BT-B: Bug B — IoU-based high-conf crop filter
**What:** Crops for embedding were gated on `d["bbox"] == t["bbox"]` (exact list equality).
**Problem:** ByteTrack returns Kalman-corrected bboxes, which are never exactly equal to YOLO's raw detection bboxes. This always returned False → no crops ever queued for embedding → no identities ever created.
**Fix:** Replaced with IoU ≥ 0.7 match between the ByteTrack track bbox and the highest-confidence matching YOLO detection for that frame.

---

## Part 8 — FAISS In-Memory Vector Index (+ Regression Investigation)

### FA-B: Gallery-sponge regression reported after FAISS — diagnostic flags added (2026-05-28)
**Observed symptom:** After Part 8 was merged, multiple visually-distinct walking people were all assigned the same `person_id` and the same bounding-box color ("the orange one"), while the sitting person stayed correctly isolated. The debug dashboard confirmed a "gallery sponge" — one person record (`6e7cc524`) accumulated snapshots of at least 3 different physical people, with all 4 canonical angle slots filled. Live-activity log showed this person "moving" between cameras 1–4 in seconds — physically impossible, confirming wrong track bindings.

**Confirmed:** fresh database was used, so the sponge forms within a single session (not caused by stale embeddings from a prior run).

**Hypotheses being tested (see plan file for full details):**
- **H1** — FAISS `IndexFlatIP` returns inflated scores vs sklearn cosine (FAISS bug).
- **H2** — FAISS is 1000× faster, shifting timing/race windows in the embedding worker.
- **H3** — Pre-existing calibration issue: `BODY_MATCH_THRESHOLD = 0.65` + Part 6 force-accept (only checks `GALLERY_MAX_DISTANCE = 0.55`) is too loose, creating a self-reinforcing sponge. FAISS made it consistently visible.

**Diagnostic changes added (Step 1):**
1. `ENABLE_FAISS = True` in `config/settings.py` — kill switch. Set `False` to fall back to SQLite linear scan for A/B testing.
2. `FAISS_AUDIT_MODE = False` in `config/settings.py` — when `True`, every identification query runs both FAISS and SQLite paths; logs `[FAISS_DRIFT]` if scores differ by > 0.01 or top-1 person IDs differ.
3. `modules/search/searcher.py` — added `_search_via_sqlite_raw()` (returns raw scored list) and `_log_audit_drift()` (compares the two paths); `search_by_embedding()` routes through audit path when `FAISS_AUDIT_MODE=True`.
4. `main.py` — FAISS construction in `run_phase2` and `run_phase3` wrapped in `if ENABLE_FAISS:`.

**Diagnostic result (2026-05-28):** `ENABLE_FAISS=False` ran — sponge still formed. **H3 confirmed. FAISS is innocent.**

### FA-C: H3 fix — BODY_MATCH_THRESHOLD raised + force-accept guard tightened (2026-05-28)
**Root cause confirmed:** Two physically-distinct walking people in the WiseNet scene had OSNet embeddings with cosine similarity ≥ 0.65, triggering an initial wrong binding. Once bound, the pose-aware force-accept path (Part 6) absorbed the wrong person's subsequent crops into the gallery because its only guard was `distance ≤ GALLERY_MAX_DISTANCE = 0.55` (sim ≥ 0.45) — far too loose. The gallery became a sponge covering all 4 canonical angles with multiple people's embeddings. Every new track's max-pool query then matched this multi-person gallery even more easily, snowballing.

**FAISS is innocent** — it faithfully reproduced the same wrong match that SQLite would have. FAISS re-enabled.

**Fix 1 — `BODY_MATCH_THRESHOLD` raised 0.65 → 0.72** (`config/settings.py`)
- 0.72 is the OSNet/Market-1501 recommended boundary from Enhancement Proposal §5.2.
- 0.65 was deliberately lowered for sitting/standing tolerance but turned out to put the threshold *inside* the different-person score range for this dataset (~0.65–0.70).
- 0.72 sits clearly above the different-person ceiling and well below the same-person floor across pose changes (0.78+).

**Fix 2 — `FORCE_ACCEPT_MAX_DISTANCE = 0.35` added; used in gallery.py force-accept check** (`config/settings.py`, `modules/embedding/gallery.py`)
- Force-accept (Part 6) now requires the new embedding to score ≤ 0.35 distance (sim ≥ 0.65) against the existing gallery before it fills an empty canonical slot.
- Previously used `GALLERY_MAX_DISTANCE = 0.55` (sim ≥ 0.45) — essentially any non-garbage crop could be force-accepted.
- This is a defense-in-depth guard: even if a track is wrongly bound (which Fix 1 makes much less likely), its subsequent gallery crops won't be accepted if they're too different from what's already stored.

**New invariant added:** `FORCE_ACCEPT_MAX_DISTANCE` must never be set higher than `BODY_MATCH_THRESHOLD`'s equivalent distance (i.e., `1 - BODY_MATCH_THRESHOLD = 0.28`). Currently 0.35 is slightly looser than identification threshold (0.28) — intentionally so, since gallery updates are lower-stakes than initial binding.

### FA-D: Dual-threshold matching — same-cam vs cross-cam (2026-05-28)
**Observed after H3 fix:** Two residual issues:
1. Small mini-sponge (`991b03df`, 3 embeds, side-only, cam1) — two different people's side views scored ≥ 0.72 → still merged at identification.
2. Cross-camera split — same person transitioning cam1 → cam2 had their aggregate embedding drop to 0.68–0.72 due to angle/lighting change → created a new person_id.

**Root cause of remaining tension:** A single threshold cannot separate the overlap zone 0.68–0.72 which contains BOTH different-people-same-cam scores AND same-person-cross-cam scores.

**Fix — context-aware dual threshold** (`config/settings.py`, `main.py`, `searcher.py`):
- `BODY_MATCH_THRESHOLD_SAME_CAM = 0.75` — applied when top-1 candidate's `last_seen_cam == current cam`. A truly returning same-camera person scores 0.85+; 0.75 clears the ~0.72 false-positive ceiling for same-camera side views.
- `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` — applied when `last_seen_cam != current cam`. Cross-camera angle/lighting drops same-person scores to 0.68–0.72; 0.68 catches them.
- `BODY_MATCH_THRESHOLD = 0.68` — set equal to the lower threshold (serves as the searcher floor). PersonSearcher._hydrate uses this so cross-camera candidates are not filtered before main.py's context check.
- `searcher.search_by_embedding()` now accepts `min_threshold` optional override, used by phase-2 identification to request the cross-cam floor.
- **Log format upgraded:** `[MATCH]` and `[BELOW]` lines now print `(cross-cam(cam1→cam2))` or `(same-cam(cam4))` for easier debugging.

**Architectural note:** The same-camera live-conflict guard (`same_cam_conflict`) already prevents simultaneous same-camera duplicates. The dual threshold handles the sequential case (same-camera different person arriving after the previous one left).

### FA-A: FAISS IndexFlatIP implementation
**File:** `modules/search/faiss_index.py`
**What:** In-memory `faiss.IndexFlatIP(512)` — exact inner-product search on L2-normalized vectors = cosine similarity.
**Why:** SQLite linear scan on every query loaded all embeddings, computed cosine similarity in a Python loop. At 30 persons × 5 views = 150 vectors: ~10ms/query. At 500 persons: ~300ms/query (real-time bottleneck).
**Result:** 916× speedup measured at 100-vector gallery (65.3ms → 0.07ms / query).

**Architecture:**
- FAISS = redundant in-memory copy for fast nearest-neighbour search.
- SQLite = source of truth for all metadata and embeddings.
- If `faiss-cpu` is not installed or the index is empty, the Searcher transparently falls back to SQLite.

**Synchronization:**
1. Startup: `rebuild_from_db()` bulk-loads all embeddings from SQLite.
2. Live insert: `Database.on_embedding_added` callback → `FAISSIndex.add()`.
3. Merge: `Database.on_merge` callback → `FAISSIndex.reassign_person()` (relabels idx→pid map, no rebuild).

**Aggregation:** Max-pool per person (same as SQLite path — ensures identical threshold semantics).

**Validated (measured):**
- 20/20 identical top-1 matches vs SQLite on 20-person × 5-view test gallery.
- Score difference: 0.000000 (within float-32 precision).
- Concurrent stress: 1,174 inserts + 3,612 searches over 2s, 0 errors.
- Stale 2048-d embeddings silently skipped — no crash.

---

## Part 8.5 — Overlap-Aware Matching (Camera Topology, prelude to Part 9)

### OV-A: Problem statement (2026-05-28)

**What:** The dual-threshold matching (0.75 same-cam / 0.68 cross-cam) was designed for *sequential* cross-camera transitions where the person leaves cam1 and arrives at cam2 with a similar canonical view. It does NOT handle the case where two or more cameras have overlapping fields of view — same room, different angles — and the same physical person appears on cam1 and cam2 *simultaneously*.

**Why it matters:** OSNet's same-person score across a sharp angle change in the same instant lands in 0.55–0.70 — *below* the 0.68 cross-cam threshold. The system would split one physical person into two `person_id`s with no live mechanism to merge them. On WiseNet (5 rooms / hallways with non-trivial camera coverage) this is a visible, observable failure mode that the dual threshold cannot fix alone.

**Dual of Part 9:** Part 9 (spatio-temporal) rejects physically impossible matches. Part 8.5 rescues legitimate matches the visual matcher missed because of an angle gap. Both need topology knowledge; both compose cleanly. Part 8.5 ships first because overlap-splits are observable today.

### OV-B: Layer 1 — Triple-threshold live matching

**Files:** `config/settings.py`, `main.py`

**What:** Replaced the dual-threshold context-aware decision in `main.py`'s embedding worker (~line 542) with a three-way pick:

| Branch | Threshold | Rationale |
|---|---|---|
| Same camera (`last_cam == cur_cam`) | `BODY_MATCH_THRESHOLD_SAME_CAM = 0.75` | unchanged — strict to reject same-camera sequential look-alikes |
| Overlap partner (`are_overlapping_cams(cur_cam, last_cam)`) | `BODY_MATCH_THRESHOLD_OVERLAP = 0.62` | NEW — looser, because same-person sharp-angle-in-same-instant scores from a lower distribution than sequential cross-cam |
| Cross-cam, non-overlap | `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` | unchanged — moderate, sequential transitions land in 0.68–0.78 |

**New config:**
- `CAMERA_OVERLAP_GROUPS: list[set[int]] = []` — user-configurable declaration of which cam_ids share physical space. Empty default = feature off, drop-in safe.
- `BODY_MATCH_THRESHOLD_OVERLAP = 0.62`.
- `are_overlapping_cams(cam_a, cam_b)` helper.
- `validate_overlap_topology(num_cams)` — warns at startup on disjoint-group violations, unknown cam_ids, threshold-ordering violations. Never raises (WORKING_INSTRUCTIONS §5).

**Searcher floor change (also in `main.py`):** `min_threshold` in the identification search call dropped from `BODY_MATCH_THRESHOLD_CROSS_CAM = 0.68` to `BODY_MATCH_THRESHOLD_OVERLAP = 0.62`. Otherwise the searcher would pre-filter overlap-band candidates before main.py's context check sees them. When `CAMERA_OVERLAP_GROUPS = []`, candidates in [0.62, 0.68) are still returned but then correctly rejected by the cross-cam threshold — harmless overhead.

**`BODY_MATCH_THRESHOLD` (legacy alias) stays at `CROSS_CAM = 0.68`.** This is the floor for callers without camera context (Phase-3 photo search). Live identification passes its own `min_threshold=` override and is unaffected.

**Log format:** `[MATCH:OVERLAP]` (new) for overlap-cam matches alongside `[MATCH]` for same-cam / cross-cam. Cam label format `(overlap-cam(cam1↔cam2))` mirrors the `(cross-cam(cam1→cam2))` / `(same-cam(cam4))` pattern added in FA-D.

**New invariant (#12):** `BODY_MATCH_THRESHOLD_OVERLAP <= BODY_MATCH_THRESHOLD_CROSS_CAM <= BODY_MATCH_THRESHOLD_SAME_CAM`. Also: `(1 - BODY_MATCH_THRESHOLD_OVERLAP) > FORCE_ACCEPT_MAX_DISTANCE` so force-accept doesn't become a sponge on overlap pairs (0.38 > 0.35 at default).

### OV-C: Layer 2 — Co-visibility boost in reconciliation

**File:** `modules/reconciliation/worker.py`, `modules/storage/database.py` (new helper `get_camera_history()`)

**What:** Before computing `mean-pool similarity` and comparing against `MERGE_CANDIDATE_THRESHOLD = 0.58`, the reconciliation cycle now also computes a *co-visibility moment count* for the pair: how many independent moments did the two persons' `camera_history` intervals temporally overlap on either (a) the same physical camera or (b) declared overlap-partner cameras. If `count >= CO_VISIBILITY_MIN_MOMENTS = 3` (each ≥ `CO_VISIBILITY_MIN_OVERLAP_SEC = 0.5` s of overlap), the threshold for THAT PAIR drops from 0.58 to `MERGE_CANDIDATE_THRESHOLD_OVERLAP_BOOSTED = 0.45`.

**Why this works:** Two unrelated people would not repeatedly share the *exact* same time window on the *same* overlap pair across multiple distinct moments — that happens only by coincidence, rarely. One physical person mis-split into two IDs will show up in lock-step on the overlap-partner cameras every time they enter the room.

**Why this is safe:** The mean-pool similarity floor still applies. A pair only gets the boosted treatment if both visual similarity (mean-pool ≥ 0.45) AND repeated co-visibility (≥ 3 moments) agree. Either signal alone is insufficient.

**New log line:** `[RECONCILE:CO-VISIBILITY]` when the boost fires, with moment count and score.

**New database method:** `Database.get_camera_history(person_id)` returns full `(cam_id, track_id, first_seen, last_seen)` rows — needed because `get_cameras_for_person()` only returns distinct cam_ids and discards the timing.

### OV-D: Forward compatibility with Part 9

When Part 9 (spatio-temporal transition matrix) lands, overlap-group pairs get `MIN_TRANSITION_SEC = 0` for free — no special case needed. The two features compose:

| Pair type | Part 9 (future) | Part 8.5 (now) |
|---|---|---|
| Same cam | min_transition = 0 | threshold = 0.75 |
| Overlap partner | min_transition = 0 | threshold = 0.62 |
| Adjacent non-overlap | min_transition = ~8 s | threshold = 0.68 |
| Far non-overlap | min_transition = ~30 s | threshold = 0.68 |

### OV-E: Status

- **WiseNet topology declared (2026-05-28):** `CAMERA_OVERLAP_GROUPS = [{3, 4}]` — cameras 3 and 4 share a physical field of view. Overlap branch is now live for cam3↔cam4 pairs.
- All Phase-2 and Phase-3 regressions verified by syntax/import check; integration validated by per-file `ast.parse()` round-trip and invariant assertions.
- Startup validator prints topology summary: `[SURVEILLANT] Overlap topology: 1 group(s) — thresholds same=0.75 overlap=0.62 cross=0.68`.
- `[MATCH:OVERLAP]` and `[RECONCILE:CO-VISIBILITY]` log lines will appear when cam3 and cam4 share a person sighting.

---

## Part 10 (Phase 4A + 4B) — LLM Body Description + Natural-Language Search

### LL-Y: Describer "quality mode" (2-pass) — implemented, benchmarked, kept OFF (2026-06-14)
**What:** Added opt-in `DESCRIBE_QUALITY_MODE` (default False). When on, `QwenVLOllamaDescriber` runs a 2-pass pipeline with the SAME model: (#2) a head-to-toe `SCAN_ADDENDUM` appended to the describe prompt, and (#1) a `SYSTEM_PROMPT_VERIFY` self-verification call that re-checks the draft against the image. Refactored the HTTP call into `_post_chat()` for reuse.
**Why:** User asked whether a "skill" (prompt/pipeline technique) could get better results from qwen2.5vl:3b without the 540 s thinking cost.
**Benchmark (3 people, warm model):**

| Person | single-pass | quality mode |
|---|---|---|
| b02148df | 8 keys, ~15 s | 7 keys, ~25 s (missed same beanie) |
| d65c54d4 | 6 keys, 12 s | 9 keys, 34 s (+"shirt underneath", −"dark pants") |
| ea573f4f | 8 keys, 13 s | 9 keys, 139 s (+"holding object", −"beard") |

**Finding:** the self-verify pass (#1) does NOT reliably improve quality — a 3B model can't critique itself, so the second pass behaves like an independent re-sample: it adds ~1–3 structured keys but DROPS other correct facts, at 2–10× the latency (high variance). The systematic scan (#2) is harmless and modestly more thorough. The beanie that thinking-mode caught was missed by both passes → a perception/resolution limit, not a verification gap.
**Decision:** keep `DESCRIBE_QUALITY_MODE = False` (default). The feature ships behind the flag for experimentation but is NOT recommended on. The higher-value "same-model" levers are input-side: #3 upscale small crops (so the vision encoder can resolve small items) and #4 deterministic HSV colour sanity-check (anti-hallucination from pixels, not LLM self-critique). Honest negative result recorded so it isn't re-attempted blindly.

### LL-X: Thinking-mode benchmark — qwen2.5vl:3b vs qwen3-vl:2b (2026-06-14)
**What:** Head-to-head on the same crop + same prompt, measuring time and description quality with reasoning OFF vs ON.

| | qwen2.5vl:3b (no-think) | qwen3-vl:2b (thinking on) |
|---|---|---|
| Time / person (CPU) | **13 s** | **540 s (~9 min)** |
| Internal `<think>` chars | 0 | 11,284 |
| Structured keys | 6 | 11 |
| Detail | good | better — caught a dark beanie the 2.5 model missed; read the partial logo literally ("NOT DO" = visible fragment of "JUST DO IT") rather than auto-completing it; added hair/headwear/bottom-colour |

**Why run it:** user asked to see the quality-vs-time difference of thinking mode.
**Finding:** reasoning genuinely improves description completeness and faithfulness (more accurate, less auto-completion), but at ~41× the latency — 540 s/person is impractical on this CPU (a full gallery = hours). **Decision: stay on qwen2.5vl:3b for CPU; thinking mode (qwen3-vl) is only viable on the GPU/Marlin backend, where the reasoning runs in seconds.** To try it manually: set `OLLAMA_VLM_MODEL="qwen3-vl:2b"`, `OLLAMA_VLM_NUM_CTX=8192`, `OLLAMA_THINK=None` (thinking on by default), `OLLAMA_VLM_TIMEOUT_SEC>=700`, then `--redescribe-all`.

### LL-W: Describer prompt tuning — reliable long_description + thorough structured keys (2026-06-14)
**What:** Two prompt refinements to `USER_PROMPT_DESCRIBE`:
1. Added a concrete few-shot JSON **example** (with `long_description` first). Before this, qwen2.5vl under `format=json` filled the enumerated structured keys but SKIPPED the free-text `long_description`, so `_clean_flexible` fell back to a synthesised "gender: male; …" string. The example made the model emit a real `long_description` reliably.
2. Added an explicit **thoroughness instruction**: "for EVERY feature you mention in long_description that you are sure about, add its structured key (e.g. 'light blue Nike t-shirt' → clothing_top='t-shirt', clothing_top_color='blue')." Before this, the model wrote rich prose but extracted only 3–5 sparse keys (e.g. described a Nike t-shirt but omitted clothing_top). After: 5–9 keys per person, mirroring the description, while still OMITTING anything not visible.
**Why:** User wanted BOTH a rich `long_description` AND a complete set of confident key/value features — not one at the expense of the other.
**Result/Status:** Verified — descriptions now carry a detailed long_description plus 5–9 structured keys; the anti-hallucination omission rule still holds (uncertain/invisible keys absent). Re-indexed via `--redescribe-all`. (Residual: the 3B model occasionally still mentions a non-visible item in the prose — a small-model adherence limit, not a schema bug.)

### LL-V: Phase 4B redesign — flexible anti-hallucination describer + semantic search (2026-06-14)
**What:** Reworked description generation and search around two user-driven principles: (1) the describer must only state what it is sure it can SEE, and (2) search must match by MEANING, not SQL keywords.

**Describer — flexible, honest schema** (`modules/llm/describer.py`):
- New strict-but-flexible prompt: the model returns a required `long_description` (rich free text of only what's visible) PLUS any structured keys it is *confident* about (`gender`, `clothing_top/color`, `clothing_bottom/color` only if the lower body is visible, `hair_*`, `glasses`, `headwear`, `accessories`). It is told to OMIT any key it cannot see — never to guess.
- Replaced `_coerce_to_schema()` (which force-filled all 17 fields with `"unknown"`) with `_clean_flexible()` which keeps ONLY the keys returned and drops empty/uncertain values (`_UNCERTAIN_VALUES`). No invented attributes.
- **Why:** the old rigid 17-field schema forced hallucination — a face-only crop was labelled "brown pants" though no legs were visible. Honesty + omission is now required.

**Search — semantic embeddings, not LIKE** (`modules/search/text_embedder.py` NEW, `text_search.py` rewritten, `database.py`):
- New `text_embedder.py`: one lazily-loaded `all-MiniLM-L6-v2` shared by describe and search; L2-normalised float32.
- At describe time the worker embeds `long_description` and stores the vector in a new `person_descriptions.embedding BLOB` column (schema + `_migrate` ALTER; `insert_description(embedding=...)`).
- `TextSearchEngine.search()` now embeds the query and cosine-ranks all stored vectors → nearest meaning. The QueryParser + Stage-1 SQL `LIKE` filter + Stage-2 synonym table + Stage-3 fallback are all GONE (one clean semantic path).
- **Why:** SQL `LIKE` keyword matching was brittle (the hoodie/sweater asymmetry, "black t-shirt" → 0 results). Embeddings handle synonyms/paraphrase/partial wording with no hand-curated tables.

**Re-indexing:** new `--redescribe-all` CLI flag (+ `db.get_all_person_ids()`) re-describes every person with the current prompt/model and rebuilds embeddings — needed after a prompt/model change. Existing rows without an embedding are skipped by search with a printed hint.

**Old approach:** rigid 17-field "unknown"-filled JSON + parse→SQL-LIKE→synonym→ST-fallback search.
**New approach:** flexible visible-only JSON + `long_description` + pure cosine semantic search over stored MiniLM embeddings.
**Result/Status:** verified `_clean_flexible` drops uncertain keys (e.g. `clothing_bottom:'unknown'` omitted). Re-describe + semantic search validated end-to-end (see below). `sentence-transformers` is now a hard dependency of search (already in requirements.txt).

### LL-U: Search refinements — synonym-symmetric Stage 1 + Stage-3 enabled (2026-06-14)
**What:** (a) Fixed a synonym asymmetry in `TextSearchEngine`: query phrase terms are now expanded to their full synonym group and OR-matched in Stage-1 SQL. New `_SYNONYM_GROUPS` reverse index + `_synonym_group()` + `_expand_for_db()` in `text_search.py`; `database.search_persons_by_attributes()` phrase-field branch generalised to accept a list (OR of LIKEs). (b) Installed `sentence-transformers` so the Stage-3 semantic fallback actually runs.
**Why:** A search for "black hoodie" matched the black *sweater* but MISSED a person literally stored as "black hoodie", because the query canonicalised `hoodie→sweater` while the describer stored the literal `hoodie` — the LIKE filter was asymmetric. Separately, near-miss queries (e.g. "black t-shirt" when only a gray t-shirt exists) returned `(no matches)` because Stage 3 (`sentence-transformers`) wasn't installed.
**Old approach:** Stage 1 matched a single canonical token; Stage 3 disabled (import error).
**New approach:** Stage 1 OR-matches the whole synonym group (symmetric — "hoodie" and "sweater" both find either stored form); Stage 3 lazily loads `all-MiniLM-L6-v2` and cosine-ranks all summaries when Stage 1 is empty.
**Result/Status:** Verified — "black hoodie" returns both hoodie+sweater people; "black t-shirt" returns 5 nearest people ranked (t-shirt person #1 at 0.53). Colour/enum fields needed no change (already coerced to the palette on storage, so exact match is symmetric). `sentence-transformers` was already pinned in requirements.txt.

### LL-T: Switch describer to qwen2.5vl:3b (non-reasoning) — qwen3-vl unusable on CPU (2026-06-14)
**What:** Changed `OLLAMA_VLM_MODEL` and `OLLAMA_QUERY_MODEL` from `qwen3-vl:2b` to **`qwen2.5vl:3b`**; reverted `OLLAMA_VLM_NUM_CTX` 8192→2048 and `OLLAMA_VLM_TIMEOUT_SEC` 600→180; set `OLLAMA_THINK = None` and made `describer.py` OMIT the `think` field when None (non-reasoning models reject it).
**Why:** Exhaustive testing proved qwen3-vl:2b cannot describe images acceptably on this CPU-only laptop, and it is NOT a config/prompt/integration bug:
  - Raw CPU speed is fine: **19 tok/s** on text.
  - But qwen3-vl ALWAYS emits ~1800 internal `<think>` tokens when given an image, and that reasoning is **undisableable** on this Ollama build (0.30.7): `think=False` ignored; `/no_think` works for text but is **ignored for vision inputs** (verified — thinking still 1600+ tokens).
  - WITH an image, generation drops to **~4.4 tok/s** (vision layers), so ~1800 reasoning tokens ⇒ **400 s+ per describe**, often >600 s when reasoning rambles to fill context → timeouts.
  - There is no `num_ctx`/timeout/prompt value that fixes a model that mandatorily reasons for minutes per image on CPU.
**Old approach:** qwen3-vl:2b (reasoning) + the num_ctx=8192/timeout=600 band-aids (LL-R) that made it *reliable* but ~150–600 s/person and fragile under RAM pressure.
**New approach:** qwen2.5vl:3b — a NON-reasoning VLM (Qwen2.5-VL's smallest size; there is no 2B — `qwen2.5vl:2b` 404s on Ollama). Answers directly in ~20–40 s/person, fits num_ctx=2048, low RAM, deterministic. User chose this (over GPU/Marlin) after seeing the evidence.
**Result/Status:** Model downloading (~3.2 GB). All the prior fixes remain valid and necessary (D:\Ollama models store LL-Q; 127.0.0.1 LL-L; lean Phase 2 LL-S). The `think`-handling is now conditional so the codebase supports both reasoning and non-reasoning backends. **Lesson:** for CPU-only structured-output VLM work, pick a NON-reasoning model; reasoning VLMs need a GPU.

### LL-S: Decouple describer from Phase 2 — ENABLE_DESCRIPTION_WORKER=False (2026-06-14)
**What:** Set `ENABLE_DESCRIPTION_WORKER = False` in `config/settings.py`. Phase 2 no longer starts the DescriptionWorker daemon or enqueues describe jobs during live tracking (both already gated on this flag in `main.py:364` and `:752`). Descriptions are generated on demand via `python main.py --phase 4 --describe-all`.
**Why:** User reported Phase 2 became slow/weak — boxes laggy and less accurate than before. Detection/tracking CODE is unchanged, so the regression is CPU contention. With qwen3-vl now confirmed to pin the CPU for ~150–250 s per person (reasoning model, LL-R), running the describer as a live daemon starves the round-robin YOLO detector and the OSNet embedding worker on this 8-core CPU-only laptop. Decoupling restores live-tracking performance; descriptions are a post-process step anyway.
**Old approach:** `ENABLE_DESCRIPTION_WORKER = True` — describer ran during Phase 2, competing for CPU.
**New approach:** `False` — Phase 2 runs lean; describe in a separate Phase-4 pass (ideally with other apps closed to free RAM).
**Result/Status:** Phase 2 detection/tracking no longer competes with the VLM. User chose to keep qwen3-vl on CPU (accepting ~150–250 s/person and the need to free RAM); this change makes that viable by keeping the cost out of the live loop.

### LL-R: The REAL describe fix — num_ctx=8192 for qwen3 reasoning overflow (2026-06-14)
**What:** Added `OLLAMA_VLM_NUM_CTX = 8192` (was a hardcoded `num_ctx=2048`) and raised `OLLAMA_VLM_TIMEOUT_SEC` 300→600 in `config/settings.py`; both Ollama payloads in `describer.py` now use `OLLAMA_VLM_NUM_CTX`.
**Why (corrects LL-P):** After fixing the Ollama store (LL-Q), describes STILL returned empty (`len(content)=0`) — but now with `done=length` and a populated `thinking` field of ~2000–2200 chars. Root cause: **qwen3-vl reasons for ~2000 tokens before answering, and at `num_ctx=2048` that reasoning fills the entire context window, so the model hits the length limit before emitting any JSON.** Critically, `think=False` AND the Qwen `/no_think` directive were BOTH ignored on this Ollama build (0.30.7) — thinking could not be disabled, only accommodated. The earlier LL-P "fix" only worked intermittently because some runs happened to think <2048 tokens and squeaked under the limit. Verified: at `num_ctx=8192`, two consecutive runs both returned valid JSON (`done=stop`, parseable), e.g. *"A bald man wearing a black t-shirt and shorts, seated…"*.
**Old approach:** `num_ctx=2048` → reasoning overflow → empty content / `done=length`.
**New approach:** `num_ctx=8192` gives room for the full reasoning trace PLUS the answer → deterministic `done=stop`. `think=False` (LL-P) is kept — harmless, and may shorten reasoning on builds that honor it.
**Result/Status:** Describe is now deterministic but slow on CPU (~150–250 s/person — the model generates ~2000+ tokens). Timeout raised to 600 s to cover it. This is the inherent cost of a reasoning VLM on CPU; the Marlin GPU backend is the path for larger galleries. **Lesson:** for reasoning models used with `format=json`, size `num_ctx` for (reasoning + answer), not just the prompt.

### LL-Q: Ollama model-store mismatch — describer got 404 / "backend returned None" (2026-06-14)
**What:** Operational/environment fix (no SURVEILLANT code change). The describer intermittently failed for every person with `404 Not Found` or `backend returned None`, while the monitoring dashboard showed only the early-described persons succeeding.
**Why:** The Ollama **desktop app** is configured (in its own `db.sqlite`, NOT an env var — invisible to User/Machine env queries) to store models in **`D:\Ollama models`** and bind `OLLAMA_HOST=http://0.0.0.0:11434`. But `qwen3-vl:2b` had been pulled with the **CLI**, which defaulted to `C:\Users\zezoe\.ollama\models`. So the app's auto-started server looked in `D:\Ollama models`, found no models, and returned empty/404 for every `/api/chat`. The earlier "two servers / IPv4 vs IPv6" confusion was a side-effect of a transient manual `ollama serve`; the real, durable cause was the **store-location split**. The "only the 2 stationary people got described" pattern was pure timing coincidence (they were described during a brief window when a correct server was up), not anything about motion or angle — confirmed because the failed persons had 12–16 crops that ALL passed the quality gate.
**Old state:** model in `C:\…\.ollama\models`; app server reading empty `D:\Ollama models`.
**New state:** merged the local store into `D:\Ollama models` (robocopy, no re-download) so the app's own server finds the model; set persistent `OLLAMA_MODELS=D:\Ollama models` so future CLI pulls land where the app looks. App server now serves `qwen3-vl:2b` on `0.0.0.0:11434` (reachable via `127.0.0.1`).
**Result/Status:** Permanent — survives reboot/app-restart because the model lives in the app's configured store. Diagnostic lesson: when Ollama returns 404/empty despite `ollama list` showing a model, check the **server log** `msg="server config"` line for the actual `OLLAMA_MODELS` path the running server uses.

### LL-P: Disable qwen3 reasoning — the real cause of empty/timeout describes (2026-05-29)
**What:** Added `OLLAMA_THINK = False` to `config/settings.py` and set `"think": OLLAMA_THINK` on both Ollama `/api/chat` payloads (describer + query parser `_llm_parse`).
**Why:** `--describe-all` kept failing with `JSON parse failed; raw=''` (empty content) and 300 s read timeouts, even with RAM free and the model resident. Systematic debugging:
  - Same crop, **short 4-field prompt → valid JSON in 10 s**; **full 17-field prompt → empty after 151 s**. So not RAM, not context size (image is ~150 tokens, num_ctx 2048 is ample), not the crop.
  - Capping `num_predict=400` returned `done_reason=length, eval_count=400, len(content)=0` — i.e. the model generated 400 tokens but emitted **none of them as content**.
  - Inspecting the response showed a populated `message.thinking` field: **qwen3-vl is a hybrid reasoning model.** Left on, it spends its token budget on an internal `<think>` monologue and, under `format=json`, never produces the JSON answer in `message.content` → empty (or times out while still "thinking"). The complex prompt triggered far more reasoning than the simple one, which is why some persons succeeded and 6a587517 always failed.
  - Setting `think=False`: the **full prompt returned complete valid 17-field JSON, `done=stop`, in ~45 s.**
**Old approach:** No `think` field → Ollama defaulted qwen3-vl to thinking-on → empty content / timeouts.
**New approach:** `think=False` forces a direct answer. Documented, config-driven; plain non-reasoning VLMs ignore the flag, so it's safe across backends.
**Result/Status:** Describe is now reliable (~45 s/person on this CPU). This — not the timeout (LL-M) or keep_alive (LL-N) — was the true blocker; those remain valid hardening. Lesson for any future qwen3/reasoning model: pass `think=False` when you need structured output.

### LL-O: Unify on one model — query parser reuses qwen3-vl:2b, rules-first (2026-05-29)
**What:** Changed `OLLAMA_QUERY_MODEL` from `qwen2.5:3b` to `qwen3-vl:2b` (same model as the describer) and reordered `QueryParser.parse()` to run the instant rule-based parser FIRST, calling the LLM only when the rules extract nothing. Extracted the LLM call into `QueryParser._llm_parse()` (returns `None` on failure). Updated `LLM_GUIDE.md` (one model to pull), `SURVEILLANT_ARCHITECTURE_REPORT.md`, and memory.
**Why:** User request — avoid downloading/managing a second model and keep the setup simple. One model for both roles is simpler to install and keep warm. Consequence handled: qwen3-vl:2b is a slow vision model (~75–110 s/call on this CPU), so calling it on every search (the old LLM-first order) would make every `--search-text` hang. The rule-based parser already covers the documented operator vocabulary (gender, build, garment+colour, headwear, glasses/beard, accessories, negation) instantly, so rules-first keeps search fast and only pays for the LLM on unusual phrasings.
**Old approach:** Separate `qwen2.5:3b` text model; `parse()` called the LLM first, rule-based only as a failure fallback.
**New approach:** Single `qwen3-vl:2b`; `parse()` = rules-first, LLM-fallback. `_llm_parse()` carries `keep_alive` and the longer `OLLAMA_VLM_TIMEOUT_SEC` like the describer.
**Result/Status:** Common searches return immediately with no model call; only rare queries load the VLM. One fewer model to download. Behaviour for the documented search examples is unchanged (rules already matched them).

### LL-N: keep_alive to stop model reloading under RAM pressure (2026-05-29)
**What:** Added `OLLAMA_KEEP_ALIVE = "30m"` to `config/settings.py` and set `keep_alive` on the describer's `/api/chat` payload.
**Why:** Even at the 300 s timeout (LL-M), `--describe-all` still timed out on most persons while one finished in 74.8 s — a 4× latency variance that indicates resource contention, not steady slowness. Diagnosis: the machine has 16 GB RAM but only ~3 GB free (Chrome ~3 GB, Antigravity IDE ~1 GB, Edge WebView ~1 GB, Claude ~1 GB). qwen3-vl:2b needs ~2 GB weights + vision-encoder activations, so loading it forces Windows to page to disk; under swap, a single describe can exceed even 300 s. GPU is Intel Iris Xe (integrated) — stock Ollama on Windows can't use it, so inference is unavoidably CPU-only.
**Old approach:** No `keep_alive`; Ollama's default 5-min unload meant a slow/swapping batch could unload the model between persons and re-pay the ~2 GB load each time, compounding the problem.
**New approach:** `keep_alive = "30m"` keeps the model resident for the whole sequential backfill, then frees the RAM. Primary mitigation remains operational: free host RAM (close Chrome/IDE) so the resident model doesn't get paged out mid-inference.
**Result/Status:** With ~6+ GB free, per-person time should stabilise at ~75–110 s. keep_alive avoids reload cost; it cannot prevent the OS paging out a resident model under genuine memory starvation — hence the RAM-freeing guidance. For large galleries, the Marlin GPU backend remains the scalable path.

### LL-M: Raise describer HTTP timeout — qwen3-vl:2b is slow on CPU (2026-05-29)
**What:** Added `OLLAMA_VLM_TIMEOUT_SEC = 300` to `config/settings.py` and wired it as the default `timeout_sec` for `QwenVLOllamaDescriber` (was a hardcoded `120`).
**Why:** After the 404 fix (LL-L), `--describe-all` reached the model and described one person in **111.9 s**, but the others failed with `Read timed out (read timeout=120)`. On this CPU, qwen3-vl:2b takes ~90–130 s per crop (vision encoder dominates) and the first call also pays a one-time model-load cost, so the 120 s read timeout was marginally too tight.
**Old approach:** `timeout_sec: int = 120` hardcoded in `QwenVLOllamaDescriber.__init__` (also a magic-number violation of WORKING_INSTRUCTIONS §5).
**New approach:** Config-driven `OLLAMA_VLM_TIMEOUT_SEC = 300` — generous headroom for slow CPU inference without hanging indefinitely on a dead backend. Failed rows are retried by the sweep up to `MAX_DESCRIPTION_ATTEMPTS`, so a one-off slow call self-heals.
**Result/Status:** Describe pass completes on CPU. ~2 min/person means a 4-person backfill ≈ 8 min; consider the Marlin GPU backend for larger galleries.

### LL-L: Fix Ollama 404 — use 127.0.0.1 instead of localhost (2026-05-29)
**What:** Changed `OLLAMA_HOST` in `config/settings.py` (and `.env.example`) from `http://localhost:11434` to `http://127.0.0.1:11434`.
**Why:** `--describe-all` failed with `404 Client Error: Not Found for url: http://localhost:11434/api/chat` for every person, even though `ollama list` showed `qwen3-vl:2b` present. Diagnosis: two `ollama serve` processes were bound to port 11434 — one on IPv4 `127.0.0.1` (had the model) and one on IPv6 `::` (empty model list). On Windows, `localhost` resolves to IPv6 `::1` first, so the app's HTTP call landed on the empty server and Ollama returned 404 model-not-found. `ollama list` worked because the CLI used the IPv4 listener. Confirmed: `GET http://127.0.0.1:11434/api/tags` returned the model; `GET http://[::1]:11434/api/tags` returned `{"models":[]}`.
**Old approach:** `OLLAMA_HOST = "http://localhost:11434"`.
**New approach:** `OLLAMA_HOST = "http://127.0.0.1:11434"` — forces IPv4, sidesteps the localhost→IPv6 ambiguity. Independent of the stray-server cleanup; correct default regardless.
**Result/Status:** Describer and query parser now reach the model-bearing server. Recommended hygiene: ensure only ONE Ollama server runs (quit duplicate `ollama serve` instances) so the IPv4/IPv6 split can't recur.

### LL-K: Describer model upgraded qwen2.5vl:2b → qwen3-vl:2b (2026-05-29)
**What:** Changed the default local describer model from `qwen2.5vl:2b` to `qwen3-vl:2b` in `config/settings.py` (`OLLAMA_VLM_MODEL`). Updated docstrings in `modules/llm/describer.py`, the setup guide `LLM_GUIDE.md`, and the current-config snapshot in `SURVEILLANT_ARCHITECTURE_REPORT.md`.
**Why:** Qwen3-VL is the newer generation; at the same 2B size it follows instructions and emits structured output more reliably, which directly benefits the strict 17-field JSON description schema. No architecture change — the prompts, schema, `Describer` ABC, and both backends are model-agnostic, so this is a single-constant swap that the default param `model=OLLAMA_VLM_MODEL` propagates automatically.
**Old approach:** `OLLAMA_VLM_MODEL = "qwen2.5vl:2b"`; setup docs instructed `ollama pull qwen2.5vl:2b`.
**New approach:** `OLLAMA_VLM_MODEL = "qwen3-vl:2b"`; setup docs instruct `ollama run qwen3-vl:2b` (downloads if missing + doubles as an interactive smoke test; SURVEILLANT itself reaches the model via the Ollama HTTP daemon, not the CLI). The text-only query parser is unchanged (`qwen2.5:3b`).
**Result/Status:** Same ~5–15 s/person on CPU (still a 2B model — quality gain, not speed). Verify the tag with `ollama list` after first run. Marlin remote backend unaffected.

### LL-A: Problem statement (2026-05-28)

**What:** Parts 1–8.5 give a stable `person_id` and a multi-view body gallery for each person, which makes "is this the same person?" queries work but leaves the system blind to semantics — "find a man in a red t-shirt and white hat" cannot be answered because the database stores embedding vectors, not appearance attributes.

**Why it matters:** A graduation demo needs a query interface the human operator can speak. Cross-camera identity tracking is invisible to a non-technical audience; natural-language search is the headline feature that makes the work feel real.

### LL-B: Model choice — Marlin-2B primary, Qwen2.5-VL fallback

**Decision (user-driven, 2026-05-28):** Use `NemoStation/Marlin-2B` (Apache-2.0, video VLM) as the primary describer, with `qwen2.5vl:2b` via local Ollama as the CPU fallback.

**Why two backends:**
- Marlin-2B is a video-trained VLM and requires a GPU per its model card. SURVEILLANT's local target is CPU-only.
- We resolve this with a remote-HTTP architecture: a small FastAPI host (`modules/llm/marlin_server/serve.py`) runs Marlin on a GPU machine (Colab notebook, cloud VM, or local workstation with NVIDIA), and SURVEILLANT POSTs base64-encoded snapshots over the network.
- When the GPU host is unreachable or `MARLIN_HOST = ""`, the system gracefully falls back to `QwenVLOllamaDescriber` so dev iterations always have a working path.
- Both backends share an abstract `Describer` interface (`modules/llm/describer.py`), so the worker code is backend-agnostic.

### LL-C: Schema — append-only `person_descriptions` table

**Decision:** Option B from the brainstorm — separate `person_descriptions` table with history, denormalised `latest_description_id` pointer on `persons`, durable `description_queue` job table.

Three new SQLite objects:

```sql
person_descriptions (id, person_id, described_at, backend, model_id,
                     snapshots_used JSON, attributes JSON, summary, confidence)
description_queue   (id, person_id, enqueued_at, status, last_error, attempts)
persons.latest_description_id INTEGER   -- denormalised pointer
```

**Why append-only:** people change appearance during a session (jacket on/off, picks up a bag). Overwriting destroys evidence that's useful at search time and for the thesis audit trail. The denormalised `latest_description_id` keeps the common-case JOIN fast.

**`merge_persons()` patched** (database.py) to re-point `person_descriptions` from `remove_id` to `keep_id`, inherit the latest pointer if `keep_id` was undescribed, and cancel any pending queue rows for `remove_id`.

### LL-D: Strict-JSON prompt with enum'd attributes

**Decision:** Both backends share the same SYSTEM + USER prompt template (`describer.py:SYSTEM_PROMPT_DESCRIBE` / `USER_PROMPT_DESCRIBE`). 17 fields, most enum-constrained:

| Field group | Examples |
|---|---|
| Demographics | `gender`, `age_range`, `body_build`, `height_class` |
| Hair / face | `hair_color`, `hair_length`, `beard`, `glasses` |
| Headwear | `headwear`, `headwear_color` |
| Top / bottom | `clothing_top`, `clothing_top_color`, `clothing_bottom`, `clothing_bottom_color` |
| Free-text | `accessories` (list), `distinctive_features`, `summary` |

A 13-color palette (`red/blue/green/yellow/black/white/gray/brown/orange/purple/pink/multi/unknown`) is the closed set for any `*_color` field. Values outside the palette are coerced to `unknown` in `_coerce_to_schema()` so SQL filters stay deterministic.

A robust `_clean_json()` helper strips ```json fences, locates `{...}` slices inside surrounding prose, and returns `None` only on unrecoverable malformations (raw text logged for the audit trail).

### LL-E: Background worker — mirrors ReconciliationWorker pattern

**File:** `modules/llm/description_worker.py`

**Pattern lifted directly from `ReconciliationWorker`** so the lifecycle is consistent across the codebase: `_stop_event`, `run_forever()`, per-task try/except → never crash the daemon.

**Loop:**
1. **Startup recovery** — any `in_progress` rows from a previous run get bumped back to `pending` (durability).
2. **Drain in-memory hint queue** (non-blocking) — wake up immediately on fresh binds.
3. **Periodic sweep** every 60 s — `get_persons_without_description()` → `enqueue_description()` so nothing falls through the cracks if the in-memory queue dropped due to backpressure.
4. **Claim + handle** — `claim_next_description()` atomically marks one pending row `in_progress`, then `_handle()` picks the best snapshot, calls the backend, inserts the result, marks the row `done`.
5. **Failure path** — `fail_description()` increments attempts; rows that hit `MAX_DESCRIPTION_ATTEMPTS = 3` are marked `failed` with the raw error preserved.

**Snapshot selection** reuses the existing `CropQualityGate` (Part 2) — filters by blur/dimension/brightness, then picks the highest Laplacian-variance crop. Zero new image-quality code.

### LL-F: Producer hook in `main.py` (one line of logic)

In the embedding worker, immediately after the snapshot crops are saved and `track_registry[key] = person_uuid` succeeds with `status_out == "new"`:

```python
if ENABLE_DESCRIPTION_WORKER and status_out == "new":
    try:
        db.enqueue_description(person_uuid)
        llm_queue.put_nowait({"person_id": person_uuid})
    except queue.Full:
        pass   # DB row is enough; sweep will catch it
```

Producer NEVER blocks. Durability lives in SQLite; the in-memory queue is just a wake-up hint.

### LL-G: Natural-language search — three-stage pipeline

**File:** `modules/search/text_search.py`

1. **`QueryParser.parse(query)`** — text-only Ollama call (`qwen2.5:3b` by default) emits a partial schema dict. Missing fields stay missing; negation lands as `"glasses": "no"`. A rule-based regex fallback runs if Ollama is unreachable.
2. **Stage 1 — SQL filter** (`db.search_persons_by_attributes`) — enum fields → exact-match via `json_extract`; phrase fields → `LIKE`; accessory lists → OR'd `LIKE`. Returns candidates with attributes denormalised.
3. **Stage 2 — soft re-rank** — synonym-aware match scoring (`crimson → red`, `hoodie → sweater`, `chubby → heavy`, etc.). Score normalised by # of filter fields the user specified.
4. **Stage 3 — semantic fallback** — only fires when Stage 1 is empty AND `ENABLE_TEXT_FALLBACK_RERANK = True`. Encodes the query and every stored summary with `sentence-transformers/all-MiniLM-L6-v2`, cosine-ranks. Lazy import so the dependency stays optional.

CLI: `python main.py --phase 4 --search-text "..."` prints ranked results with `person_id`, summary, last-seen-cam/time, snapshot path.

### LL-H: New invariants

Added to `project_invariants.md`:

| # | Rule | Why |
|---|---|---|
| 14 | DescriptionWorker NEVER blocks the embedding worker | Producer uses `put_nowait` + durable SQLite row. Even if `queue.Full` raises, the description_queue row already exists and the periodic sweep will pick it up. |
| 15 | `person_descriptions` rows ACCUMULATE | Never `UPDATE`-overwrite; every describe inserts a new row. `latest_description_id` moves forward. Re-describes preserve history (essential for thesis audit + future "what was this person wearing earlier today?" queries). |
| 16 | `description_queue.status` transitions are append-only | `pending → in_progress → done | failed`. Never moves backwards. (Exception: startup recovery may reset stuck `in_progress` rows back to `pending` because the previous worker is provably gone.) |
| 17 | `BODY_MATCH_THRESHOLD` (legacy alias) stays at `BODY_MATCH_THRESHOLD_CROSS_CAM` | Documented previously as invariant #13 in the topology section; reaffirmed here because the LLM PR added another caller that could be tempted to lower it. Phase-3 photo search and any context-less call uses this floor; live identification passes its own override. |

### LL-I: Verification

All checks pass on an in-memory database:

- Schema migrates cleanly (in-memory DB, fresh init, and `_migrate()` path).
- `enqueue_description` is idempotent — second call with pending row is a no-op.
- `claim_next_description` returns one row and marks it `in_progress`.
- `insert_description` + `complete_description` update `latest_description_id` correctly.
- `fail_description` flips to `failed` after `max_attempts` and preserves `last_error`.
- `search_persons_by_attributes` returns expected matches on enum, phrase, and accessory filters.
- `merge_persons` correctly re-points descriptions and inherits when the kept person had no description.
- `QueryParser._rule_based_fallback` parses gender, build, top color, headwear, glasses, negation correctly (with word-boundary regex — a 0-day bug from the initial implementation where "woman" matched "man" was caught and fixed).
- `TextSearchEngine.search()` round-trip works end-to-end with a stub describer.
- All seven new/touched Python files pass `ast.parse()`.

### LL-J: Status

- **`DESCRIPTION_BACKEND = "qwen-vl"`** is the default — works on CPU with no remote host required (assuming `ollama run qwen3-vl:2b` and `ollama pull qwen2.5:3b` have been run locally; see LL-K for the model upgrade).
- **Marlin remote host** ships ready-to-deploy in `modules/llm/marlin_server/` with a README covering Colab, cloud VM, and local-GPU deployment.
- **Phase 4C / 4D deferred** per the locked scope: multi-snapshot consensus, deterministic HSV color sanity-check, re-description on appearance change, hybrid photo+text reranking.
- Phase 4 CLI:
  - `python main.py --phase 4 --describe-all` — one-shot describe pass.
  - `python main.py --phase 4 --search-text "a man in a red t-shirt"` — search.
  - Both can be combined.
- During Phase 2 the description worker runs as a daemon — descriptions populate automatically as new persons are bound.

---

## Reconciliation Worker Improvements

### RW-A: Mean-pool similarity (Bug D fix)
**File:** `modules/reconciliation/worker.py`
**Old:** Max-pool — `max(similarity(a_i, b_j))` over all pairs. One accidentally-similar pair out of N×M was enough to trigger a false proposal.
**New:** Mean-pool — `mean(similarity(a_i, b_j))` over all compatible pairs. A legitimate duplicate scores high on most pairs; a false positive does not.

### RW-B: Proposal deduplication (Bug E fix)
**File:** `modules/storage/database.py` — `propose_merge()`
**Old:** `INSERT INTO merge_proposals ...` — every 120-second cycle added a new row for the same pair.
**New:** Upsert by pair — if a pending proposal for (pid_a, pid_b) or (pid_b, pid_a) exists, update its similarity score instead of inserting a duplicate.

### RW-C: Reconciliation guards
**What:** Two quality gates before running pairwise comparison:
1. Both persons must have ≥ 2 gallery embeddings (`MIN_GALLERY_FOR_RECONCILIATION = 2`).
2. Both persons must have view-coverage ≥ 0.5 (`MIN_VIEW_COVERAGE_FOR_MATCHING`).
**Why:** A 1-embedding prototype is too noisy to be a reliable merge target. A single-angle gallery can produce false matches against another person viewed from the same angle.

---

## Bug Audit Summary (all fixed)

| Bug | Root Cause | Fix Location |
|---|---|---|
| A | OSNet using ImageNet weights, not Re-ID weights | `embedder.py` — auto-download via gdown |
| B | Crop filter `d["bbox"] == t["bbox"]` always False | `main.py` — replaced with IoU ≥ 0.7 |
| C | Initial embedding tagged "initial" → view-coverage 0.0 forever | `database.py` + `main.py` — canonical tag at creation |
| D | Max-pool reconciliation generating false proposals | `worker.py` — switched to mean-pool |
| E | `propose_merge()` always INSERT → duplicate rows | `database.py` — upsert by pair |
| F | View-coverage gate in searcher blocked re-entries → duplicate IDs | `searcher.py` — gate removed from search |
| G | `CROSS_TYPE_MULTIPLIER` penalty was dead code (no face embeddings) | `searcher.py` — code removed |
| H | Mean aggregation skewed by outlier frames | `embedder.py` — switched to median |
| I | CLAHE ran unconditionally on every frame | `enhancement.py` — conditional on V-mean |
| J | Linear SQLite scan would bottleneck at 500+ persons | `searcher.py` + new `faiss_index.py` |

---

## Parts 9 & 10 — Not Yet Implemented

### Part 9 — Spatio-Temporal Camera Constraints (Planned)
**What:** A transition-time matrix `CAM_TRANSITION_MIN_SEC[(cam_a, cam_b)]` that rejects physically impossible cross-camera matches (person can't teleport from cam 3 to cam 1 in 2 seconds if the walk takes 30 seconds).
**Files to change:** `config/settings.py` (add matrix), `modules/search/searcher.py` (apply constraint).
**Expected improvement:** Eliminates a class of false positives where two similar-looking people on different cameras at the same time are wrongly merged.

### Part 10 — LLM Description Matching (Planned)
**What:** Use Qwen2.5-VL:2b (Ollama) to generate text descriptions of stored persons. Use description as a secondary matching signal when visual similarity is uncertain.
**When to trigger:** Visual similarity in range [0.55, BODY_MATCH_THRESHOLD) — uncertain zone.
**Decision logic:** Uncertain visual + matching description → accept. Uncertain visual + conflicting description → reject.
**Files:** `modules/llm/describer.py` (already scaffolded), `modules/search/searcher.py`.

---

---

## Web Interface (2026-06-18) — `webapp/` FastAPI + React dashboard (Phase 2 slice)

A web layer was added under `webapp/` (separate from the engine): a FastAPI API
that imports the engine and a React dashboard. The API serves a SEEDED COPY of
the DB (`webapp/api/seed_demo.py` copies `database/surveillant.db` + the
referenced snapshot folders into `webapp/api/data/demo/`) so a fresh engine run —
which wipes `surveillant.db` — can never destroy what the site serves.

**Engine change — `modules/storage/database.py` gained 4 methods** (for the
dashboard's correction tools; all reads/writes still go through Database, never
raw SQL from the web layer):
- `delete_person(person_id)` — single-transaction purge of the person across
  `person_embeddings`, `face_embeddings`, `camera_history`, `person_descriptions`,
  `description_queue` (+ resolves pending `merge_proposals`, deletes the `persons` row).
- `split_person(person_id, embedding_ids, history_ids=None, new_person_id=None)` —
  peels selected gallery embeddings / camera-history rows into a NEW person_id
  (fix for "two people fused into one"); recomputes `gallery_size` + `known_angles`
  on both sides.
- `get_gallery_entries_meta(person_id)` / `get_camera_history_rows(person_id)` —
  expose primary-key ids (NO vectors) so the split UI can pick which rows to move.

**Invariant note:** `delete_person` / `split_person` deliberately do NOT fire the
`on_embedding_added` / `on_merge` cache hooks. Identity-mutating web corrections
rebuild FAISS from SQLite afterward (the web API calls
`engine.invalidate_search_caches`), consistent with invariant #4 (SQLite is the
source of truth). Existing `merge_persons` is reused as-is for the merge tool.

The engine's live pipeline (`main.py`, searcher, reconciliation) is unchanged.

---

*Last updated: 2026-06-18 — Web interface Phase 2 (API + dashboard skeleton) added under `webapp/`; `database.py` gained delete_person / split_person / get_gallery_entries_meta / get_camera_history_rows for the correction tools. Engine live pipeline unchanged. (Earlier: Parts 1–8.5 + Part 10 Phase 4A+4B complete; Part 9 + Phase 4C/4D deferred.)*
