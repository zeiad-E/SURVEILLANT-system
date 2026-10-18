# SURVEILLANT

**AI-powered multi-camera surveillance intelligence system** (graduation project, Alexandria University)

SURVEILLANT watches several camera feeds at once, tracks every person across all of them, and gives each one a single identity even when they move between cameras. Operators can then find anyone by uploading a photo, searching by face, or just asking in plain language through a built-in AI assistant.

## Features

- **Multi-camera tracking:** YOLOv8 segmentation detects people and ByteTrack keeps their tracks stable through occlusions
- **Cross-camera re-identification:** OSNet embeddings with cosine similarity and a FAISS index link the same person across cameras, with a background reconciliation worker that merges split identities
- **Face recognition and face search:** InsightFace face analysis with search by face image
- **Violence detection:** real-time alerts with snapshots, short clips, and an event log
- **AI descriptions with a local LLM:** a local vision-language model (Qwen2.5-VL via Ollama) writes a description of each person, prompted to describe only what is visible
- **Semantic search:** descriptions are embedded with `all-MiniLM-L6-v2`, so searches like "man in a black jacket with a backpack" match by meaning, not keywords
- **AI chatbot assistant:** a chat interface for searching, refining results over multiple turns, opening a person's profile, and checking stats and alerts. Rule-based routing answers instantly, with an optional Ollama fallback, and any change it proposes runs only after the user confirms
- **Audio analysis:** transcription, language detection, translation, speaker identification, and spoken-threat detection
- **Web command center:** FastAPI backend with JWT auth and a React + TypeScript dashboard with live cameras, people, search, alerts, audio, and the assistant

## Architecture

```
Cameras ──> Detection (YOLOv8-seg) ──> Tracking (ByteTrack)
                                             │
                    ┌────────────────────────┼──────────────────────────┐
                    ▼                        ▼                          ▼
          Embedding worker          Description worker           Violence worker
         (OSNet Re-ID, FAISS)   (Qwen2.5-VL via Ollama, text    (alerts, clips, log)
                    │              embeddings for search)
                    ▼                        │
       Reconciliation worker ──> SQLite <────┘
                                   │
                         FastAPI (REST + chatbot)
                                   │
                         React + TypeScript dashboard
```

The pipeline is multi-threaded: fast work (detection and tracking) runs separately from slow work (embeddings, reconciliation, and LLM descriptions), so the cameras never wait on the AI models.

## Tech stack

| Area | Tools |
|---|---|
| Detection and tracking | YOLOv8 (Ultralytics), ByteTrack, OpenCV |
| Re-identification and search | OSNet (torchreid), FAISS, scikit-learn, sentence-transformers |
| Faces | InsightFace, ONNX Runtime |
| LLMs | Ollama, Qwen2.5-VL |
| Audio | Whisper, ECAPA-TDNN, MarianMT |
| Backend | Python, FastAPI, SQLAlchemy, SQLite, JWT |
| Frontend | React, TypeScript, Vite, Tailwind CSS, TanStack Query |

## Project structure

```
surveillant/            Core engine
├── main.py             Entry point (runs the pipeline phases)
├── config/             Thresholds and paths
└── modules/            camera, detection, tracking, embedding, reconciliation,
                        face, violence, llm, search, audio, storage
webapp/
├── api/                FastAPI backend, routers, and chatbot
└── web/                React + TypeScript dashboard
grad_project_book/      Graduation project book (LaTeX + PDF)
```

## Running it

Requires Python 3.12, Node.js, and [Ollama](https://ollama.com) with `qwen2.5vl:3b` pulled. Model weights for some modules are kept outside the repo because of their size.

```bash
# Engine
cd surveillant
pip install -r requirements.txt
python main.py --phase 2 --videos data/videos/cam1.mp4 data/videos/cam2.mp4   # track people
python main.py --phase 4 --describe-all                                     # generate descriptions

# API (from the repo root)
pip install -r webapp/api/requirements.txt
python -m webapp.api.seed_demo          # one-time demo data
uvicorn webapp.api.main:app --port 8000 # docs at /docs

# Dashboard
cd webapp/web
npm install
npm run dev
```

## Author

**Zeiad Emad**: [LinkedIn](https://www.linkedin.com/in/zeiad-emad/) · [GitHub](https://github.com/zeiad-E)
