"""
webapp/api/engine.py
--------------------
Bridge between the FastAPI app and the SURVEILLANT engine.

This is the ONLY place the engine is wired in. It:
  1. Puts surveillant/ on sys.path so the engine's top-level `config` and
     `modules` packages import cleanly.
  2. Exposes lazy, cached singletons for the Database and the heavy ML
     components (OSNet embedder, FAISS index, InsightFace analyzer, searchers).

Design rules:
  - Heavy imports (torch, insightface, faiss) happen INSIDE functions, never at
    module top — so importing this module (and starting the API) stays fast.
  - All DB access goes through the engine's invariant-safe `Database` class.
  - Search caches are keyed by db path (the multi-tenant seam). After an
    identity-mutating correction, call invalidate_search_caches() so the next
    photo search rebuilds FAISS from SQLite (SQLite is the source of truth).
"""

import sys

from . import config

# Make the engine importable: surveillant/ holds the top-level `config` and
# `modules` packages the engine code imports by bare name.
if str(config.SURVEILLANT_DIR) not in sys.path:
    sys.path.insert(0, str(config.SURVEILLANT_DIR))


# ── caches ──────────────────────────────────────────────────────────────────
_db_cache: dict = {}
_faiss_cache: dict = {}
_person_searcher_cache: dict = {}
_face_searcher_cache: dict = {}
_embedder = None          # process-wide (model weights shared across tenants)
_face_analyzer = None      # process-wide


def _key(db_path=None) -> str:
    return str(db_path or config.DEMO_DB_PATH)


# ── Database (light — no torch) ─────────────────────────────────────────────
def get_database(db_path=None):
    """Return a cached Database for the given path (defaults to the demo DB)."""
    from modules.storage.database import Database  # noqa: PLC0415

    key = _key(db_path)
    db = _db_cache.get(key)
    if db is None:
        db = Database(key)
        _db_cache[key] = db
    return db


# ── Natural-language semantic search (light — MiniLM loads inside .search) ──
def get_text_search_engine(db):
    from modules.search.text_search import TextSearchEngine  # noqa: PLC0415

    return TextSearchEngine(db)


# ── Body Re-ID (heavy: OSNet + FAISS) ───────────────────────────────────────
def get_embedder():
    global _embedder
    if _embedder is None:
        from modules.embedding.embedder import PersonEmbedder  # noqa: PLC0415

        _embedder = PersonEmbedder()
    return _embedder


def get_faiss_index(db, key: str):
    from modules.search.faiss_index import FAISSIndex  # noqa: PLC0415

    idx = _faiss_cache.get(key)
    if idx is None:
        idx = FAISSIndex()
        idx.rebuild_from_db(db)
        _faiss_cache[key] = idx
    return idx


def get_person_searcher(db_path=None):
    from modules.search.searcher import PersonSearcher  # noqa: PLC0415

    key = _key(db_path)
    s = _person_searcher_cache.get(key)
    if s is None:
        db = get_database(key)
        s = PersonSearcher(db, get_embedder(), faiss_index=get_faiss_index(db, key))
        _person_searcher_cache[key] = s
    return s


# ── Face search (heavy: InsightFace) ────────────────────────────────────────
def get_face_analyzer():
    global _face_analyzer
    if _face_analyzer is None:
        from modules.face.face_analyzer import FaceAnalyzer  # noqa: PLC0415

        _face_analyzer = FaceAnalyzer()
    return _face_analyzer


def get_face_searcher(db_path=None):
    from modules.face.face_searcher import FaceSearcher  # noqa: PLC0415

    key = _key(db_path)
    s = _face_searcher_cache.get(key)
    if s is None:
        s = FaceSearcher(get_database(key), get_face_analyzer())
        _face_searcher_cache[key] = s
    return s


# ── Audio intelligence (heavy: Whisper + ECAPA-TDNN + MarianMT) ─────────────
_audio_analyzer = None


def get_audio_analyzer():
    """Process-wide cached AudioAnalyzer (lazy — models load on first analyze)."""
    global _audio_analyzer
    if _audio_analyzer is None:
        from modules.audio import AudioAnalyzer  # noqa: PLC0415

        _audio_analyzer = AudioAnalyzer()
    return _audio_analyzer


def audio_availability() -> dict:
    """Report whether audio analysis can run and what packages are missing."""
    try:
        from modules.audio import availability  # noqa: PLC0415

        return availability()
    except Exception as exc:  # noqa: BLE001
        return {"enabled": False, "available": False, "error": str(exc),
                "missing_core": [], "missing_speaker": []}


# ── Cache invalidation after identity-mutating corrections ──────────────────
def invalidate_search_caches(db_path=None):
    """Drop cached FAISS + searchers for this DB so the next search rebuilds
    from SQLite. Call after merge / delete / split."""
    key = _key(db_path)
    _faiss_cache.pop(key, None)
    _person_searcher_cache.pop(key, None)
    _face_searcher_cache.pop(key, None)
