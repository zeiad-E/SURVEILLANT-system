"""
webapp/api/main.py
------------------
SURVEILLANT web API — FastAPI application.

Run from the repo root:
    uvicorn webapp.api.main:app --reload

Interactive docs (and de-facto API reference) at /docs.
"""

import sys

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from . import config

# The engine emits non-ASCII diagnostics (─, ↔, →) via print(). On a Windows
# cp1252 console that raises UnicodeEncodeError and would crash a request.
# Force UTF-8 on the server's stdout/stderr (guarded — no-op if unsupported).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

app = FastAPI(
    title="SURVEILLANT API",
    version="0.1.0",
    description=(
        "Web API over the SURVEILLANT multi-camera person re-identification "
        "engine. Serves a seeded copy of the engine's database; never re-runs "
        "AI per request."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _ensure_demo_account():
    """Idempotently make sure the demo tenant + user exist (so login works
    even before seed_demo is run with the auth module present)."""
    try:
        from .auth import store

        store.seed_demo_account()
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] could not seed demo account: {exc}")


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["meta"])
def health():
    """Liveness + a quick confirmation the seeded DB is readable."""
    from .engine import get_database

    try:
        db = get_database()
        return {
            "status": "ok",
            "demo_db": str(config.DEMO_DB_PATH),
            "persons": len(db.get_all_person_ids()),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "error",
            "demo_db": str(config.DEMO_DB_PATH),
            "detail": str(exc),
            "hint": "Run:  python -m webapp.api.seed_demo",
        }


# ── Static media (snapshot images) ──────────────────────────────────────────
# Unauthenticated bytes mount so <img> tags work. Fine for the single-tenant
# demo; per-tenant signed URLs are the later multi-tenant upgrade.
from fastapi.staticfiles import StaticFiles  # noqa: E402

config.DEMO_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
app.mount(
    "/media/snapshots",
    StaticFiles(directory=str(config.DEMO_SNAPSHOTS_DIR)),
    name="snapshots",
)

# ── Routers ─────────────────────────────────────────────────────────────────
from .auth.router import router as auth_router  # noqa: E402
from .routers import audio, cameras, corrections, persons, search, stats  # noqa: E402
from .chatbot.router import router as chat_router  # noqa: E402

app.include_router(auth_router)
app.include_router(persons.router)
app.include_router(stats.router)
app.include_router(cameras.router)
app.include_router(search.router)
app.include_router(corrections.router)
app.include_router(chat_router)
app.include_router(audio.router)
