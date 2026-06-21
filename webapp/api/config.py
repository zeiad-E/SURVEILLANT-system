"""
webapp/api/config.py
--------------------
Central configuration for the SURVEILLANT web API.

Everything is env-overridable so the same code runs locally or in a container
(decision: "deployment TBD"). Defaults point at a seeded *copy* of the engine's
data so a fresh engine run (which wipes surveillant/database/surveillant.db)
can never destroy the demo the API serves.
"""

import os
from pathlib import Path

# ── Filesystem anchors ──────────────────────────────────────────────────────
API_DIR         = Path(__file__).resolve().parent          # webapp/api
WEBAPP_DIR      = API_DIR.parent                            # webapp
PROJECT_ROOT    = WEBAPP_DIR.parent                         # repo root
SURVEILLANT_DIR = PROJECT_ROOT / "surveillant"             # the engine package


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


# ── Seeded demo data (what the API serves) ──────────────────────────────────
DATA_DIR           = _env_path("SURVEILLANT_API_DATA", API_DIR / "data" / "demo")
DEMO_DB_PATH       = _env_path("SURVEILLANT_API_DB", DATA_DIR / "surveillant.db")
DEMO_SNAPSHOTS_DIR = _env_path("SURVEILLANT_API_SNAPSHOTS", DATA_DIR / "snapshots")
DEMO_ALERTS_LOG    = _env_path("SURVEILLANT_API_ALERTS", DATA_DIR / "violence_log.json")
AUTH_DB_PATH       = _env_path("SURVEILLANT_API_AUTH_DB", DATA_DIR / "auth.db")

# ── Source (engine) data to seed FROM ───────────────────────────────────────
ENGINE_DB_PATH        = SURVEILLANT_DIR / "database" / "surveillant.db"
ENGINE_SNAPSHOTS_DIR  = SURVEILLANT_DIR / "data" / "snapshots"
ENGINE_VIDEOS_DIR     = SURVEILLANT_DIR / "data" / "videos"
ENGINE_ALERTS_LOG     = SURVEILLANT_DIR / "alerts" / "violence_log.json"

# ── Live-cams: which video files to loop (one per camera) ───────────────────
# Defaults to the WiseNet set used by the default engine run (video1_1..5.avi).
DEMO_CAMERA_VIDEOS = [
    ENGINE_VIDEOS_DIR / f"video1_{i}.avi" for i in range(1, 6)
]

# ── Overlay recorder sidecars (real detection boxes/IDs per frame) ──────────
# Produced offline by `python -m webapp.api.tools.record_overlays`; consumed by
# the MJPEG endpoint to burn boxes into the stream when ?overlay=1.
OVERLAYS_DIR = _env_path("SURVEILLANT_API_OVERLAYS", DATA_DIR / "overlays")

# ── Audio intelligence: bundled demo sample clips (12 WAVs, speakers j/m) ────
AUDIO_SAMPLES_DIR = _env_path("SURVEILLANT_API_AUDIO_SAMPLES", API_DIR / "data" / "audio_samples")


def overlay_sidecar(cam_id: int) -> Path:
    """Path to the per-camera overlay JSON sidecar."""
    return OVERLAYS_DIR / f"cam{cam_id}.json"

# ── Auth / JWT ──────────────────────────────────────────────────────────────
JWT_SECRET     = os.environ.get("SURVEILLANT_JWT_SECRET", "dev-insecure-change-me")
JWT_ALG        = "HS256"
JWT_EXPIRE_MIN = int(os.environ.get("SURVEILLANT_JWT_EXPIRE_MIN", "720"))  # 12h

# Demo account seeded by seed_demo.py (and the tenant it belongs to).
DEMO_USER_EMAIL    = os.environ.get("SURVEILLANT_DEMO_EMAIL", "demo@surveillant.ai")
DEMO_USER_PASSWORD = os.environ.get("SURVEILLANT_DEMO_PASSWORD", "demo1234")
DEMO_TENANT_ID     = "demo"
DEMO_TENANT_NAME   = "Demo Deployment"

# ── CORS (Vite dev server by default) ───────────────────────────────────────
CORS_ORIGINS = os.environ.get(
    "SURVEILLANT_CORS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

# ── Camera topology (read from the engine settings at runtime) ──────────────
# Exposed via /cameras so the journey map can draw overlap groups. engine.py
# puts surveillant/ on sys.path, making the engine's top-level `config` package
# importable; call this only after engine has been imported (i.e. at request time).
def overlap_groups() -> list[list[int]]:
    """Return CAMERA_OVERLAP_GROUPS from the engine settings as JSON-able lists."""
    try:
        from config.settings import CAMERA_OVERLAP_GROUPS  # type: ignore
    except Exception:
        return []
    return [sorted(g) for g in CAMERA_OVERLAP_GROUPS]
