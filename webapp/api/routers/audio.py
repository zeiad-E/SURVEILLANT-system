"""
webapp/api/routers/audio.py
---------------------------
Audio intelligence (Part 13): transcription, language detection, translation,
speaker recognition and spoken-threat detection over an uploaded clip or one of
the bundled demo samples.

Heavy models load lazily on first analyze. If the audio packages are absent (or
the feature is disabled) the endpoints return 503 with the missing-package list
— never a 500/crash. On a detected spoken threat an entry is appended to the same
alerts log the vision violence feed uses, so it also surfaces on the Alerts page.
"""

import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from .. import config, engine
from ..auth.deps import get_current_user

router = APIRouter(prefix="/audio", tags=["audio"])

_ALLOWED_UPLOAD = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def _samples() -> list:
    d = config.AUDIO_SAMPLES_DIR
    if not d.exists():
        return []
    return [{"name": p.name, "size": p.stat().st_size} for p in sorted(d.glob("*.wav"))]


def _safe_sample_path(name: str) -> Path:
    base = config.AUDIO_SAMPLES_DIR.resolve()
    target = (base / name).resolve()
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="bad sample path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="sample not found")
    return target


@router.get("/status")
def status_(user=Depends(get_current_user)):
    """Whether audio analysis can run, plus any missing packages."""
    return engine.audio_availability()


@router.get("/samples")
def samples(user=Depends(get_current_user)):
    """List the bundled demo clips."""
    return {"items": _samples()}


@router.get("/samples/{name}")
def sample_file(name: str):
    """Serve a bundled sample WAV (for in-page <audio> playback). Unauthenticated
    like the camera stream — these are non-sensitive demo clips and an <audio>
    tag cannot send an Authorization header."""
    target = _safe_sample_path(name)
    return FileResponse(str(target), media_type="audio/wav", filename=name)


def _append_audio_alert(result: dict) -> None:
    """On a spoken threat, append to the shared alerts log (Alerts-page feed)."""
    threat = result.get("threat") or {}
    if not threat.get("detected"):
        return
    try:
        log = config.DEMO_ALERTS_LOG
        existing = []
        if log.exists():
            try:
                existing = json.loads(log.read_text(encoding="utf-8"))
            except Exception:
                existing = []
        if not isinstance(existing, list):
            existing = []
        kws = ", ".join(threat.get("keywords", []))
        existing.append({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cam_id": None,
            "score": 1.0,
            "level": "VIOLENCE",
            "label": f"Audio threat ({result.get('speaker', '?')}): {kws}",
            "source": "audio",
            "keywords": threat.get("keywords", []),
            "transcript": (result.get("transcript") or "")[:300],
        })
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(json.dumps(existing[-200:], indent=2), encoding="utf-8")
    except Exception:
        pass


@router.post("/analyze")
def analyze(
    sample: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    language: Optional[str] = Form(None),  # "" / "auto" = auto-detect; else force (e.g. "en", "ar")
    user=Depends(get_current_user),
):
    """Analyze an uploaded clip OR a bundled sample. Sync (runs in a threadpool)
    because Whisper inference is blocking and can take a while on CPU."""
    avail = engine.audio_availability()
    if not avail.get("available"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Audio analysis is unavailable.",
                "enabled": avail.get("enabled"),
                "missing": avail.get("missing_core", []),
                "hint": (
                    "Set ENABLE_AUDIO_ANALYSIS=True and install: "
                    "openai-whisper torch torchaudio speechbrain transformers "
                    "langdetect deep-translator"
                ),
            },
        )

    tmp_path = None
    try:
        if file is not None:
            ext = Path(file.filename or "").suffix.lower()
            if ext not in _ALLOWED_UPLOAD:
                raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")
            fd, tmp_path = tempfile.mkstemp(suffix=ext or ".wav")
            with open(fd, "wb") as out:
                shutil.copyfileobj(file.file, out)
            wav_path = tmp_path
        elif sample:
            wav_path = str(_safe_sample_path(sample))
        else:
            raise HTTPException(status_code=400, detail="Provide a 'file' upload or a 'sample' name.")

        force_lang = "" if (language or "").lower() in ("", "auto") else language.lower()
        try:
            result = engine.get_audio_analyzer().analyze_file(wav_path, force_lang=force_lang)
        except RuntimeError as exc:
            # Missing model / dependency surfaced at run time.
            raise HTTPException(status_code=503, detail=str(exc))

        _append_audio_alert(result)
        result["source"] = "upload" if file is not None else f"sample:{sample}"
        return result
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass
