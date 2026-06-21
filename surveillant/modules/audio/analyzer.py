"""
surveillant/modules/audio/analyzer.py
-------------------------------------
Import-safe facade over the vendored audio pipeline (`audio_processor.py`) for
FILE-BASED analysis. Heavy models (Whisper, ECAPA-TDNN, MarianMT) load lazily on
first use; when a capability's packages are missing it is reported rather than
crashing the caller.

ISOLATION INVARIANT: audio is an ADDITIVE modality. Nothing here reads or writes
the person / embedding tables or tracking state — it only produces a transcript,
language, translation, speaker name and spoken-threat verdict. Cross-camera body
and face identity are never influenced by audio.
"""

from __future__ import annotations

import importlib
from typing import Optional

from config.settings import (
    ENABLE_AUDIO_ANALYSIS,
    AUDIO_WHISPER_MODEL,
    AUDIO_TARGET_LANG,
    AUDIO_TRANSLATE,
    AUDIO_SPEAKER_ID,
    AUDIO_VIOLENCE_CHECK,
    AUDIO_SIMILARITY_THRESHOLD,
    AUDIO_SPEAKER_PROFILES,
    AUDIO_FAST_DECODE,
    AUDIO_DISAMBIGUATE_LANG,
)


def _missing(pkgs) -> list:
    out = []
    for p in pkgs:
        try:
            importlib.import_module(p)
        except Exception:
            out.append(p)
    return out


# Packages each capability needs (heavy — checked, never imported eagerly here).
_CORE_PKGS    = ["whisper", "torch"]                 # transcription (blocking)
_SPEAKER_PKGS = ["torch", "torchaudio", "speechbrain"]  # speaker-id only


def availability() -> dict:
    """Whether audio analysis can run, and what (if anything) is missing."""
    missing_core = _missing(_CORE_PKGS)
    return {
        "enabled":        bool(ENABLE_AUDIO_ANALYSIS),
        "available":      bool(ENABLE_AUDIO_ANALYSIS) and not missing_core,
        "missing_core":   missing_core,                 # blocks transcription
        "missing_speaker": _missing(_SPEAKER_PKGS),     # speaker-id only
        "whisper_model":  AUDIO_WHISPER_MODEL,
        "target_lang":    AUDIO_TARGET_LANG,
    }


class AudioAnalyzer:
    """Reusable, lazy file-based audio analyzer."""

    def __init__(
        self,
        whisper_model: Optional[str] = None,
        target_lang: Optional[str] = None,
        translate: Optional[bool] = None,
        speaker_id: Optional[bool] = None,
        violence_check: Optional[bool] = None,
    ):
        self.whisper_model  = whisper_model or AUDIO_WHISPER_MODEL
        self.target_lang    = target_lang or AUDIO_TARGET_LANG
        self.translate      = AUDIO_TRANSLATE if translate is None else translate
        self.speaker_id     = AUDIO_SPEAKER_ID if speaker_id is None else speaker_id
        self.violence_check = AUDIO_VIOLENCE_CHECK if violence_check is None else violence_check
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from .audio_processor import AudioProcessingPipeline, SpeakerRecognizer

        use_speaker = self.speaker_id and not _missing(_SPEAKER_PKGS)
        pipe = AudioProcessingPipeline(
            whisper_model        = self.whisper_model,
            target_lang          = self.target_lang,
            translate            = self.translate,
            speaker_id           = use_speaker,
            violence_check       = self.violence_check,
            similarity_threshold = AUDIO_SIMILARITY_THRESHOLD,
            fast_decode          = AUDIO_FAST_DECODE,
            disambiguate_lang    = AUDIO_DISAMBIGUATE_LANG,
        )
        # Point the speaker recognizer at the bundled ECAPA profiles.
        if use_speaker:
            pipe.recognizer = SpeakerRecognizer(
                profiles_file=str(AUDIO_SPEAKER_PROFILES),
                threshold=AUDIO_SIMILARITY_THRESHOLD,
            )
        self._pipeline = pipe
        return pipe

    def analyze_file(self, wav_path: str, force_lang: str = "") -> dict:
        """Run the full pipeline on a WAV file and return a normalised result.

        `force_lang` (e.g. "en", "ar") forces Whisper to transcribe in that
        language instead of auto-detecting — use it when the caller knows the
        language (auto-detect is ambiguous for accented/bilingual speech). Empty
        string = auto-detect."""
        if not ENABLE_AUDIO_ANALYSIS:
            raise RuntimeError(
                "Audio analysis is disabled. Set ENABLE_AUDIO_ANALYSIS=True in "
                "config/settings.py.")
        missing = _missing(_CORE_PKGS)
        if missing:
            raise RuntimeError(
                f"Audio analysis needs: {', '.join(missing)}. "
                f"Run: pip install openai-whisper torch")
        pipe = self._get_pipeline()
        # Per-request language override (set every call so it resets to auto).
        pipe.stt.force_lang = (force_lang or "").strip().lower()
        raw = pipe.run(wav_file=str(wav_path))
        return self._normalize(raw)

    @staticmethod
    def _normalize(raw: dict) -> dict:
        """Flatten the nested pipeline output into a stable, UI-friendly shape."""
        stt   = raw.get("transcription") or {}
        lang  = raw.get("language") or {}
        trans = raw.get("translation")
        viol  = raw.get("violence") or {}
        speaker = raw.get("speaker", "N/A")

        return {
            "transcript": stt.get("text", ""),
            "speaker": speaker,
            "speaker_known": speaker not in ("Unknown Person", "N/A", "", None),
            "language": {
                "code": lang.get("language_code") or stt.get("language"),
                "name": lang.get("language_name"),
                "confidence": lang.get("confidence"),
                "whisper_lang": lang.get("whisper_lang"),
                "whisper_conf": lang.get("whisper_conf"),
            },
            "translation": None if not trans else {
                "text": trans.get("translated_text", ""),
                "engine": trans.get("engine"),
                "target_lang": trans.get("target_lang"),
                "source_lang": trans.get("source_lang"),
            },
            "threat": {
                "detected": bool(viol.get("threat_detected")),
                "keywords": viol.get("matched_keywords", []),
                "alert_sent": bool(viol.get("alert_sent")),
            },
            "whisper": {
                "model": stt.get("backend"),
                "confidence": stt.get("confidence"),
                "segments": stt.get("segments"),
            },
        }
