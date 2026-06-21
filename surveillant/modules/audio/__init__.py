"""
surveillant.modules.audio
-------------------------
Audio-intelligence modality (Part 13) — Whisper speech-to-text, multi-engine
language detection, MarianMT/Google translation, ECAPA-TDNN speaker recognition,
and multilingual spoken-threat detection.

ADDITIVE + ISOLATED: nothing here touches body/face identity or tracking. See
`analyzer.AudioAnalyzer` for the file-based facade and `availability()` for a
dependency check. The original CLI (live mic, enrolment, etc.) remains in
`audio_processor.py` and is runnable directly with `python -m
surveillant.modules.audio.audio_processor`.
"""

from .analyzer import AudioAnalyzer, availability

__all__ = ["AudioAnalyzer", "availability"]
