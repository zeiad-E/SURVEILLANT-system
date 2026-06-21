"""
Audio Processing Pipeline  –  High-Accuracy Language Detection Edition
=======================================================================

KEY FIXES (v3 – Microphone/No-Headphone Fix)
─────────────────────────────────────────────
PROBLEM: Without headphones, speaker feedback causes Whisper to receive a
degraded signal. The script_heuristic in LanguageClassifier then wrongly
forces "ar" because distorted audio produces Arabic-looking Unicode noise.
Meanwhile Whisper's own language detection (which runs on mel-spectrogram,
not decoded text) is FAR more robust to this distortion.

FIX 1 – WHISPER ALWAYS WINS FOR LATIN LANGUAGES:
  classify() now checks Whisper confidence BEFORE script_heuristic.
  If Whisper confidence ≥ WHISPER_LANG_CONFIDENCE_LOW (0.40) the script
  heuristic is only allowed to confirm, not override.

FIX 2 – SCRIPT HEURISTIC GUARDED BY WHISPER:
  If Whisper says a Latin language (en/fr/de/es/…) with any confidence,
  script_heuristic is skipped entirely (noise was making text look Arabic).

FIX 3 – OPTIONAL NOISE REDUCTION PRE-PROCESSING:
  If `noisereduce` is installed, a noise-reduction pass is applied to the
  WAV before transcription when --noise-reduce flag is used.
  Install: pip install noisereduce

FIX 4 – FORCE-LANGUAGE OPTION:
  --force-lang XX  forces Whisper to transcribe in language XX, bypassing
  auto-detection entirely (useful for known single-language environments).

FIX 5 – SILENCE GATE RAISED:
  SILENCE_THRESH raised from 500 → 800 to avoid capturing ambient hum
  from speakers as "speech" in silence-stop mode.

SPEAKER RECOGNITION UPGRADE (v2 – Maximum Accuracy)  [unchanged]
────────────────────────────────────────────────────
1. FULL EMBEDDING BANK (nearest-neighbour, not centroid)
2. ADAPTIVE PER-SPEAKER THRESHOLD (mean − 1.0 × std, floor 0.35)
3. BACKGROUND SCORE NORMALISATION (margin ≥ 0.05 → unconditional accept)
4. TEST-TIME AUGMENTATION VOTE (5 embeddings per query)
5. VERBOSE DIAGNOSTICS

Install:
    pip install speechbrain torch torchaudio librosa soundfile
    pip install transformers sentencepiece sacremoses
    pip install langdetect langid deep-translator pyaudio numpy
    pip install openai-whisper
    pip install noisereduce   # optional but recommended without headphones
"""

import argparse
import collections
import datetime
import math
import os
import pickle
import re
import smtplib
import struct
import sys
import tempfile
import wave
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

# ── Third-party dependencies ─────────────────────────────────────────────────
# numpy is a hard dependency of the whole engine, so it is always present.
import numpy as np

# pyaudio is ONLY needed for live microphone capture (AudioCapture). For
# file-based analysis on a server it may be absent — import softly so the rest
# of the module still loads (INTEGRATION: was a hard sys.exit).
try:
    import pyaudio
    _HAS_PYAUDIO = True
except Exception:
    pyaudio = None
    _HAS_PYAUDIO = False

# langdetect — one of several language-ID engines; optional.
try:
    from langdetect import detect, LangDetectException
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    _HAS_LANGDETECT = True
except Exception:
    detect = None

    class LangDetectException(Exception):
        ...

    _HAS_LANGDETECT = False

# deep-translator — Google fallback translator; optional (MarianMT still works).
try:
    from deep_translator import GoogleTranslator
    _HAS_DEEP_TRANSLATOR = True
except Exception:
    GoogleTranslator = None
    _HAS_DEEP_TRANSLATOR = False

# INTEGRATION: this module prints emoji diagnostics (💻 🎤 🌐 …). On a Windows
# cp1252 console that raises UnicodeEncodeError mid-pipeline and aborts the run.
# Force UTF-8 on stdout/stderr (guarded — no-op if unsupported), so the module is
# safe whether driven from the CLI, the web server, or a bare `python -c`.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:
        pass

# INTEGRATION: on Windows without Developer Mode, huggingface_hub/speechbrain
# default to SYMLINK when caching models → WinError 1314 ("required privilege not
# held"). Force copy instead so ECAPA loads on a plain account. Must be set before
# huggingface_hub is imported.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# ════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════
SAMPLE_RATE    = 16_000
CHANNELS       = 1
CHUNK          = 1_024
RECORD_SECONDS = 5
FORMAT         = pyaudio.paInt16 if _HAS_PYAUDIO else None
SILENCE_THRESH = 800          # raised from 500 – avoids speaker hum

PROFILES_FILE        = "speaker_profiles_ecapa.pkl"
SIMILARITY_THRESHOLD = 0.55
AUGMENT_ENROLMENT    = True
WHISPER_MODEL_SIZE   = "large-v3"

# ── Speaker recognition tuning ───────────────────────────────────────────────
ENROL_STD_MULTIPLIER   = 1.0
ENROL_THRESHOLD_FLOOR  = 0.35
MARGIN_UNCONDITIONAL   = 0.05
TTA_ENABLED            = True

# ── Language detection thresholds ────────────────────────────────────────────
# FIX: lowered HIGH threshold so Whisper wins more often
WHISPER_LANG_CONFIDENCE_HIGH  = 0.70   # was 0.85
WHISPER_LANG_CONFIDENCE_LOW   = 0.35   # was 0.40

# Latin-script language codes — script heuristic SKIPPED if Whisper says one of these
LATIN_LANGS = {
    "en","fr","de","es","it","pt","nl","pl","sv","fi","da","no",
    "ro","cs","hu","sk","hr","sl","lt","lv","et","tr","id","ms",
    "vi","af","cy","eu","ca","gl","sq","is","ga","mt","lb",
}

# ── Force language override (set via --force-lang CLI arg) ───────────────────
FORCE_LANGUAGE: str = ""   # e.g. "en" or "fr"

MARIAN_MODEL_MAP = {
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("it", "en"): "Helsinki-NLP/opus-mt-it-en",
    ("pt", "en"): "Helsinki-NLP/opus-mt-pt-en",
    ("nl", "en"): "Helsinki-NLP/opus-mt-nl-en",
    ("ru", "en"): "Helsinki-NLP/opus-mt-ru-en",
    ("ar", "en"): "Helsinki-NLP/opus-mt-ar-en",
    ("zh", "en"): "Helsinki-NLP/opus-mt-zh-en",
    ("ja", "en"): "Helsinki-NLP/opus-mt-ja-en",
    ("ko", "en"): "Helsinki-NLP/opus-mt-ko-en",
    ("tr", "en"): "Helsinki-NLP/opus-mt-tr-en",
    ("pl", "en"): "Helsinki-NLP/opus-mt-pl-en",
    ("sv", "en"): "Helsinki-NLP/opus-mt-sv-en",
    ("fi", "en"): "Helsinki-NLP/opus-mt-fi-en",
    ("da", "en"): "Helsinki-NLP/opus-mt-da-en",
    ("no", "en"): "Helsinki-NLP/opus-mt-no-en",
    ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en",
    ("he", "en"): "Helsinki-NLP/opus-mt-he-en",
    ("uk", "en"): "Helsinki-NLP/opus-mt-uk-en",
    ("cs", "en"): "Helsinki-NLP/opus-mt-cs-en",
    ("ro", "en"): "Helsinki-NLP/opus-mt-ro-en",
    ("hu", "en"): "Helsinki-NLP/opus-mt-hu-en",
    ("el", "en"): "Helsinki-NLP/opus-mt-el-en",
    ("id", "en"): "Helsinki-NLP/opus-mt-id-en",
    ("vi", "en"): "Helsinki-NLP/opus-mt-vi-en",
    ("en", "ar"): "Helsinki-NLP/opus-mt-en-ar",
    ("fr", "ar"): "Helsinki-NLP/opus-mt-fr-ar",
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("ar", "fr"): "Helsinki-NLP/opus-mt-ar-fr",
    ("de", "fr"): "Helsinki-NLP/opus-mt-de-fr",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("fr", "de"): "Helsinki-NLP/opus-mt-fr-de",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("fr", "es"): "Helsinki-NLP/opus-mt-fr-es",
    ("en", "zh"): "Helsinki-NLP/opus-mt-en-zh",
    ("en", "ru"): "Helsinki-NLP/opus-mt-en-ru",
    ("en", "pt"): "Helsinki-NLP/opus-mt-en-pt",
    ("en", "tr"): "Helsinki-NLP/opus-mt-en-tr",
}

VIOLENCE_KEYWORDS = [
    "kill", "attack", "bomb", "gun", "shoot", "knife",
    "fight", "terror", "explosion", "murder", "dead",
    "blood", "threat",
]

VIOLENCE_KEYWORDS_MULTILINGUAL = {
    "ar": [
        "اقتل", "قتل", "هجوم", "قنبلة", "مسدس", "بندقية", "إطلاق نار",
        "سكين", "خنجر", "قتال", "إرهاب", "انفجار", "جريمة قتل", "دم",
        "تهديد", "موت", "ذبح", "تفجير", "اغتيال", "حرب", "دمار",
    ],
    "fr": [
        "tuer", "tué", "attaque", "bombe", "pistolet", "fusil", "tirer",
        "couteau", "bagarre", "terrorisme", "explosion", "meurtre", "sang",
        "menace", "mort", "massacrer", "assassiner", "attentat", "violence",
        "arme", "guerre",
    ],
    "de": [
        "töten", "getötet", "angriff", "bombe", "pistole", "gewehr", "schießen",
        "messer", "kampf", "terror", "explosion", "mord", "blut", "bedrohung",
        "tot", "massaker", "attentäter", "krieg", "waffe", "gewalt",
    ],
    "es": [
        "matar", "mató", "ataque", "bomba", "pistola", "fusil", "disparar",
        "cuchillo", "pelea", "terrorismo", "explosión", "asesinato", "sangre",
        "amenaza", "muerto", "masacre", "asesino", "guerra", "arma", "violencia",
    ],
    "it": [
        "uccidere", "ucciso", "attacco", "bomba", "pistola", "fucile", "sparare",
        "coltello", "rissa", "terrorismo", "esplosione", "omicidio", "sangue",
        "minaccia", "morto", "massacro", "assassino", "guerra", "arma", "violenza",
    ],
    "pt": [
        "matar", "matou", "ataque", "bomba", "pistola", "fuzil", "atirar",
        "faca", "briga", "terrorismo", "explosão", "assassinato", "sangue",
        "ameaça", "morto", "massacre", "assassino", "guerra", "arma", "violência",
    ],
    "nl": [
        "doden", "dood", "aanval", "bom", "pistool", "geweer", "schieten",
        "mes", "gevecht", "terrorisme", "explosie", "moord", "bloed",
        "bedreiging", "dood", "massamoord", "aanslag", "oorlog", "wapen",
    ],
    "ru": [
        "убить", "убийство", "нападение", "бомба", "пистолет", "оружие",
        "стрелять", "нож", "драка", "терроризм", "взрыв", "убийца",
        "кровь", "угроза", "смерть", "резня", "война", "насилие",
    ],
    "zh": [
        "杀", "杀死", "攻击", "炸弹", "枪", "射击", "刀", "刺",
        "打架", "恐怖", "爆炸", "谋杀", "血", "威胁", "死亡",
        "屠杀", "暗杀", "战争", "武器", "暴力",
    ],
    "ja": [
        "殺す", "殺人", "攻撃", "爆弾", "銃", "射撃", "ナイフ", "刃物",
        "暴力", "テロ", "爆発", "殺害", "血", "脅迫", "死亡", "虐殺",
        "戦争", "武器",
    ],
    "ko": [
        "죽이다", "살인", "공격", "폭탄", "총", "사격", "칼", "폭력",
        "테러", "폭발", "살해", "피", "위협", "사망", "학살", "전쟁", "무기",
    ],
    "hi": [
        "मारना", "हत्या", "हमला", "बम", "बंदूक", "गोली", "चाकू",
        "लड़ाई", "आतंक", "विस्फोट", "खून", "धमकी", "मौत", "नरसंहार",
        "युद्ध", "हथियार", "हिंसा",
    ],
    "tr": [
        "öldürmek", "cinayet", "saldırı", "bomba", "silah", "tabanca",
        "ateş etmek", "bıçak", "kavga", "terör", "patlama", "kan",
        "tehdit", "ölüm", "katliam", "savaş", "şiddet",
    ],
    "pl": [
        "zabić", "zabójstwo", "atak", "bomba", "pistolet", "broń",
        "strzelać", "nóż", "rvačka", "terroryzm", "eksplozja", "morderstwo",
        "krew", "groźba", "śmierć", "masakr", "válka", "násilí",
    ],
    "sv": [
        "döda", "mord", "attack", "bomb", "pistol", "skjuta", "kniv",
        "slagsmål", "terrorism", "explosion", "blod", "hot", "död",
        "massakra", "krig", "vapen", "våld",
    ],
    "fi": [
        "tappaa", "murha", "hyökkäys", "pommi", "pistooli", "ampua",
        "veitsi", "tappelu", "terrorismi", "räjähdys", "veri", "uhkaus",
        "kuolema", "verilöyly", "sota", "ase", "väkivalta",
    ],
    "da": [
        "dræbe", "mord", "angreb", "bombe", "pistol", "skyde", "kniv",
        "slagsmål", "terrorisme", "eksplosion", "blod", "trussel", "død",
        "massakre", "krig", "våben", "vold",
    ],
    "no": [
        "drepe", "mord", "angrep", "bombe", "pistol", "skyte", "kniv",
        "slagsmål", "terrorisme", "eksplosjon", "blod", "trussel", "død",
        "massakre", "krig", "våpen", "vold",
    ],
    "he": [
        "להרוג", "רצח", "התקפה", "פצצה", "אקדח", "לירות", "סכין",
        "קרב", "טרור", "פיצוץ", "דם", "איום", "מוות", "טבח",
        "מלחמה", "נשק", "אלימות",
    ],
    "uk": [
        "вбити", "вбивство", "напад", "бомба", "пістолет", "зброя",
        "стріляти", "ніж", "бійка", "тероризм", "вибух", "кров",
        "загроза", "смерть", "різанина", "війна", "насильство",
    ],
    "cs": [
        "zabít", "vražda", "útok", "bomba", "pistole", "zbraň",
        "střílet", "nůž", "rvačka", "terorismus", "výbuch", "krev",
        "hrozba", "smrt", "masakr", "válka", "násilí",
    ],
    "ro": [
        "omorî", "crimă", "atac", "bombă", "pistol", "armă",
        "împușca", "cuțit", "bătaie", "terorism", "explozie", "sânge",
        "amenințare", "moarte", "masacru", "război", "violență",
    ],
    "hu": [
        "megölni", "gyilkosság", "támadás", "bomba", "pisztoly", "fegyver",
        "lőni", "kés", "verekedés", "terrorizmus", "robbanás", "vér",
        "fenyegetés", "halál", "mészárlás", "háború", "erőszak",
    ],
    "el": [
        "σκοτώνω", "φόνος", "επίθεση", "βόμβα", "πιστόλι", "όπλο",
        "πυροβολώ", "μαχαίρι", "καβγάς", "τρομοκρατία", "έκρηξη", "αίμα",
        "απειλή", "θάνατος", "σφαγή", "πόλεμος", "βία",
    ],
    "id": [
        "membunuh", "pembunuhan", "serangan", "bom", "pistol", "senjata",
        "menembak", "pisau", "perkelahian", "terorisme", "ledakan", "darah",
        "ancaman", "mati", "pembantaian", "perang", "kekerasan",
    ],
    "vi": [
        "giết", "giết người", "tấn công", "bom", "súng", "bắn", "dao",
        "đánh nhau", "khủng bố", "vụ nổ", "máu", "đe dọa", "chết",
        "thảm sát", "chiến tranh", "vũ khí", "bạo lực",
    ],
    "fa": [
        "کشتن", "قتل", "حمله", "بمب", "تپانچه", "اسلحه", "شلیک",
        "چاقو", "درگیری", "تروریسم", "انفجار", "خون", "تهدید",
        "مرگ", "کشتار", "جنگ", "خشونت",
    ],
    "th": [
        "ฆ่า", "การฆาตกรรม", "โจมตี", "ระเบิด", "ปืน", "ยิง", "มีด",
        "ต่อสู้", "ก่อการร้าย", "การระเบิด", "เลือด", "ภัยคุกคาม",
        "ความตาย", "การสังหารหมู่", "สงคราม", "อาวุธ", "ความรุนแรง",
    ],
    "bn": [
        "হত্যা", "আক্রমণ", "বোমা", "বন্দুক", "গুলি", "ছুরি",
        "মারামারি", "সন্ত্রাস", "বিস্ফোরণ", "রক্ত", "হুমকি",
        "মৃত্যু", "গণহত্যা", "যুদ্ধ", "অস্ত্র", "সহিংসতা",
    ],
    "ms": [
        "membunuh", "pembunuhan", "serangan", "bom", "pistol", "senjata",
        "menembak", "pisau", "pergaduhan", "keganasan", "letupan", "darah",
        "ancaman", "mati", "pembantaian", "perang", "kekerasan",
    ],
}

# INTEGRATION: align alert credentials with the vision side's SURVEILLANT_ALERT_*
# env vars (no recipient is hard-coded; falls back to the legacy address only if
# nothing is configured). Empty sender/password => email disabled.
ALERT_RECIPIENT       = (os.getenv("SURVEILLANT_ALERT_RECEIVER")
                         or os.getenv("ALERT_RECIPIENT", "emad.ayoub2@gmail.com"))
ALERT_SENDER_EMAIL    = (os.getenv("SURVEILLANT_ALERT_SENDER")
                         or os.getenv("ALERT_SENDER_EMAIL", ""))
ALERT_SENDER_PASSWORD = (os.getenv("SURVEILLANT_ALERT_PASSWORD")
                         or os.getenv("ALERT_SENDER_PASSWORD", ""))
SMTP_HOST             = "smtp.gmail.com"
SMTP_PORT             = 587

LANGUAGE_NAMES = {
    "en": "English",  "ar": "Arabic",   "fr": "French",
    "de": "German",   "es": "Spanish",  "it": "Italian",
    "zh-cn": "Chinese (Simplified)", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean",   "pt": "Portuguese",
    "ru": "Russian",  "tr": "Turkish",  "nl": "Dutch",
    "pl": "Polish",   "sv": "Swedish",  "fi": "Finnish",
    "no": "Norwegian","da": "Danish",   "hi": "Hindi",
    "bn": "Bengali",  "he": "Hebrew",   "fa": "Persian",
    "th": "Thai",     "vi": "Vietnamese","uk": "Ukrainian",
    "cs": "Czech",    "ro": "Romanian", "hu": "Hungarian",
    "el": "Greek",    "id": "Indonesian","ms": "Malay",
}

RTL_LANGS = {"ar", "he", "fa", "ur"}


# ════════════════════════════════════════════════════════════════════════════
#  NOISE REDUCTION HELPER  (FIX 3)
# ════════════════════════════════════════════════════════════════════════════

def apply_noise_reduction(wav_path: str, tmp_dir: str) -> str:
    """
    Apply spectral noise reduction to the WAV file.
    Returns path to cleaned WAV (may be same as input if noisereduce unavailable).
    Uses the first 0.5s as noise profile (room noise / speaker hum).
    """
    try:
        import noisereduce as nr
    except ImportError:
        print("  ℹ️   noisereduce not installed — skipping noise reduction.")
        print("       Install: pip install noisereduce")
        return wav_path

    try:
        import soundfile as sf
    except ImportError:
        try:
            import scipy.io.wavfile as scwav
            rate, data = scwav.read(wav_path)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / 32768.0
            noise_sample = data[:int(rate * 0.5)]
            reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample,
                                      prop_decrease=0.85, stationary=False)
            out = os.path.join(tmp_dir, "nr_" + os.path.basename(wav_path))
            scwav.write(out, rate, (reduced * 32767).astype(np.int16))
            print("  ✅  Noise reduction applied (scipy).")
            return out
        except Exception as e:
            print(f"  ⚠️   Noise reduction failed: {e}")
            return wav_path

    try:
        data, rate = sf.read(wav_path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        noise_sample = data[:int(rate * 0.5)]
        reduced = nr.reduce_noise(y=data, sr=rate, y_noise=noise_sample,
                                  prop_decrease=0.85, stationary=False)
        out = os.path.join(tmp_dir, "nr_" + os.path.basename(wav_path))
        sf.write(out, reduced, rate)
        print("  ✅  Noise reduction applied.")
        return out
    except Exception as e:
        print(f"  ⚠️   Noise reduction failed: {e}")
        return wav_path


# ════════════════════════════════════════════════════════════════════════════
#  AUDIO LOADING (INTEGRATION)
# ════════════════════════════════════════════════════════════════════════════
def _load_audio_mono(path: str, target_sr: int = SAMPLE_RATE) -> "np.ndarray":
    """Load audio as mono float32 at target_sr using **soundfile**.

    Deliberately avoids librosa: once speechbrain 1.x is imported, its broken
    `k2_fsa` lazy module poisons `inspect.stack()` (which librosa's lazy_loader
    calls), making every `librosa.load` raise. soundfile is a plain CFFI reader
    (no ffmpeg, no lazy_loader), so it is immune.

    CRITICAL: the bundled clips are 48 kHz and Whisper/ECAPA expect 16 kHz. The
    downsample MUST be anti-aliased — a naive linear (np.interp) decimation at a
    3:1 ratio folds high frequencies back into the speech band and turns clean
    audio into mush (Whisper then hallucinates / mis-detects the language). Use
    scipy's polyphase resampler; fall back to linear only if scipy is absent."""
    import soundfile as sf
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    data = np.asarray(data, dtype=np.float32)
    if data.ndim > 1:                      # stereo → mono
        data = data.mean(axis=1)
    if sr != target_sr and data.size > 1:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), int(target_sr))
            data = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
        except Exception:                  # dependency-free fallback (poorer)
            n_out = int(round(data.size * target_sr / sr))
            if n_out > 0:
                xp = np.linspace(0.0, 1.0, num=data.size, endpoint=False)
                xq = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
                data = np.interp(xq, xp, data).astype(np.float32)
    return np.ascontiguousarray(data, dtype=np.float32)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 0 – SPEAKER RECOGNITION  (ECAPA-TDNN  v2 – maximum accuracy)    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SpeakerRecognizer:
    _SPKR_MODEL = "speechbrain/spkrec-ecapa-voxceleb"

    def __init__(self, profiles_file=PROFILES_FILE,
                 threshold=SIMILARITY_THRESHOLD):
        self.profiles_file    = profiles_file
        self.global_threshold = threshold
        self._encoder         = None
        self._tmp_dir         = tempfile.mkdtemp(prefix="ecapa_")
        # INTEGRATION: dep check deferred to first embed (so the pipeline can be
        # constructed with speaker-id disabled even when speechbrain is absent).

    @staticmethod
    def _check_deps():
        missing = []
        for pkg in ("torch", "torchaudio", "speechbrain"):
            try: __import__(pkg)
            except ImportError: missing.append(pkg)
        if missing:
            # INTEGRATION: raise (catchable) instead of sys.exit, which would
            # terminate a web-server worker.
            raise RuntimeError(
                f"Speaker recognition needs: {', '.join(missing)}. "
                f"Run: pip install {' '.join(missing)}")

    @staticmethod
    def _neutralize_k2_lazy_module():
        """speechbrain 1.x registers a `k2_fsa` lazy module whose attribute access
        raises ImportError when k2 isn't installed (no Windows wheels). That
        poisons inspect.stack()/getmodule and breaks model loading + unrelated
        libs (librosa). k2 is NOT needed for ECAPA, so replace it with a benign
        stub so any attribute access just returns nothing."""
        import sys as _sys, types as _types
        for _name in list(_sys.modules):
            if _name.endswith("k2_fsa") or _name.endswith("integrations.k2"):
                stub = _types.ModuleType(_name)
                stub.__file__ = "<neutralized>"
                _sys.modules[_name] = stub

    def _get_encoder(self):
        if self._encoder is not None:
            return self._encoder
        self._check_deps()
        print("  📥  Loading ECAPA-TDNN …")
        # Import speechbrain first (this registers its lazy integration modules),
        # then neutralise the broken k2_fsa one BEFORE any model load touches it.
        try:
            from speechbrain.inference import SpeakerRecognition, EncoderClassifier
        except Exception:
            from speechbrain.pretrained import SpeakerRecognition, EncoderClassifier
        self._neutralize_k2_lazy_module()
        kw = dict(source=self._SPKR_MODEL,
                  savedir=os.path.join(self._tmp_dir, "ecapa_model"),
                  run_opts={"device": self._device()})
        # Copy (not symlink) into savedir — avoids WinError 1314 on Windows.
        try:
            from speechbrain.utils.fetching import LocalStrategy
            kw["local_strategy"] = LocalStrategy.COPY
        except Exception:
            pass
        try:
            self._encoder = SpeakerRecognition.from_hparams(**kw)
        except Exception:
            self._encoder = EncoderClassifier.from_hparams(**kw)
        self._neutralize_k2_lazy_module()
        print("  ✅  ECAPA-TDNN ready.")
        return self._encoder

    @staticmethod
    def _device():
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception: return "cpu"

    def _embed(self, wav_path: str) -> np.ndarray:
        import torch
        # INTEGRATION: load via soundfile (not torchaudio.load, which needs
        # TorchCodec/ffmpeg in torchaudio 2.9+, and not librosa, which is broken
        # once speechbrain's k2_fsa lazy module is registered).
        y = _load_audio_mono(wav_path, SAMPLE_RATE)
        waveform = torch.from_numpy(y).float().unsqueeze(0)  # [1, T]
        encoder = self._get_encoder()
        with torch.no_grad():
            if hasattr(encoder, "encode_batch"):
                vec = encoder.encode_batch(waveform).squeeze().cpu().numpy()
            else:
                feats = encoder.mods.compute_features(waveform)
                feats = encoder.mods.mean_var_norm(
                    feats, torch.ones(feats.shape[0], device=feats.device))
                vec = encoder.mods.embedding_model(feats).squeeze().cpu().numpy()
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 1e-8 else vec

    @staticmethod
    def _augment_wav(wav_path: str, tmp_dir: str,
                     light: bool = False) -> list:
        try:
            import librosa
        except ImportError:
            return []
        aug_paths = []
        try:
            y, sr = librosa.load(wav_path, sr=SAMPLE_RATE, mono=True)
            semitones = [-1, 1] if light else [-2, -1, 1, 2]
            rates     = [0.95, 1.05] if light else [0.90, 0.95, 1.05, 1.10]
            variants  = []
            for st in semitones:
                variants.append(librosa.effects.pitch_shift(y, sr=sr, n_steps=st))
            for rate in rates:
                variants.append(librosa.effects.time_stretch(y, rate=rate))
            for i, v in enumerate(variants):
                out = os.path.join(
                    tmp_dir, f"aug_{i}_{os.path.basename(wav_path)}")
                try:
                    import soundfile as sf
                    sf.write(out, v, sr)
                except ImportError:
                    import scipy.io.wavfile as scwav
                    scwav.write(out, sr, (v * 32767).astype(np.int16))
                aug_paths.append(out)
        except Exception as e:
            print(f"    ⚠  Augmentation skipped: {e}")
        return aug_paths

    def _load_profiles(self) -> dict:
        if os.path.exists(self.profiles_file):
            with open(self.profiles_file, "rb") as f:
                return pickle.load(f)
        return {}

    def _save_profiles(self, profiles: dict) -> None:
        with open(self.profiles_file, "wb") as f:
            pickle.dump(profiles, f)

    def enroll(self, name: str, audio_paths: list) -> None:
        print(f"\n── Enrolling '{name}' with ECAPA-TDNN (v2) ──")
        embeddings = []

        for path in audio_paths:
            if not os.path.exists(path):
                print(f"  ⚠  Not found: {path}  (skipped)")
                continue
            print(f"  Processing: {path}")
            try:
                emb = self._embed(path)
                embeddings.append(emb)
                print(f"    ✔  Original embedded  shape={emb.shape}")
            except Exception as e:
                print(f"    ✘  Failed: {e}")
                continue

            if AUGMENT_ENROLMENT:
                for ap in self._augment_wav(path, self._tmp_dir, light=False):
                    try:
                        embeddings.append(self._embed(ap))
                    except Exception:
                        pass
                    finally:
                        try: os.unlink(ap)
                        except Exception: pass

        if not embeddings:
            print(f"  ✘  No valid audio for '{name}'.")
            return

        stack    = np.vstack(embeddings)
        centroid = stack.mean(axis=0)
        norm     = np.linalg.norm(centroid)
        centroid = (centroid / norm) if norm > 1e-8 else centroid

        cos_scores  = stack @ centroid
        enrol_mean  = float(cos_scores.mean())
        enrol_std   = float(cos_scores.std())
        per_spk_thr = max(
            ENROL_THRESHOLD_FLOOR,
            enrol_mean - ENROL_STD_MULTIPLIER * enrol_std
        )

        profiles = self._load_profiles()
        profiles[name] = {
            "embeddings":    stack,
            "centroid":      centroid,
            "threshold":     per_spk_thr,
            "enrol_mean":    enrol_mean,
            "enrol_std":     enrol_std,
            "n_clips":       len(audio_paths),
            "n_embeddings":  len(embeddings),
            "embedding_dim": centroid.shape[0],
        }
        self._save_profiles(profiles)
        print(
            f"  ✔  '{name}' enrolled  |  {len(embeddings)} embeddings  |  "
            f"cos_mean={enrol_mean:.3f}  cos_std={enrol_std:.3f}  "
            f"thr={per_spk_thr:.3f}"
        )

    def _score_one_embedding(self, query_emb: np.ndarray,
                             profiles: dict) -> dict:
        scores = {}
        for name, data in profiles.items():
            bank = data["embeddings"]
            cos_scores = bank @ query_emb
            scores[name] = float(cos_scores.max())
        return scores

    def identify(self, wav_path: str, verbose: bool = True) -> str:
        profiles = self._load_profiles()
        if not profiles:
            if verbose: print("  ⚠  No speakers enrolled.")
            return "Unknown Person"

        query_embeddings = []
        try:
            query_embeddings.append(self._embed(wav_path))
        except Exception as e:
            print(f"  ✘  Embedding failed: {e}")
            return "Unknown Person"

        if TTA_ENABLED:
            for ap in self._augment_wav(wav_path, self._tmp_dir, light=True):
                try:
                    query_embeddings.append(self._embed(ap))
                except Exception:
                    pass
                finally:
                    try: os.unlink(ap)
                    except Exception: pass

        if verbose:
            print(f"\n  Running identification on {len(query_embeddings)} "
                  f"embedding(s) …")

        vote_tally: dict = collections.defaultdict(int)
        all_scores: list = []

        for i, q_emb in enumerate(query_embeddings):
            scores = self._score_one_embedding(q_emb, profiles)
            all_scores.append(scores)
            sorted_scores = sorted(scores.items(), key=lambda x: -x[1])

            best_name,  best_score  = sorted_scores[0]
            second_name, second_score = (sorted_scores[1]
                                         if len(sorted_scores) > 1
                                         else (None, -1))
            best_thr = profiles[best_name].get("threshold",
                                               self.global_threshold)

            margin            = best_score - second_score
            unconditional_win = (second_name is not None and
                                 margin >= MARGIN_UNCONDITIONAL)

            if best_score >= best_thr or unconditional_win:
                decision = best_name
            else:
                decision = "Unknown Person"

            vote_tally[decision] += 1

            if verbose:
                flag   = "✔" if decision != "Unknown Person" else "✘"
                reason = ("margin" if unconditional_win and best_score < best_thr
                          else "threshold")
                print(
                    f"    Emb[{i}] → {best_name} "
                    f"(score={best_score:.4f}, thr={best_thr:.4f}, "
                    f"margin={margin:.4f}) [{flag} {reason}] → {decision}"
                )

        final = max(vote_tally, key=vote_tally.__getitem__)

        if verbose:
            avg_scores = {}
            for name in profiles:
                avg_scores[name] = float(
                    np.mean([s[name] for s in all_scores]))
            sorted_avg = sorted(avg_scores.items(), key=lambda x: -x[1])

            print(f"\n  Average cosine scores across {len(query_embeddings)} "
                  f"embedding(s):")
            print(f"  {'Speaker':<22} {'Avg Score':>10}  {'Threshold':>10}  "
                  f"{'Votes':>6}")
            print("  " + "─" * 56)
            for name, avg_s in sorted_avg:
                thr = profiles[name].get("threshold", self.global_threshold)
                v   = vote_tally.get(name, 0)
                print(f"  {name:<22} {avg_s:10.4f}  {thr:10.4f}  {v:6d}")
            unknown_v = vote_tally.get("Unknown Person", 0)
            print(f"  {'Unknown Person':<22} {'':10}  {'':10}  "
                  f"{unknown_v:6d}")

            icon = "✅ ACCEPTED" if final != "Unknown Person" else "❌ REJECTED"
            print(f"\n  → Final decision (majority vote): "
                  f"{final}  [{icon}]  "
                  f"({vote_tally[final]}/{len(query_embeddings)} votes)")

        return final

    def list_speakers(self) -> list:
        return list(self._load_profiles().keys())

    def delete_speaker(self, name: str) -> bool:
        profiles = self._load_profiles()
        if name in profiles:
            del profiles[name]
            self._save_profiles(profiles)
            return True
        return False

    def speaker_info(self) -> None:
        profiles = self._load_profiles()
        if not profiles:
            print("  (no speakers enrolled)")
            return
        print(f"\n  {'Name':<22} {'Dim':>5}  {'N_emb':>6}  "
              f"{'Enrol_cos':>10}  {'Threshold':>10}")
        print("  " + "─" * 62)
        for name, d in profiles.items():
            print(
                f"  {name:<22} {d.get('embedding_dim','-'):>5}  "
                f"{d.get('n_embeddings','-'):>6}  "
                f"{d.get('enrol_mean',0):.4f}      "
                f"{d.get('threshold', self.global_threshold):.4f}"
            )


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3 – HIGH-ACCURACY LANGUAGE CLASSIFICATION  (v3 – FIXED)         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_TRIGRAM_HINTS = {
    "en": ["the","ing","tion","ght","wh","ould","ness","ship","ough","ment"],
    "fr": ["les","des","que","eux","ais","oir","eau","ant","ait","aux"],
    "de": ["sch","ung","ich","cht","eit","nde","rch","mme","sse","halb"],
    "es": ["ción","que","los","ado","ell","ñon","tra","nte","aba","ien"],
    "it": ["gli","llo","lla","zione","nte","cci","ggi","que","tti","abb"],
    "pt": ["ção","lho","nha","ões","ado","que","uma","mos","nte","ara"],
    "nl": ["ijk","aar","oor","ees","van","een","dag","ook","ang","ell"],
    "ru": ["что","это","его","как","для","все","она","они","так","или"],
    "ar": ["ال","من","في","على","إل","هذ","وال","لا","ما","عن"],
    "zh": ["的","了","是","在","有","我","他","这","来","不"],
    "ja": ["の","は","を","に","て","が","で","た","と","い"],
    "ko": ["이","는","을","가","에","의","도","로","한","이"],
    "hi": ["ने","का","में","की","कि","है","हैं","से","और","को"],
    "tr": ["lar","ler","için","var","bir","ile","dan","den","nin","nin"],
    "pl": ["nie","się","jak","że","rze","prz","dzi","ści","ąc","iel"],
    "sv": ["och","att","som","ett","ska","bli","ade","ens","ing","lla"],
    "fi": ["nen","lla","lla","sta","kin","lle","ssa","ksi","kaan","han"],
    "da": ["det","og","til","sig","dsk","dte","hed","igt","rer","jer"],
    "no": ["det","og","til","seg","lig","nen","ing","vel","kes","ble"],
}


def _dominant_script(text: str):
    counts: dict = collections.defaultdict(int)
    for ch in text:
        if   "\u0600" <= ch <= "\u06FF": counts["ar"] += 1
        elif "\u0590" <= ch <= "\u05FF": counts["he"] += 1
        elif "\u0900" <= ch <= "\u097F": counts["hi"] += 1
        elif "\u0E00" <= ch <= "\u0E7F": counts["th"] += 1
        elif "\u0400" <= ch <= "\u04FF": counts["ru"] += 1
        elif "\u0370" <= ch <= "\u03FF": counts["el"] += 1
        elif ("\u4E00" <= ch <= "\u9FFF" or "\u3040" <= ch <= "\u30FF"):
            counts["zh"] += 1
        elif "\uAC00" <= ch <= "\uD7A3": counts["ko"] += 1
        elif ch.isalpha():               counts["__latin__"] += 1
    if not counts: return None
    best      = max(counts, key=counts.__getitem__)
    total     = sum(counts.values())
    dominance = counts[best] / total if total else 0
    kana = sum(1 for c in text if "\u3040" <= c <= "\u30FF")
    cjk  = sum(1 for c in text if "\u4E00" <= c <= "\u9FFF")
    if kana > 0 and cjk > 0: return "ja"
    if best == "zh" and kana == 0: return "zh"
    if best == "__latin__" or dominance < 0.60: return None
    return best


def _trigram_score(text: str, lang: str) -> float:
    hints = _TRIGRAM_HINTS.get(lang, [])
    if not hints: return 0.0
    t = text.lower()
    return sum(1 for h in hints if h in t) / len(hints)


class LanguageClassifier:
    _WHISPER_WEIGHT_MAX = 0.90

    def __init__(self, history_len: int = 1):
        self._history          = collections.deque(maxlen=max(1, history_len))
        self._fasttext_model   = None
        self._ft_available     = self._try_load_fasttext()
        self._langid_available = self._try_import_langid()

    @staticmethod
    def _try_import_langid() -> bool:
        try: import langid; return True
        except ImportError: return False

    def _try_load_fasttext(self) -> bool:
        try:
            import fasttext, warnings
            p = os.getenv("FASTTEXT_LID_MODEL", "lid.176.bin")
            if os.path.exists(p):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self._fasttext_model = fasttext.load_model(p)
                return True
        except Exception: pass
        return False

    def _engine_langdetect(self, text: str) -> list:
        try:
            from langdetect import detect_langs
            r = detect_langs(text)
            return [(r[0].lang, float(r[0].prob))] if r else []
        except Exception: return []

    def _engine_langid(self, text: str) -> list:
        if not self._langid_available: return []
        try:
            import langid
            lang, conf = langid.classify(text)
            prob = max(0.30, min(0.85, (conf + 50.0) / 45.0 * 0.55 + 0.30))
            return [(lang, prob)]
        except Exception: return []

    def _engine_fasttext(self, text: str) -> list:
        if not self._ft_available or not self._fasttext_model: return []
        try:
            labels, probs = self._fasttext_model.predict(
                text.replace("\n", " "), k=3)
            return [(l.replace("__label__", ""), float(p))
                    for l, p in zip(labels, probs)]
        except Exception: return []

    def _engine_trigram(self, text: str) -> list:
        scores = {l: _trigram_score(text, l) for l in _TRIGRAM_HINTS}
        top    = sorted(scores.items(), key=lambda x: -x[1])[:3]
        total  = sum(s for _, s in top)
        return [(l, s/total) for l, s in top] if total else []

    def _soft_vote(self, engine_results: list) -> dict:
        tally, total_w = collections.defaultdict(float), 0.0
        for w, results in engine_results:
            if not results: continue
            total_w += w
            for lang, prob in results:
                tally[lang] += w * prob
        if not tally or total_w == 0: return {}
        return {k: v / total_w for k, v in tally.items()}

    def _smooth(self, code: str, conf: float):
        self._history.append((code, conf))
        tally = collections.defaultdict(float)
        for c, f in self._history: tally[c] += f
        best = max(tally, key=tally.__getitem__)
        return best, min(tally[best] / len(self._history), 0.99)

    @staticmethod
    def _parse_whisper_probs(whisper_result: dict) -> tuple:
        lang_probs = whisper_result.get("language_probs", {})
        if lang_probs:
            top_lang = max(lang_probs, key=lang_probs.__getitem__)
            top_conf = float(lang_probs[top_lang])
            return top_lang, top_conf, lang_probs
        lang = whisper_result.get("language", "")
        if lang:
            return lang, 0.70, {lang: 0.70}
        return "", 0.0, {}

    def classify(self, text: str, whisper_result: dict = None) -> dict:
        # ── FORCE LANGUAGE OVERRIDE (CLI --force-lang) ────────────────────
        if FORCE_LANGUAGE:
            code = FORCE_LANGUAGE.lower()
            return {
                "language_code": code,
                "language_name": LANGUAGE_NAMES.get(code, code.upper()),
                "confidence":    1.0,
                "method":        "forced",
                "engines_used":  ["force_lang"],
                "whisper_lang":  code,
                "whisper_conf":  1.0,
            }

        if not text or len(text.strip()) < 5:
            return {
                "language_code": "unknown", "language_name": "Unknown",
                "confidence": 0.0, "method": "too_short",
                "engines_used": [], "whisper_lang": "", "whisper_conf": 0.0,
            }

        text = text.strip()

        # ── Parse Whisper result early ────────────────────────────────────
        whisper_lang, whisper_conf, whisper_probs = ("", 0.0, {})
        if whisper_result:
            whisper_lang, whisper_conf, whisper_probs = \
                self._parse_whisper_probs(whisper_result)

        # ── FIX 1+2: If Whisper confidently says a Latin language,
        #    skip script_heuristic entirely (noise was faking Arabic text)
        #    and return Whisper's answer directly if conf is high enough.
        if whisper_lang and whisper_conf >= WHISPER_LANG_CONFIDENCE_HIGH:
            code, conf = self._smooth(whisper_lang, whisper_conf)
            return {
                "language_code": code,
                "language_name": LANGUAGE_NAMES.get(code, code.upper()),
                "confidence":    round(conf, 3),
                "method":        "whisper_primary",
                "engines_used":  ["whisper"],
                "whisper_lang":  whisper_lang,
                "whisper_conf":  round(whisper_conf, 3),
            }

        # ── Script heuristic — only run if Whisper is NOT saying Latin lang ─
        #    This prevents noise-distorted audio from being labeled Arabic
        #    when Whisper already detected (e.g.) English or French.
        whisper_says_latin = whisper_lang in LATIN_LANGS
        script_lang = _dominant_script(text)

        if script_lang and not whisper_says_latin:
            # Script clearly non-Latin and Whisper also agrees → trust it
            code, conf = self._smooth(script_lang, 0.98)
            return {
                "language_code": code,
                "language_name": LANGUAGE_NAMES.get(code, code.upper()),
                "confidence":    round(conf, 3),
                "method":        "script_heuristic",
                "engines_used":  ["unicode_script"],
                "whisper_lang":  whisper_lang,
                "whisper_conf":  round(whisper_conf, 3),
            }

        # ── Text ensemble ─────────────────────────────────────────────────
        text_engine_results = []
        engines_used        = []

        if self._ft_available:
            r = self._engine_fasttext(text)
            if r:
                text_engine_results.append((0.45, r))
                engines_used.append("fasttext")

        r = self._engine_langid(text)
        if r:
            text_engine_results.append(
                (0.40 if self._ft_available else 0.55, r))
            engines_used.append("langid")

        r = self._engine_langdetect(text)
        if r:
            text_engine_results.append(
                (0.30 if self._ft_available else 0.35, r))
            engines_used.append("langdetect")

        r = self._engine_trigram(text)
        if r:
            text_engine_results.append((0.10, r))
            engines_used.append("trigram")

        text_tally = self._soft_vote(text_engine_results)

        # ── Blend with Whisper (low confidence range) ─────────────────────
        if whisper_lang and whisper_conf >= WHISPER_LANG_CONFIDENCE_LOW:
            w_whisper = self._WHISPER_WEIGHT_MAX * (
                (whisper_conf - WHISPER_LANG_CONFIDENCE_LOW) /
                (WHISPER_LANG_CONFIDENCE_HIGH - WHISPER_LANG_CONFIDENCE_LOW)
            )
            w_whisper = min(w_whisper, self._WHISPER_WEIGHT_MAX)
            w_text    = 1.0 - w_whisper

            blended: dict = collections.defaultdict(float)
            for lang, prob in text_tally.items():
                blended[lang] += w_text * prob
            if whisper_probs:
                for lang, prob in whisper_probs.items():
                    blended[lang] += w_whisper * prob
            else:
                blended[whisper_lang] += w_whisper

            total = sum(blended.values())
            if total > 0:
                blended = {k: v / total for k, v in blended.items()}

            best_code = max(blended, key=blended.__getitem__)
            best_conf = round(blended[best_code], 3)
            best_code, best_conf = self._smooth(best_code, best_conf)

            engines_used.insert(0, "whisper")
            return {
                "language_code": best_code,
                "language_name": LANGUAGE_NAMES.get(best_code, best_code.upper()),
                "confidence":    round(best_conf, 3),
                "method":        f"whisper_blend(w={w_whisper:.2f})+text",
                "engines_used":  engines_used,
                "whisper_lang":  whisper_lang,
                "whisper_conf":  round(whisper_conf, 3),
            }

        if not text_tally:
            # Last resort: if Whisper said anything at all, use it
            if whisper_lang:
                code, conf = self._smooth(whisper_lang, whisper_conf if whisper_conf > 0 else 0.5)
                return {
                    "language_code": code,
                    "language_name": LANGUAGE_NAMES.get(code, code.upper()),
                    "confidence":    round(conf, 3),
                    "method":        "whisper_fallback",
                    "engines_used":  ["whisper"],
                    "whisper_lang":  whisper_lang,
                    "whisper_conf":  round(whisper_conf, 3),
                }
            return {
                "language_code": "unknown", "language_name": "Unknown",
                "confidence": 0.0, "method": "no_engines",
                "engines_used": [], "whisper_lang": whisper_lang,
                "whisper_conf": round(whisper_conf, 3),
            }

        best_code = max(text_tally, key=text_tally.__getitem__)
        best_conf = round(text_tally[best_code], 3)
        best_code, best_conf = self._smooth(best_code, best_conf)

        return {
            "language_code": best_code,
            "language_name": LANGUAGE_NAMES.get(best_code, best_code.upper()),
            "confidence":    round(best_conf, 3),
            "method":        "text_ensemble",
            "engines_used":  engines_used,
            "whisper_lang":  whisper_lang,
            "whisper_conf":  round(whisper_conf, 3),
        }

    def reset_history(self):
        self._history.clear()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5 – VIOLENCE DETECTION  (multilingual)                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ViolenceDetector:
    def __init__(self, keywords=None, recipient=ALERT_RECIPIENT,
                 sender=ALERT_SENDER_EMAIL, password=ALERT_SENDER_PASSWORD,
                 smtp_host=SMTP_HOST, smtp_port=SMTP_PORT):
        self.keywords  = [k.lower() for k in (keywords or VIOLENCE_KEYWORDS)]
        self.recipient = recipient
        self.sender    = sender
        self.password  = password
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port

    def _get_lang_keywords(self, detected_lang: str) -> list:
        lang     = detected_lang.split("-")[0].lower() if detected_lang else "en"
        lang_kws = [k.lower() for k in VIOLENCE_KEYWORDS_MULTILINGUAL.get(lang, [])]
        return self.keywords + lang_kws

    @staticmethod
    def _is_non_latin_lang(lang: str) -> bool:
        return lang.split("-")[0].lower() in {
            "ar", "he", "fa", "ur",
            "zh", "ja", "ko",
            "hi", "bn", "pa", "mr",
            "th",
        }

    def scan(self, text: str, detected_lang: str = "en") -> list:
        if not text:
            return []
        all_keywords = self._get_lang_keywords(detected_lang)
        non_latin    = self._is_non_latin_lang(detected_lang)
        matched      = []
        for kw in all_keywords:
            if non_latin:
                if kw in text.lower():
                    matched.append(kw)
            else:
                if re.search(rf"\b{re.escape(kw)}\b", text.lower()):
                    matched.append(kw)
        seen, deduped = set(), []
        for kw in matched:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        return deduped

    def send_alert(self, matched_keywords, transcription,
                   speaker="Unknown", translated_text="") -> bool:
        if not self.sender or not self.password:
            print("  ⚠️  Email not configured.")
            return False
        ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        kwd = ", ".join(
            f"<b style='color:red'>{k}</b>" for k in matched_keywords)
        hl  = transcription
        for kw in matched_keywords:
            hl = re.sub(
                rf"({re.escape(kw)})",
                r"<mark style='background:#ff4444;color:white;"
                r"padding:0 3px;border-radius:3px'>\1</mark>",
                hl, flags=re.IGNORECASE)
        html_body = f"""
        <html><body style="font-family:Arial,sans-serif;max-width:680px;margin:auto">
          <div style="background:#c0392b;color:white;padding:16px 24px;
                      border-radius:8px 8px 0 0">
            <h2 style="margin:0">🚨 VIOLENCE ALERT DETECTED</h2>
            <p style="margin:4px 0 0">{ts}</p>
          </div>
          <div style="border:2px solid #c0392b;border-top:none;padding:20px 24px;
                      border-radius:0 0 8px 8px">
            <table style="width:100%;border-collapse:collapse;margin-bottom:16px">
              <tr><td style="padding:6px 12px;background:#f9f9f9;
                             border:1px solid #ddd;width:160px"><b>Speaker</b></td>
                  <td style="padding:6px 12px;border:1px solid #ddd">{speaker}</td></tr>
              <tr><td style="padding:6px 12px;background:#f9f9f9;
                             border:1px solid #ddd"><b>Keywords found</b></td>
                  <td style="padding:6px 12px;border:1px solid #ddd">{kwd}</td></tr>
            </table>
            <h3>Transcription</h3>
            <p style="background:#fff8e1;border-left:4px solid #f39c12;
                      padding:10px 14px;border-radius:4px">{hl}</p>
            {"<h3>Translation</h3><p style='background:#e8f4fd;border-left:4px solid"
             " #3498db;padding:10px 14px;border-radius:4px'>" + translated_text +
             "</p>" if translated_text else ""}
          </div>
        </body></html>"""
        plain = (
            f"VIOLENCE ALERT\nTime: {ts}\nSpeaker: {speaker}\n"
            f"Keywords: {', '.join(matched_keywords)}\n\n{transcription}"
            + (f"\n\nTranslation:\n{translated_text}" if translated_text else "")
        )
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"🚨 Violence Alert – {', '.join(matched_keywords)}"
        msg["From"]    = self.sender
        msg["To"]      = self.recipient
        msg.attach(MIMEText(plain, "plain"))
        msg.attach(MIMEText(html_body, "html"))
        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as srv:
                srv.ehlo(); srv.starttls()
                srv.login(self.sender, self.password)
                srv.sendmail(self.sender, self.recipient, msg.as_string())
            print(f"  ✅  Alert sent to {self.recipient}")
            return True
        except smtplib.SMTPAuthenticationError:
            print("  ❌  SMTP auth failed.")
        except smtplib.SMTPException as e:
            print(f"  ❌  SMTP error: {e}")
        except Exception as e:
            print(f"  ❌  Email error: {e}")
        return False

    def check_and_alert(self, transcription, speaker="Unknown",
                        translated_text="", detected_lang="en") -> dict:
        matched = self.scan(transcription, detected_lang=detected_lang)
        if not matched:
            print("\n🟢  Violence check: No threats detected.")
            return {"threat_detected": False, "matched_keywords": [],
                    "alert_sent": False}
        if self._is_non_latin_lang(detected_lang):
            kw_display = "  ".join([f"\u202B{k}\u202C" for k in matched])
        else:
            kw_display = str(matched)
        print(f"\n🔴  THREAT DETECTED — keywords: {kw_display}")
        sent = self.send_alert(matched, transcription, speaker, translated_text)
        return {"threat_detected": True, "matched_keywords": matched,
                "alert_sent": sent}


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1 – AUDIO INPUT                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AudioCapture:
    def __init__(self, rate=SAMPLE_RATE, channels=CHANNELS, chunk=CHUNK):
        # INTEGRATION: live mic capture requires pyaudio. File-based analysis
        # never constructs this class, so the module still imports without it.
        if not _HAS_PYAUDIO:
            raise RuntimeError(
                "Microphone capture needs PyAudio. Run: pip install pyaudio "
                "(file-based analysis does not require it).")
        self.rate     = rate
        self.channels = channels
        self.chunk    = chunk
        self._pa      = pyaudio.PyAudio()

    @staticmethod
    def _rms(data: bytes) -> float:
        count  = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        return math.sqrt(sum(s*s for s in shorts) / count) if shorts else 0.0

    def list_devices(self):
        print("\n📻  Available input devices:")
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"   [{i}] {info['name']}")

    def record_fixed(self, seconds, device_index=None, show_live=True) -> str:
        print(f"\n🔴  Recording for {seconds}s …")
        stream = self._pa.open(format=FORMAT, channels=self.channels,
                               rate=self.rate, input=True,
                               input_device_index=device_index,
                               frames_per_buffer=self.chunk)
        frames = []
        for i in range(int(self.rate / self.chunk * seconds)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            if show_live:
                rms = self._rms(data)
                bar = "█" * min(40, int(rms / 50))
                rem = seconds - (i * self.chunk / self.rate)
                print(f"\r   ⏱ {rem:5.1f}s  |{bar:<40}|", end="", flush=True)
        print("\n✅  Recording complete.")
        stream.stop_stream()
        stream.close()
        return self._save_wav(frames)

    def record_until_silence(self, max_seconds=30.0, silence_duration=2.0,
                             device_index=None) -> str:
        print(f"\n🔴  Listening … (stops after {silence_duration}s silence)")
        stream = self._pa.open(format=FORMAT, channels=self.channels,
                               rate=self.rate, input=True,
                               input_device_index=device_index,
                               frames_per_buffer=self.chunk)
        frames = []
        silent = 0
        max_sil = int(self.rate / self.chunk * silence_duration)
        for _ in range(int(self.rate / self.chunk * max_seconds)):
            data = stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            rms  = self._rms(data)
            icon = "🎙" if rms > SILENCE_THRESH else "🔇"
            print(f"\r   {icon}  |{'█'*min(40,int(rms/50)):<40}|",
                  end="", flush=True)
            silent = (silent + 1) if rms < SILENCE_THRESH else 0
            if silent >= max_sil:
                break
        print("\n✅  Audio captured.")
        stream.stop_stream()
        stream.close()
        return self._save_wav(frames)

    def _save_wav(self, frames) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(self._pa.get_sample_size(FORMAT))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return tmp.name

    def close(self):
        self._pa.terminate()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2 – SPEECH-TO-TEXT  (Whisper large-v3)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class SpeechToText:
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE,
                 azure_key: str = "", azure_region: str = "",
                 backend: str = "whisper",
                 force_lang: str = "",
                 fast_decode: bool = False,
                 disambiguate_lang: str = ""):
        self.backend      = backend.lower()
        self.model_size   = model_size
        # INTEGRATION: fast_decode = greedy decode + no word timestamps — much
        # faster on CPU for the web demo (the CLI keeps beam search for accuracy).
        self.fast_decode  = fast_decode
        # Preferred language to also try when auto-detect disagrees (see config).
        self.disambiguate_lang = (disambiguate_lang or "").strip().lower()
        self.azure_key    = azure_key or os.getenv("AZURE_SPEECH_KEY", "")
        self.azure_region = (azure_region or
                             os.getenv("AZURE_SPEECH_REGION", "eastus"))
        self.force_lang   = force_lang or FORCE_LANGUAGE
        self._model       = None
        self._device      = self._best_device()

    @staticmethod
    def _best_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                print("  🖥️   GPU detected — Whisper will run on CUDA")
                return "cuda"
        except ImportError:
            pass
        print("  💻  No GPU — Whisper will run on CPU")
        return "cpu"

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            import whisper
        except ImportError:
            raise RuntimeError(
                "openai-whisper not installed. Run: pip install openai-whisper")
        print(f"  📥  Loading Whisper {self.model_size} …")
        self._model = whisper.load_model(self.model_size, device=self._device)
        print(f"  ✅  Whisper {self.model_size} ready on "
              f"{self._device.upper()}")
        return self._model

    def transcribe(self, wav_path: str) -> dict:
        print(f"\n🎤  Transcribing with Whisper {self.model_size} …")
        if self.backend == "azure":
            return self._azure_transcribe(wav_path)

        import whisper, torch
        model    = self._load_model()
        use_fp16 = (self._device == "cuda")

        # INTEGRATION: decode with soundfile (no ffmpeg for wav/flac/ogg; no
        # librosa, which speechbrain's k2_fsa lazy module breaks). Fall back to
        # whisper's ffmpeg loader only for formats soundfile can't read (mp3…).
        try:
            audio = _load_audio_mono(wav_path, 16000)
        except Exception:
            audio = whisper.load_audio(wav_path)
        audio_pad = whisper.pad_or_trim(audio)
        mel       = whisper.log_mel_spectrogram(
            audio_pad,
            n_mels=model.dims.n_mels,
        ).to(model.device)

        # Language detection (always run even if force_lang is set, for logging)
        with torch.no_grad():
            _, lang_probs = model.detect_language(mel)

        top_lang = max(lang_probs, key=lang_probs.__getitem__)
        top_conf = float(lang_probs[top_lang])

        print(f"  🔍  Whisper language detection: "
              f"{top_lang} ({top_conf:.1%} confidence)")
        top5 = sorted(lang_probs.items(), key=lambda x: -x[1])[:5]
        print("       Top-5: " +
              "  ".join(f"{l}={p:.1%}" for l, p in top5))

        # One Whisper decode pass in a given language.
        def _decode(lang):
            return model.transcribe(
                audio,
                fp16                        = use_fp16,
                # Greedy (beam_size=None) + no word-timestamps is markedly faster
                # on CPU; beam search + alignment is kept for the accuracy CLI.
                beam_size                   = None if self.fast_decode else 5,
                best_of                     = None if self.fast_decode else 5,
                temperature                 = 0,
                condition_on_previous_text  = not self.fast_decode,
                compression_ratio_threshold = 2.4,
                logprob_threshold           = -1.0,
                no_speech_threshold         = 0.6,
                word_timestamps             = not self.fast_decode,
                language                    = lang,
            )

        def _avg_logprob(res):
            segs = res.get("segments", [])
            return (sum(s.get("avg_logprob", 0.0) for s in segs) / len(segs)
                    ) if segs else -9.9

        if self.force_lang:
            # User picked a language — trust it (single decode).
            transcribe_lang = self.force_lang
            print(f"  ⚙️   Force-lang override: using '{self.force_lang}' "
                  f"instead of detected '{top_lang}'")
            result = _decode(transcribe_lang)
        else:
            # Auto: decode the detected language; if it differs from the preferred
            # disambiguation language (e.g. English), ALSO decode that and keep
            # whichever Whisper is more confident about. Accented English is
            # routinely mis-detected as ar/de/…; the correct language gives the
            # higher-confidence decode.
            candidates = [top_lang]
            if self.disambiguate_lang and self.disambiguate_lang != top_lang:
                candidates.append(self.disambiguate_lang)
            result, transcribe_lang, best_conf = None, top_lang, -1e9
            for lang in candidates:
                r = _decode(lang)
                c = _avg_logprob(r)
                if len(candidates) > 1:
                    print(f"  🔎  candidate [{lang}] avg_logprob={c:+.3f}")
                if c > best_conf:
                    result, transcribe_lang, best_conf = r, lang, c
            if len(candidates) > 1 and transcribe_lang != top_lang:
                print(f"  ⚙️   Disambiguation: '{transcribe_lang}' is more confident "
                      f"than detected '{top_lang}' — using it.")

        # Make the reported probs reflect the language we actually transcribed in
        # (forced or disambiguated), so the downstream language label/name matches
        # the text instead of Whisper's original (possibly wrong) detection.
        if transcribe_lang and transcribe_lang != top_lang:
            lang_probs = dict(lang_probs)
            lang_probs[transcribe_lang] = max(float(lang_probs.get(transcribe_lang, 0.0)), 0.95)

        text     = result.get("text", "").strip()
        segments = result.get("segments", [])
        avg_lp   = (sum(s.get("avg_logprob", 0) for s in segments) /
                    len(segments)) if segments else 0.0
        confidence = round(math.exp(avg_lp), 3) if avg_lp else 0.0

        print(f"  ℹ️   avg_logprob={avg_lp:.3f}  "
              f"confidence={confidence:.1%}  segments={len(segments)}")

        return {
            "text":           text,
            "language":       transcribe_lang,
            "language_probs": lang_probs,
            "backend":        f"whisper-{self.model_size}",
            "confidence":     confidence,
            "segments":       len(segments),
        }

    def _azure_transcribe(self, wav_path: str) -> dict:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except ImportError:
            raise RuntimeError("azure-cognitiveservices-speech not installed.")
        if not self.azure_key:
            raise RuntimeError("Azure key missing.")
        cfg  = speechsdk.SpeechConfig(subscription=self.azure_key,
                                       region=self.azure_region)
        acfg = speechsdk.AudioConfig(filename=wav_path)
        rec  = speechsdk.SpeechRecognizer(speech_config=cfg,
                                           audio_config=acfg)
        res  = rec.recognize_once_async().get()
        if res.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {"text": res.text.strip(), "language": None,
                    "language_probs": {}, "backend": "azure",
                    "confidence": None, "segments": 0}
        elif res.reason == speechsdk.ResultReason.NoMatch:
            return {"text": "", "language": None, "language_probs": {},
                    "backend": "azure", "confidence": 0.0, "segments": 0}
        raise RuntimeError(
            f"Azure STT failed: {res.cancellation_details.reason}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4 – TRANSLATION  (MarianMT + Google fallback)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class RealTimeTranslator:
    def __init__(self, target_lang: str = "en"):
        self.target_lang   = target_lang
        self._marian_cache = {}

    @staticmethod
    def _device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception: return "cpu"

    def _load_marian(self, src: str, tgt: str):
        key        = (src, tgt)
        if key in self._marian_cache:
            return self._marian_cache[key]
        model_name = MARIAN_MODEL_MAP.get(key)
        if not model_name: return None
        try:
            from transformers import MarianMTModel, MarianTokenizer
        except ImportError:
            print("  ⚠️  transformers not installed — using Google Translate.")
            return None
        print(f"  📥  Loading MarianMT: {model_name} …")
        try:
            tok   = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(
                model_name).to(self._device())
            model.eval()
            self._marian_cache[key] = (tok, model)
            print(f"  ✅  MarianMT ({src}→{tgt}) ready")
            return self._marian_cache[key]
        except Exception as e:
            print(f"  ⚠️  MarianMT load failed ({e}) — using Google Translate.")
            return None

    def _translate_marian(self, text: str, src: str, tgt: str):
        pair = self._load_marian(src, tgt)
        if pair is None: return None
        tokenizer, model = pair
        try:
            import torch
            inputs = tokenizer(
                [text], return_tensors="pt", padding=True,
                truncation=True, max_length=512).to(self._device())
            with torch.no_grad():
                ids = model.generate(**inputs, num_beams=5,
                                     early_stopping=True, max_length=512)
            return tokenizer.decode(ids[0], skip_special_tokens=True)
        except Exception as e:
            print(f"  ⚠️  MarianMT inference failed: {e}")
            return None

    def _translate_google(self, text: str, src: str) -> str:
        src_arg = ("auto" if src in ("unknown", "auto", None, "")
                   else src)
        try:
            return GoogleTranslator(
                source=src_arg, target=self.target_lang).translate(text)
        except Exception as e:
            print(f"  ⚠️  Google Translate error: {e}")
            return text

    @staticmethod
    def _norm(code: str) -> str:
        m = {"zh-cn": "zh", "zh-tw": "zh", "zh-CN": "zh",
             "jap": "ja", "por": "pt", "spa": "es", "deu": "de",
             "fra": "fr", "ara": "ar", "rus": "ru", "kor": "ko",
             "tur": "tr"}
        return m.get(code, code.split("-")[0].lower())

    def translate(self, text: str, source_lang: str = "auto") -> dict:
        if not text or not text.strip():
            return {"original_text": text, "translated_text": "",
                    "source_lang": source_lang,
                    "target_lang": self.target_lang, "engine": "none"}
        src = self._norm(source_lang)
        tgt = self._norm(self.target_lang)
        if src == tgt:
            return {"original_text": text, "translated_text": text,
                    "source_lang": src, "target_lang": tgt,
                    "engine": "passthrough"}
        result = self._translate_marian(text, src, tgt)
        engine = f"MarianMT ({src}→{tgt})"
        if result is None:
            print(f"  ℹ️   No MarianMT for {src}→{tgt}, "
                  f"using Google Translate")
            result = self._translate_google(text, source_lang)
            engine = "Google Translate (fallback)"
        print(f"  ℹ️   Translation engine: {engine}")
        return {"original_text": text, "translated_text": result,
                "source_lang": src, "target_lang": tgt, "engine": engine}

    @staticmethod
    def list_languages():
        for name, code in sorted(
                GoogleTranslator().get_supported_languages(
                    as_dict=True).items()):
            print(f"   {code:<10} {name}")

    @staticmethod
    def list_marian_pairs():
        print("\n  Supported MarianMT pairs:")
        for (src, tgt), model in sorted(MARIAN_MODEL_MAP.items()):
            print(f"   {src} → {tgt:<6}  {model}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PIPELINE ORCHESTRATOR                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AudioProcessingPipeline:
    def __init__(self, backend="whisper", whisper_model=WHISPER_MODEL_SIZE,
                 azure_key="", azure_region="", target_lang="en",
                 record_mode="fixed", record_seconds=RECORD_SECONDS,
                 translate=False, speaker_id=True, violence_check=True,
                 similarity_threshold=SIMILARITY_THRESHOLD,
                 noise_reduce=False, force_lang="", fast_decode=False,
                 disambiguate_lang=""):
        # INTEGRATION: capture is lazy — only built for live-mic modes, so
        # file-based analysis works on a server with no microphone / pyaudio.
        self._capture       = None
        self.stt            = SpeechToText(model_size=whisper_model,
                                           azure_key=azure_key,
                                           azure_region=azure_region,
                                           backend=backend,
                                           force_lang=force_lang,
                                           fast_decode=fast_decode,
                                           disambiguate_lang=disambiguate_lang)
        self.classifier     = LanguageClassifier()
        self.translator     = RealTimeTranslator(target_lang)
        self.recognizer     = SpeakerRecognizer(threshold=similarity_threshold)
        self.violence       = ViolenceDetector()
        self.record_mode    = record_mode
        self.record_seconds = record_seconds
        self.translate      = translate
        self.speaker_id     = speaker_id
        self.violence_check = violence_check
        self.noise_reduce   = noise_reduce
        self._tmp_dir       = tempfile.mkdtemp(prefix="pipeline_")

    def _get_capture(self):
        if self._capture is None:
            self._capture = AudioCapture()
        return self._capture

    def run(self, wav_file=None) -> dict:
        result = {}
        self.classifier.reset_history()

        # Step 1: Audio
        if wav_file:
            print(f"\n📂  Using: {wav_file}")
            audio_path = wav_file
        elif self.record_mode == "silence":
            audio_path = self._get_capture().record_until_silence()
        else:
            audio_path = self._get_capture().record_fixed(self.record_seconds)
        result["audio_file"] = audio_path

        # Optional noise reduction (FIX 3)
        if self.noise_reduce:
            print("\n🔇  Applying noise reduction …")
            audio_path = apply_noise_reduction(audio_path, self._tmp_dir)

        # Step 0: Speaker (ECAPA-TDNN v2)
        speaker = "N/A"
        if self.speaker_id:
            print("\n🔍  Identifying speaker (ECAPA-TDNN v2) …")
            speaker = self.recognizer.identify(audio_path)
            result["speaker"] = speaker
            icon = "✅" if speaker != "Unknown Person" else "❓"
            print(f"\n👤  Speaker: {icon} {speaker}")
        else:
            result["speaker"] = "N/A"

        # Step 2: STT — Whisper large-v3
        stt_result = self.stt.transcribe(audio_path)
        text       = stt_result["text"]
        result["transcription"] = stt_result

        whisper_lang = stt_result.get("language", "")
        if whisper_lang in RTL_LANGS and text:
            display_text = f"\u202B{text}\u202C"
        else:
            display_text = text or "(empty)"
        print(f"\n📝  Transcription: {display_text}")

        # Step 3: Language classifier (v3 – Whisper-first)
        lang_result = self.classifier.classify(text, whisper_result=stt_result)
        result["language"] = lang_result

        method  = lang_result.get("method", "?")
        engines = ", ".join(lang_result.get("engines_used", []))
        w_lang  = lang_result.get("whisper_lang", "")
        w_conf  = lang_result.get("whisper_conf", 0.0)

        print(f"\n🌐  Language: {lang_result['language_name']} "
              f"({lang_result['language_code']})  "
              f"conf={lang_result['confidence']:.0%}  "
              f"method=[{method}]  engines=[{engines}]")
        if w_lang:
            print(f"     Whisper said: {w_lang} ({w_conf:.0%})")

        # Step 4: Translation
        translated_text = ""
        if self.translate and text:
            src         = lang_result["language_code"]
            translation = self.translator.translate(text, source_lang=src)
            result["translation"] = translation
            translated_text = translation["translated_text"]
            tgt_name = LANGUAGE_NAMES.get(self.translator.target_lang,
                                          self.translator.target_lang.upper())
            print(f"\n🔤  Translation ({tgt_name}) "
                  f"[{translation['engine']}]:\n    {translated_text}")
        else:
            result["translation"] = None

        # Step 5: Violence detection
        detected_lang = lang_result.get("language_code", "en")
        if self.violence_check and text:
            result["violence"] = self.violence.check_and_alert(
                text, speaker, translated_text,
                detected_lang=detected_lang)
        else:
            result["violence"] = {"threat_detected": False,
                                  "matched_keywords": [], "alert_sent": False}

        if not wav_file:
            Path(result["audio_file"]).unlink(missing_ok=True)

        print("\n" + "═" * 60)
        threat = result["violence"]["threat_detected"]
        print(f"Pipeline complete.  Speaker: {speaker}  |  "
              f"Violence: "
              f"{'🚨 THREAT DETECTED' if threat else '✅  No threats'}")
        return result

    def close(self):
        if self._capture is not None:
            self._capture.close()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CLI ENTRY POINT                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args():
    p = argparse.ArgumentParser(
        description="Audio Pipeline v3 – Whisper-first language detection fix "
                    "+ noise reduction + force-lang + MarianMT + ECAPA-TDNN v2")
    p.add_argument("--backend", default="whisper",
                   choices=["whisper", "azure"])
    p.add_argument("--whisper-model", default=WHISPER_MODEL_SIZE,
                   choices=["tiny", "base", "small", "medium",
                            "large", "large-v2", "large-v3"])
    p.add_argument("--azure-key",    default="")
    p.add_argument("--azure-region", default="")
    p.add_argument("--seconds",  type=float, default=RECORD_SECONDS)
    p.add_argument("--mode",     default="fixed",
                   choices=["fixed", "silence"])
    p.add_argument("--translate", metavar="LANG",
                   help="Translate to this language code (e.g. en, fr, ar)")
    p.add_argument("--file",     metavar="WAV")
    p.add_argument("--threshold", type=float, default=SIMILARITY_THRESHOLD)
    p.add_argument("--no-augment",        action="store_true")
    p.add_argument("--no-speaker-id",     action="store_true")
    p.add_argument("--no-violence-check", action="store_true")
    p.add_argument("--enroll", nargs="+", metavar=("NAME", "AUDIO"))
    p.add_argument("--list-speakers",    action="store_true")
    p.add_argument("--speaker-info",     action="store_true")
    p.add_argument("--delete-speaker",   metavar="NAME")
    p.add_argument("--list-languages",   action="store_true")
    p.add_argument("--list-marian-pairs", action="store_true")
    p.add_argument("--list-devices",     action="store_true")
    # ── NEW FLAGS (v3) ──────────────────────────────────────────────────────
    p.add_argument("--noise-reduce", action="store_true",
                   help="Apply spectral noise reduction before transcription "
                        "(recommended without headphones). "
                        "Requires: pip install noisereduce")
    p.add_argument("--force-lang", metavar="LANG", default="",
                   help="Force Whisper to transcribe in this language code "
                        "(e.g. --force-lang en). Bypasses auto-detection. "
                        "Useful when you KNOW the input language.")
    return p.parse_args()


def main():
    args = parse_args()
    global AUGMENT_ENROLMENT, FORCE_LANGUAGE
    if args.no_augment:
        AUGMENT_ENROLMENT = False
    if args.force_lang:
        FORCE_LANGUAGE = args.force_lang.lower()

    if args.list_languages:
        RealTimeTranslator.list_languages(); return
    if args.list_marian_pairs:
        RealTimeTranslator.list_marian_pairs(); return
    if args.list_devices:
        AudioCapture().list_devices(); return

    recognizer = SpeakerRecognizer(threshold=args.threshold)
    if args.list_speakers:
        sp = recognizer.list_speakers()
        print("\n👥  Enrolled:", ", ".join(sp) if sp else "(none)")
        return
    if args.speaker_info:
        recognizer.speaker_info(); return
    if args.delete_speaker:
        ok = recognizer.delete_speaker(args.delete_speaker)
        print(f"{'✔' if ok else '⚠'} '{args.delete_speaker}' "
              f"{'removed' if ok else 'not found'}.")
        return
    if args.enroll:
        name, files = args.enroll[0], args.enroll[1:]
        if not files:
            sys.exit("❌  Provide audio files after the name.")
        recognizer.enroll(name, files)
        return

    pipeline = AudioProcessingPipeline(
        backend              = args.backend,
        whisper_model        = args.whisper_model,
        azure_key            = args.azure_key,
        azure_region         = args.azure_region,
        target_lang          = args.translate or "en",
        record_mode          = args.mode,
        record_seconds       = args.seconds,
        translate            = bool(args.translate),
        speaker_id           = not args.no_speaker_id,
        violence_check       = not args.no_violence_check,
        similarity_threshold = args.threshold,
        noise_reduce         = args.noise_reduce,
        force_lang           = args.force_lang,
    )
    try:
        pipeline.run(wav_file=args.file)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted.")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()