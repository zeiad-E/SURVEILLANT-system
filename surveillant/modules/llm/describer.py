"""
modules/llm/describer.py
-------------------------
LLM-based body description for SURVEILLANT (Part 10 / Phase 4).

This module never raises into the embedding worker. Backend failures
return ``None`` so the DescriptionWorker can decide whether to retry.

Two backends behind one abstract interface, swappable via the
``DESCRIPTION_BACKEND`` config flag:

* ``QwenVLOllamaDescriber`` — local Ollama, CPU, default.
* ``MarlinRemoteDescriber`` — POSTs snapshots to a remote GPU host
  running ``modules/llm/marlin_server/serve.py``.

A third class (``QueryParser``) turns free-text operator queries into
the same JSON schema using a small text-only Ollama model — orders of
magnitude faster than the VLM and runs entirely on CPU.
"""

from __future__ import annotations

import base64
import json
import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import (
    OLLAMA_HOST,
    OLLAMA_VLM_MODEL,
    OLLAMA_VLM_TIMEOUT_SEC,
    OLLAMA_VLM_NUM_CTX,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_THINK,
    OLLAMA_QUERY_MODEL,
    MARLIN_HOST,
    MARLIN_TIMEOUT_SEC,
    DESCRIPTION_BACKEND,
    DESCRIBE_QUALITY_MODE,
)


# ---------------------------------------------------------------------------
# Schema (kept in code so the prompts and SQL filters stay in sync)
# ---------------------------------------------------------------------------

# Structured keys the describer MAY include WHEN CONFIDENT. None are required:
# the model returns `long_description` plus any of these it can clearly see and
# OMITS the rest (an honest, flexible schema — no hallucinated "unknown" fillers).
RECOMMENDED_DESCRIPTION_KEYS = (
    "gender", "age_range", "body_build",
    "hair_color", "hair_length", "glasses",
    "headwear", "headwear_color",
    "clothing_top", "clothing_top_color",
    "clothing_bottom", "clothing_bottom_color",
    "accessories",
)

# Backward-compatible alias for older imports. `long_description` is the one
# key that should always be present for a usable result.
REQUIRED_DESCRIPTION_FIELDS = RECOMMENDED_DESCRIPTION_KEYS + ("long_description",)

COLOR_PALETTE = (
    "red", "blue", "green", "yellow", "black", "white", "gray",
    "brown", "orange", "purple", "pink", "multi", "unknown",
)

# Values that mean "I am not sure / not visible" — dropped from the output so a
# key is present only when the model is actually confident about it.
_UNCERTAIN_VALUES = {
    "", "unknown", "none", "n/a", "na", "null", "not visible", "not sure",
    "unsure", "unclear", "uncertain", "not applicable", "not specified",
}

# System prompt is identical across backends so model output stays comparable.
SYSTEM_PROMPT_DESCRIBE = (
    "You are a careful surveillance vision assistant analysing a single image "
    "of one person. Describe ONLY what is clearly and unambiguously visible. "
    "NEVER guess, assume, or invent anything. If a body part (e.g. the legs or "
    "feet), a garment, or any attribute is not visible in the frame, or you are "
    "not certain, DO NOT mention it and DO NOT include its key. Omitting an "
    "attribute is correct and expected; stating something that might be wrong "
    "is a serious error. Output ONLY a single JSON object — no markdown, no prose."
)

USER_PROMPT_DESCRIBE = """\
Output ONE JSON object describing the person.

The FIRST key MUST be "long_description": a RICH, DETAILED paragraph of AT LEAST
4 to 6 full sentences (more is better) describing EVERYTHING you can actually
see. Do NOT stop at one or two sentences - a short description is a failure here.
Work top to bottom and cover every part that is visible: head and hair (colour,
length, style), the face (glasses, beard, expression if clear), any headwear,
the upper-body garment (its exact type, colour/shade, fit, sleeves, collar, and
any visible pattern, logo or text), the lower-body garment and footwear ONLY if
the legs/feet are in frame, any carried or worn accessories (bag, backpack,
watch, phone), and the person's build, posture and which way they are facing.

Make the paragraph LONG by describing the SAME visible things in more precise
detail - the exact shade of a colour, the cut and fit of a garment, how layers
sit on each other, the pose and orientation, where the person is in the frame -
NOT by guessing hidden details. This is the strict, non-negotiable rule:
describe ONLY what is clearly and unambiguously visible. If a body part or item
is out of frame, occluded, blurry, or you are not 100% certain, simply DO NOT
mention it (you may briefly note that lower-body/feet are out of frame). NEVER
invent, assume, or write "maybe", "possibly", "appears to", or "likely". Being
long is required, but being WRONG is far worse than being short - achieve length
only by extracting more genuine detail from what is actually shown.

Then ALSO add these structured keys. Be THOROUGH: for EVERY feature you mention
in long_description that you are sure about, add its key. (e.g. if you wrote
"light blue Nike t-shirt", you MUST add clothing_top="t-shirt" and
clothing_top_color="blue".) Only leave a key OUT when that feature is NOT
visible or you are not sure - never write "unknown" and never guess:
  "gender", "age_range", "body_build",
  "hair_color", "hair_length", "glasses" ("yes" or "no"),
  "headwear", "headwear_color",
  "clothing_top", "clothing_top_color",
  "clothing_bottom", "clothing_bottom_color"   (ONLY if the lower body is visible)
  "accessories": [ list of visible items, e.g. "backpack", "watch" ]

EXAMPLE — a person seen from the chest up (legs NOT visible):
{
  "long_description": "A young adult man of average build, seen from roughly the chest up and facing slightly to his left. He has short, neatly cut black hair and a light, closely-trimmed beard along his jawline. He is wearing a dark green hooded jacket, zipped about halfway up, with the hood lying flat against the back of his shoulders. Underneath the jacket the collar of a plain white crew-neck shirt is clearly visible. His hands, lower body and feet are out of frame, so his trousers, footwear and anything he might be carrying cannot be seen.",
  "gender": "male",
  "age_range": "young_adult",
  "body_build": "average",
  "hair_color": "black",
  "hair_length": "short",
  "glasses": "no",
  "clothing_top": "jacket",
  "clothing_top_color": "green"
}
Notice EVERY visible feature got a structured key, while clothing_bottom /
clothing_bottom_color are OMITTED because the legs cannot be seen. Do exactly the
same: extract a key for everything you can see, omit only what you cannot.

Now output the JSON for the person in THIS image:
"""

# (#2) Systematic head-to-toe scan — appended to the describe prompt in quality
# mode. Near-zero extra cost; nudges the model to be thorough and not skip parts.
SCAN_ADDENDUM = """

Before writing, scan the person systematically from top to bottom:
  head & hair -> face (glasses? beard?) -> headwear -> upper-body clothing + colour
  -> lower-body clothing + colour (ONLY if the legs are visible) -> hands &
  accessories (watch, bag, phone). Note each part you can clearly see, omit the
  parts you cannot, and be precise about colours and any visible text or logos."""

# (#1) Self-verification pass — a second call that re-checks the draft against
# the same image, removing invented details and adding missed ones.
SYSTEM_PROMPT_VERIFY = (
    "You are double-checking a draft description of the person in THIS image. "
    "Look at the image carefully and correct the draft: REMOVE any detail or key "
    "that is not actually visible or that you are not certain about, and ADD any "
    "clearly-visible feature that is missing. Fix any wrong colour or garment. "
    "Never invent anything. Keep the same JSON shape (a long_description plus "
    "confident structured keys). Output ONLY the corrected JSON object."
)

USER_PROMPT_VERIFY = """\
Draft description of the person in the image:
{draft}

Look at the image again and return the CORRECTED JSON. Remove anything not
clearly visible, add anything clearly visible that is missing, and fix any wrong
colour or item. Keep only what you are sure about. Keep the long_description
RICH and DETAILED (4-6+ sentences) - do NOT shorten it; only strip out details
that are not actually visible and expand any visible feature you can describe
more precisely.
"""


SYSTEM_PROMPT_PARSE_QUERY = (
    "You parse a security operator's spoken description of a person into a "
    "search filter for a database. Output ONLY a single JSON object — no "
    "commentary, no markdown fences. Omit any field the operator did not "
    "mention. Use 'no' for negations like 'without glasses' or 'no beard'."
)

USER_PROMPT_PARSE_QUERY = """\
Parse this query into the same JSON schema (omit unspecified fields):
{schema}

Query: {query}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)\s*```", re.DOTALL)


def _clean_json(raw: str) -> Optional[Dict[str, Any]]:
    """
    Robust JSON extraction from a possibly-noisy LLM output.

    Handles ```json fences, leading/trailing prose, and stray characters.
    Returns the parsed dict or None if the response is unrecoverable. The
    caller is expected to log the raw text on None so we can audit model
    failures later.
    """
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        return None

    # 1. Try the raw text directly.
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        pass

    # 2. Strip ```json ... ``` fences if present.
    m = _FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except (ValueError, TypeError):
            pass

    # 3. Last resort: locate the first '{' and the last '}' and try that slice.
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            return json.loads(snippet)
        except (ValueError, TypeError):
            return None
    return None


def _clean_flexible(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep ONLY the keys the model actually returned, dropping any whose value is
    empty or signals uncertainty (see ``_UNCERTAIN_VALUES``). No field is forced
    or filled with a placeholder — this preserves the model's honest, flexible
    output (it omits what it cannot see), so we never store a hallucinated
    attribute like "brown pants" for a person whose legs aren't in frame.

    ``long_description`` is preserved with its original casing (it is the text
    that gets embedded for semantic search). If the model gave structured keys
    but no long_description, one is synthesised from them so the row is still
    searchable.
    """
    out: Dict[str, Any] = {}
    long_desc: Optional[str] = None

    for raw_key, v in (parsed or {}).items():
        key = str(raw_key).strip().lower()
        if not key or v is None:
            continue

        # Free-text description — accept several aliases, keep original case.
        if key in ("long_description", "description", "summary", "details"):
            if isinstance(v, str) and v.strip():
                long_desc = v.strip()
            continue

        if key == "accessories":
            if isinstance(v, list):
                items = [str(x).strip().lower() for x in v
                         if x and str(x).strip().lower() not in _UNCERTAIN_VALUES]
            elif isinstance(v, str) and v.strip().lower() not in _UNCERTAIN_VALUES:
                items = [v.strip().lower()]
            else:
                items = []
            if items:
                out["accessories"] = items
            continue

        # Scalar attribute: keep only if confident (non-uncertain).
        sval = str(v).strip().lower()
        if sval in _UNCERTAIN_VALUES:
            continue
        out[key] = sval

    if long_desc:
        out["long_description"] = long_desc
    elif out:
        # Synthesise a description from the structured keys we kept.
        bits = []
        for k, val in out.items():
            if k == "accessories":
                bits.append("accessories: " + ", ".join(val))
            else:
                bits.append(f"{k.replace('_', ' ')}: {val}")
        out["long_description"] = "; ".join(bits)

    return out


def _read_image_b64(path: str) -> Optional[str]:
    """Read an image file and return base64-encoded bytes (or None on failure)."""
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except OSError as exc:
        print(f"[DESCRIBE] read fail {path}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class Describer(ABC):
    """Abstract description backend. Implementations must never raise upward."""

    backend_name: str = "abstract"
    model_id:     str = ""

    @abstractmethod
    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        """
        Return a canonical-schema dict (see REQUIRED_DESCRIPTION_FIELDS),
        or None on any failure. The caller logs and decides whether to retry.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Qwen2.5-VL via local Ollama (default CPU backend)
# ---------------------------------------------------------------------------

class QwenVLOllamaDescriber(Describer):
    """
    Calls a local Ollama server running a vision LLM (default qwen2.5vl:3b).
    Talks via Ollama's HTTP /api/chat with image attachments.

    Run on the local machine; CPU-only is fine (~20–40 s per image for
    qwen2.5vl:3b, a non-reasoning VLM). Pull once with: ``ollama pull
    qwen2.5vl:3b``. (qwen3-vl was tried but its undisableable reasoning made
    it 400 s+/image on CPU — see DECISION_LOG LL-T.)
    """

    backend_name = "qwen-vl"

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_VLM_MODEL,
        timeout_sec: int = OLLAMA_VLM_TIMEOUT_SEC,
    ) -> None:
        self.host = host.rstrip("/")
        self.model_id = model
        self.timeout_sec = timeout_sec

    def _post_chat(
        self, messages: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """
        One Ollama /api/chat round-trip. Returns the parsed JSON dict (raw,
        un-cleaned) or None on any HTTP / parse failure. Never raises.
        """
        try:
            import requests  # local import so the dependency stays optional
        except ImportError:
            print("[DESCRIBE] 'requests' not installed; cannot call Ollama.")
            return None

        payload = {
            "model":      self.model_id,
            "stream":     False,
            "format":     "json",  # Ollama's structured-output mode
            "keep_alive": OLLAMA_KEEP_ALIVE,  # keep model warm across the batch
            "messages":   messages,
            "options": {
                "temperature": 0.1,
                "num_ctx":     OLLAMA_VLM_NUM_CTX,
                # Headroom for the longer (4-6 sentence) paragraph + structured
                # keys, so the JSON is never cut off mid-object (which would fail
                # to parse). The model still stops early when it is done.
                "num_predict": 768,
            },
        }
        # Only send `think` for reasoning models that honour it. Non-reasoning
        # VLMs (qwen2.5vl) reject/ignore the field, so OLLAMA_THINK=None omits it.
        if OLLAMA_THINK is not None:
            payload["think"] = OLLAMA_THINK

        try:
            r = requests.post(
                f"{self.host}/api/chat", json=payload, timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[DESCRIBE] qwen-vl HTTP failure: {exc}")
            return None

        raw = (data.get("message") or {}).get("content") or ""
        parsed = _clean_json(raw)
        if parsed is None:
            preview = raw[:300].replace("\n", " ")
            print(f"[DESCRIBE] qwen-vl JSON parse failed; raw='{preview}...'")
        return parsed

    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        if not snapshot_paths:
            return None
        # 4A: single-image input. Multi-image consensus is deferred to 4C.
        img_b64 = _read_image_b64(snapshot_paths[0])
        if img_b64 is None:
            return None

        # Pass 1 — describe. In quality mode (#2) append the head-to-toe scan.
        user_prompt = USER_PROMPT_DESCRIBE + (SCAN_ADDENDUM if DESCRIBE_QUALITY_MODE else "")
        parsed = self._post_chat([
            {"role": "system", "content": SYSTEM_PROMPT_DESCRIBE},
            {"role": "user", "content": user_prompt, "images": [img_b64]},
        ])
        if parsed is None:
            return None

        # Pass 2 (#1) — quality mode only: re-check the draft against the image,
        # removing invented details and adding missed ones. Falls back to the
        # draft if the verify call fails to parse.
        if DESCRIBE_QUALITY_MODE:
            draft = json.dumps(_clean_flexible(parsed), ensure_ascii=False)
            verified = self._post_chat([
                {"role": "system", "content": SYSTEM_PROMPT_VERIFY},
                {"role": "user",
                 "content": USER_PROMPT_VERIFY.format(draft=draft),
                 "images": [img_b64]},
            ])
            if verified:
                parsed = verified

        return _clean_flexible(parsed)


# ---------------------------------------------------------------------------
# Marlin-2B via remote GPU host
# ---------------------------------------------------------------------------

class MarlinRemoteDescriber(Describer):
    """
    POSTs snapshots to a FastAPI host running modules/llm/marlin_server/serve.py.

    Marlin-2B is a video-VLM that requires a GPU; SURVEILLANT runs on CPU.
    The remote host runs the model and exposes a single endpoint, so the
    CPU-only client only needs `requests`. If MARLIN_HOST is empty or the
    host is unreachable, describe() returns None.
    """

    backend_name = "marlin"

    def __init__(
        self,
        host: str = MARLIN_HOST,
        model_id: str = "NemoStation/Marlin-2B",
        timeout_sec: int = MARLIN_TIMEOUT_SEC,
    ) -> None:
        self.host = (host or "").rstrip("/")
        self.model_id = model_id
        self.timeout_sec = timeout_sec

    def describe(self, snapshot_paths: List[str]) -> Optional[Dict[str, Any]]:
        if not self.host:
            print("[DESCRIBE] marlin backend selected but MARLIN_HOST is empty.")
            return None
        if not snapshot_paths:
            return None

        try:
            import requests
        except ImportError:
            print("[DESCRIBE] 'requests' not installed; cannot call Marlin.")
            return None

        images_b64 = [b for b in (_read_image_b64(p) for p in snapshot_paths) if b]
        if not images_b64:
            return None

        payload = {
            "images_b64":      images_b64,
            "system_prompt":   SYSTEM_PROMPT_DESCRIBE,
            "user_prompt":     USER_PROMPT_DESCRIBE,
            "expect_schema":   list(REQUIRED_DESCRIPTION_FIELDS),
            "color_palette":   list(COLOR_PALETTE),
        }

        try:
            r = requests.post(
                f"{self.host}/describe",
                json=payload,
                timeout=self.timeout_sec,
            )
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[DESCRIBE] marlin HTTP failure: {exc}")
            return None

        # The Marlin server returns either a structured dict directly OR a
        # 'raw' field that we still need to clean (depending on model output).
        if "attributes" in data and isinstance(data["attributes"], dict):
            return _clean_flexible(data["attributes"])

        parsed = _clean_json(data.get("raw", ""))
        if parsed is None:
            return None
        return _clean_flexible(parsed)


# ---------------------------------------------------------------------------
# Query parser (text-only Ollama)
# ---------------------------------------------------------------------------

class QueryParser:
    """
    Parses a free-text operator query into a partial description filter.

    Reuses the describer model (qwen3-vl:2b) — no second model to download.
    To stay fast and simple, parsing tries the instant rule-based parser
    FIRST and only calls the LLM when the rules extract nothing (the LLM is
    slow on CPU). Missing fields = no filter on that field. Returns the dict
    directly so TextSearchEngine can pass it to search_persons_by_attributes.
    """

    def __init__(
        self,
        host: str = OLLAMA_HOST,
        model: str = OLLAMA_QUERY_MODEL,
        timeout_sec: int = OLLAMA_VLM_TIMEOUT_SEC,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout_sec = timeout_sec

    def parse(self, query: str) -> Dict[str, Any]:
        # Fast path: the rule-based parser is instant (no model, no network)
        # and covers the common operator vocabulary — gender, build,
        # garment+color, headwear, glasses/beard, accessories, negation.
        # Only fall back to the LLM when the rules find nothing.
        rule = self._rule_based_fallback(query)
        if rule:
            return rule
        llm = self._llm_parse(query)
        return llm if llm else rule

    def _llm_parse(self, query: str) -> Optional[Dict[str, Any]]:
        """Call the LLM to parse the query. Returns None on any failure."""
        try:
            import requests
        except ImportError:
            print("[QUERY] 'requests' not installed; cannot call Ollama.")
            return None

        schema = "{\n" + ",\n".join(f'  "{f}": ...' for f in REQUIRED_DESCRIPTION_FIELDS) + "\n}"
        payload = {
            "model":      self.model,
            "stream":     False,
            "format":     "json",
            "keep_alive": OLLAMA_KEEP_ALIVE,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_PARSE_QUERY},
                {"role": "user",   "content": USER_PROMPT_PARSE_QUERY.format(
                    schema=schema, query=query,
                )},
            ],
            "options": {"temperature": 0.0, "num_ctx": OLLAMA_VLM_NUM_CTX},
        }
        if OLLAMA_THINK is not None:
            payload["think"] = OLLAMA_THINK

        try:
            r = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout_sec)
            r.raise_for_status()
            data = r.json()
        except Exception as exc:
            print(f"[QUERY] LLM failure: {exc}")
            return None

        parsed = _clean_json((data.get("message") or {}).get("content") or "")
        if not parsed:
            print("[QUERY] LLM returned unparseable JSON.")
            return None

        # Keep missing fields missing so the SQL builder ignores them.
        cleaned: Dict[str, Any] = {}
        for f in REQUIRED_DESCRIPTION_FIELDS:
            if f not in parsed:
                continue
            v = parsed[f]
            if f == "accessories":
                if isinstance(v, list):
                    cleaned[f] = [str(x).strip().lower() for x in v if x]
                elif isinstance(v, str) and v.strip():
                    cleaned[f] = [v.strip().lower()]
            elif isinstance(v, str) and v.strip():
                cleaned[f] = v.strip().lower()
            elif v is not None:
                cleaned[f] = v
        return cleaned

    # --- Rule-based fallback ----------------------------------------------

    _KEYWORDS = {
        # gender
        "male":   ("man", "guy", "male", "boy", "gentleman"),
        "female": ("woman", "lady", "female", "girl"),
    }
    _COLORS = COLOR_PALETTE
    _GARMENTS_TOP = ("t-shirt", "tshirt", "shirt", "jacket", "hoodie", "coat", "sweater", "blouse", "vest")
    _GARMENTS_BOTTOM = ("jeans", "pants", "shorts", "skirt", "trousers", "dress")
    _HEADWEAR = ("hat", "cap", "beanie", "hood", "scarf", "helmet")
    _ACCESSORIES = ("backpack", "handbag", "bag", "umbrella", "watch", "phone")

    def _rule_based_fallback(self, query: str) -> Dict[str, Any]:
        q = query.lower()
        out: Dict[str, Any] = {}

        # Gender — check female FIRST and use word boundaries so "woman"
        # doesn't trigger the "man" substring match. (This is the kind of
        # bug a real LLM parser doesn't have.)
        if any(re.search(rf"\b{kw}\b", q) for kw in self._KEYWORDS["female"]):
            out["gender"] = "female"
        elif any(re.search(rf"\b{kw}\b", q) for kw in self._KEYWORDS["male"]):
            out["gender"] = "male"

        # Build (word-boundary so "chubby" doesn't match inside other words)
        if re.search(r"\b(fat|heavy|overweight|chubby|stocky)\b", q):
            out["body_build"] = "heavy"
        elif re.search(r"\b(slim|thin|skinny|lean)\b", q):
            out["body_build"] = "slim"

        # Negation for binary fields
        if re.search(r"\b(no|without)\s+glasses\b", q): out["glasses"] = "no"
        elif "glasses" in q:                            out["glasses"] = "yes"
        if re.search(r"\b(no|without)\s+beard\b", q):   out["beard"] = "no"
        elif "beard" in q:                              out["beard"] = "yes"

        # Top + top color: "<color> <top-garment>"
        for color in self._COLORS:
            for top in self._GARMENTS_TOP:
                if re.search(rf"\b{color}\b.{{0,15}}\b{top}\b", q):
                    out["clothing_top"]       = top
                    out["clothing_top_color"] = color
                    break
        # Bottom
        for color in self._COLORS:
            for bot in self._GARMENTS_BOTTOM:
                if re.search(rf"\b{color}\b.{{0,15}}\b{bot}\b", q):
                    out["clothing_bottom"]       = bot
                    out["clothing_bottom_color"] = color
                    break
        # Headwear
        for color in self._COLORS:
            for hw in self._HEADWEAR:
                if re.search(rf"\b{color}\b.{{0,15}}\b{hw}\b", q):
                    out["headwear"]       = hw
                    out["headwear_color"] = color
                    break

        # Accessories
        accs = [a for a in self._ACCESSORIES if re.search(rf"\b{a}\b", q)]
        if accs:
            out["accessories"] = accs

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_describer(backend: Optional[str] = None) -> Describer:
    """
    Construct the configured describer backend. Falls back to qwen-vl with
    a warning if Marlin is requested but MARLIN_HOST is empty.
    """
    name = (backend or DESCRIPTION_BACKEND or "qwen-vl").lower()
    if name == "marlin":
        if MARLIN_HOST:
            return MarlinRemoteDescriber()
        print(
            "[DESCRIBE] DESCRIPTION_BACKEND='marlin' but MARLIN_HOST is empty — "
            "falling back to local qwen-vl. Set MARLIN_HOST in settings.py to "
            "use the GPU host."
        )
    return QwenVLOllamaDescriber()
