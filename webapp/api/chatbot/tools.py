"""
webapp/api/chatbot/tools.py
---------------------------
The "tools" the assistant can call. Thin wrappers over the engine Database +
existing services so the chatbot reuses the EXACT same search/shaping the rest
of the webapp uses (no parallel logic to drift).

Read tools:  extract_filters, search, get_person, stats, alerts
Write tools live in webapp/api/corrections_service.py and are invoked (after a
confirmation + role check) by the chat router.

Two attribute sources, mirrored from routers/search.py:
  - Face/persons-row attributes (gender, age_range, ethnicity, glasses, status,
    camera, name) — present today from the Part-11 face layer. Filtered in
    memory over get_all_persons().
  - Appearance/description attributes (clothing_*, hair_color, headwear,
    body_build, accessories) — from the LLM descriptions; need --describe-all.
    Matched via db.search_persons_by_attributes().
"""

import json
import re
from typing import Any, Dict, List, Optional

from .. import config, engine, services

# ── Age buckets (must match modules/face/face_analyzer.py) ──────────────────
AGE_BUCKETS = [
    ("0-3", 0, 3), ("4-6", 4, 6), ("7-10", 7, 10), ("11-19", 11, 19),
    ("20-30", 20, 30), ("30-45", 30, 45), ("45-55", 45, 55),
    ("55-70", 55, 70), ("70+", 70, 200),
]
_CHILD = ["0-3", "4-6", "7-10"]
_TEEN = ["11-19"]
_YOUNG = ["11-19", "20-30"]
_ELDERLY = ["55-70", "70+"]

_ETHNICITY_WORDS = {
    "asian": "Asian", "black": "Black", "african": "Black", "indian": "Indian",
    "white": "White", "caucasian": "White",
    "latino": "Latino_Hispanic", "hispanic": "Latino_Hispanic",
    "middle eastern": "Middle_Eastern", "middle-eastern": "Middle_Eastern",
    "arab": "Middle_Eastern",
}
_STATUS_WORDS = ("unverified", "confirmed", "multi_view", "flagged", "ghost")
_APPEARANCE_KEYS = (
    "clothing_top", "clothing_bottom", "clothing_top_color", "clothing_bottom_color",
    "hair_color", "headwear", "headwear_color", "body_build", "accessories",
)

# ── Supplemental clothing-colour extraction ──────────────────────────────────
# Used inside extract_filters() when QueryParser misses colour from chat phrasing
# (e.g. "black T shirt", "red hoodie").  Colours are reliable; clothing type is
# only extracted when no colour was matched (brand + type queries like "Nike T shirt").
_COLOUR_TOP_RE = re.compile(
    r"\b(black|white|red|blue|green|yellow|orange|purple|pink|gray|grey|brown|navy|"
    r"beige|dark\s+(?:blue|green|gray|grey|red|brown)|light\s+(?:blue|green|gray|grey))"
    r"\s+"
    r"(t[\s-]?shirt|shirt|hoodie|sweatshirt|sweater|jacket|coat|blouse|polo|jersey|vest|top)\b",
    re.I,
)
_COLOUR_BOT_RE = re.compile(
    r"\b(black|white|red|blue|green|yellow|orange|purple|pink|gray|grey|brown|navy|"
    r"beige|dark\s+(?:blue|green|gray|grey)|light\s+(?:blue|green|gray|grey))"
    r"\s+"
    r"(pants|jeans|trousers|shorts|skirt|leggings)\b",
    re.I,
)
_TOP_ONLY_RE = re.compile(
    r"\b(t[\s-]?shirt|hoodie|sweatshirt|sweater|polo|jersey)\b", re.I
)

# ── Query cleaning for semantic embedding ─────────────────────────────────────
_INTENT_STRIP_RE = re.compile(
    r"^\s*(?:search(?:\s+for(?:\s+me)?)?|find|look(?:ing)?\s+for|show\s+me|"
    r"locate|get\s+me|spot|identify)\s+(?:me\s+)?(?:about\s+)?",
    re.I,
)
_QUANTITY_STRIP_RE = re.compile(
    r"\b(?:only|just)\s+"
    r"(?:one|1|two|2|three|3|four|4|five|5|six|6|seven|7|eight|8|nine|9|ten|10)\s+"
    r"(?:person|people|man|woman|persons)?\s*",
    re.I,
)


def bucket_for_age(n: int) -> Optional[str]:
    for label, lo, hi in AGE_BUCKETS:
        if lo <= n <= hi:
            return label
    return None


def buckets_for_age_range(lo: int, hi: int) -> List[str]:
    """All bucket labels that OVERLAP the inclusive age span [lo, hi].

    The DB stores coarse buckets, not exact ages, so a query like "40 to 55"
    must include every bucket that could contain a person in that span (e.g. a
    person bucketed "30-45" might be 44 — inside the range). Overlap is the
    correct semantics here, not strict containment.
    """
    lo, hi = sorted((lo, hi))
    return [label for label, blo, bhi in AGE_BUCKETS if blo <= hi and bhi >= lo]


def _clean_query(text: str) -> str:
    """Strip intent verbs and quantity words before semantic embedding so the
    MiniLM vector focuses on the descriptive content, not navigation words.

    "Search for me about a man wearing Nike T shirt" → "a man wearing Nike T shirt"
    "Find Only one person wearing black T shirt"     → "wearing black T shirt"
    """
    q = _INTENT_STRIP_RE.sub("", text.strip())
    q = _QUANTITY_STRIP_RE.sub("", q)
    return q.strip() or text


# ── Query -> filters ────────────────────────────────────────────────────────
def _rule_parse(text: str) -> Dict[str, Any]:
    """Use ONLY the engine QueryParser's instant rule path (no LLM call here —
    Tier-1 must stay fast/offline)."""
    try:
        from surveillant.modules.llm.describer import QueryParser  # noqa: PLC0415

        return QueryParser()._rule_based_fallback(text) or {}
    except Exception:  # noqa: BLE001
        return {}


def extract_filters(text: str) -> Dict[str, Any]:
    """Turn a free-text message into {face: {...}, appearance: {...}, text}.

    Face filters drive the persons-row match; appearance filters drive the
    description-attribute match. ``text`` is kept for the semantic fallback.
    """
    low = text.lower()
    qp = _rule_parse(text)

    face: Dict[str, Any] = {}
    appearance: Dict[str, Any] = {}

    if qp.get("gender"):
        face["gender"] = qp["gender"]
    if qp.get("glasses"):
        face["glasses"] = qp["glasses"]  # "yes"/"no" — fuzzy-matched in _face_keep
    for k in _APPEARANCE_KEYS:
        if qp.get(k):
            appearance[k] = qp[k]

    # Age — explicit RANGE wins ("40 to 55", "between 40 and 55", "age range
    # 40-55"), then an explicit single number, else child/teen/young/elderly words.
    rng = (
        re.search(r"\bage(?:\s*range)?\s*(?:from\s*)?(\d{1,3})\s*(?:to|–|—|-|and|through|thru)\s*(\d{1,3})\b", low)
        or re.search(r"\bbetween\s+(\d{1,3})\s+and\s+(\d{1,3})\b", low)
        or re.search(r"\b(\d{1,3})\s*(?:to|–|—|-)\s*(\d{1,3})\s*(?:years?|yrs?|yo|year[- ]olds?)\b", low)
    )
    m = re.search(r"\b(\d{1,3})\s*(?:years?|yrs?|yo|year[- ]old|years?[- ]old)\b", low)
    if not m:
        m = re.search(r"\bage[d]?\s*(\d{1,3})\b", low)
    if rng:
        buckets = buckets_for_age_range(int(rng.group(1)), int(rng.group(2)))
        if buckets:
            face["age_range"] = buckets
    elif m:
        b = bucket_for_age(int(m.group(1)))
        if b:
            face["age_range"] = b
    elif re.search(r"\b(child|children|kid|kids|toddler|little (?:boy|girl|kid))\b", low):
        face["age_range"] = list(_CHILD)
    elif re.search(r"\b(teen|teenager|adolescent)\b", low):
        face["age_range"] = list(_TEEN)
    elif re.search(r"\b(young)\b", low):
        face["age_range"] = list(_YOUNG)
    elif re.search(r"\b(elderly|old|senior|aged)\b", low):
        face["age_range"] = list(_ELDERLY)

    # boy/girl imply a young person too (QueryParser already set gender).
    if re.search(r"\b(boy|girl)\b", low) and "age_range" not in face:
        face["age_range"] = _CHILD + _TEEN

    # ── Supplemental clothing-colour extraction ──────────────────────────────
    # Fills gaps where QueryParser doesn't extract clothing colours from chat
    # phrasing.  Uses colour as the hard filter (reliable); clothing type is only
    # added when no colour was found (e.g. brand-only queries like "Nike T shirt").
    if not appearance.get("clothing_top_color"):
        _mc = _COLOUR_TOP_RE.search(low)
        if _mc:
            # LLM stores colours in lowercase (e.g. "black", "light_blue");
            # the DB uses exact-match equality so case must match exactly.
            appearance["clothing_top_color"] = _mc.group(1).strip().lower().replace(" ", "_")
            # Intentionally do NOT set clothing_top here: the stored type may differ
            # ("long-sleeve shirt" vs "t-shirt") causing a false AND-filter miss.

    if not appearance.get("clothing_bottom_color"):
        _mc = _COLOUR_BOT_RE.search(low)
        if _mc:
            appearance["clothing_bottom_color"] = _mc.group(1).strip().lower().replace(" ", "_")

    # Standalone top type (no colour), e.g. "Nike T shirt" — keep as soft hint;
    # semantic re-ranking in search() will discriminate by brand / description.
    if not appearance.get("clothing_top") and not _COLOUR_TOP_RE.search(low):
        _mt = _TOP_ONLY_RE.search(low)
        if _mt:
            _tw = _mt.group(1).lower()
            if re.search(r"t[\s-]?shirt", _tw):
                _tw = "t-shirt"
            appearance["clothing_top"] = _tw

    # Ethnicity
    for word, val in _ETHNICITY_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            # Prevent "black T shirt" from assigning ethnicity=Black.
            if word in ("black", "white"):
                # If the word was already captured as a clothing/headwear colour,
                # and it only appears once in the text, it's not an ethnicity.
                word_count = len(re.findall(rf"\b{word}\b", low))
                color_vals = [str(v).lower() for k, v in appearance.items() if "color" in k]
                if word in color_vals and word_count <= 1:
                    continue
            face["ethnicity"] = val
            break

    # Status
    for st in _STATUS_WORDS:
        if re.search(rf"\b{st.replace('_', ' ')}\b", low) or re.search(rf"\b{st}\b", low):
            face["status"] = st
            break

    # Camera N
    cam = re.search(r"\b(?:camera|cam)\s*#?\s*(\d{1,2})\b", low)
    if cam:
        face["camera"] = int(cam.group(1))

    # Name: 'named X' / 'called X'
    nm = re.search(r"\b(?:named|called)\s+([A-Za-z][A-Za-z .'-]{0,40})", text)
    if nm:
        face["name"] = nm.group(1).strip().strip(".")

    return {"face": face, "appearance": appearance, "text": text}


def has_any_filter(filters: Dict[str, Any]) -> bool:
    return bool(filters.get("face")) or bool(filters.get("appearance"))


def describe_filters(filters: Dict[str, Any]) -> str:
    """Human-readable summary of the active filters (for the reply text)."""
    f = filters.get("face", {})
    a = filters.get("appearance", {})
    bits: List[str] = []
    if f.get("gender"):
        bits.append(f["gender"])
    if f.get("age_range"):
        v = f["age_range"]
        bits.append("age " + (", ".join(v) if isinstance(v, list) else str(v)))
    if f.get("ethnicity"):
        bits.append(str(f["ethnicity"]).replace("_", " "))
    if f.get("glasses"):
        bits.append("glasses" if f["glasses"] in ("yes", True) else "no glasses")
    if a.get("clothing_top_color") or a.get("clothing_top"):
        bits.append(" ".join(x for x in (a.get("clothing_top_color"), a.get("clothing_top")) if x))
    if a.get("clothing_bottom_color") or a.get("clothing_bottom"):
        bits.append(" ".join(x for x in (a.get("clothing_bottom_color"), a.get("clothing_bottom")) if x))
    if a.get("headwear"):
        bits.append(" ".join(x for x in (a.get("headwear_color"), a.get("headwear")) if x))
    if a.get("hair_color"):
        bits.append(f"{a['hair_color']} hair")
    if a.get("body_build"):
        bits.append(f"{a['body_build']} build")
    if a.get("accessories"):
        bits.append(", ".join(a["accessories"]))
    if f.get("status"):
        bits.append(f"status {f['status']}")
    if f.get("camera") is not None:
        bits.append(f"camera {f['camera']}")
    if f.get("name"):
        bits.append(f'named "{f["name"]}"')
    return ", ".join(bits) if bits else "anyone"


def facet_hint(ctx, filters: Dict[str, Any]) -> Optional[str]:
    """When a demographic search matched nobody, report which values DO exist so
    the operator can adjust (e.g. they asked for an ethnicity the data lacks).

    Only inspects the hard face facets that commonly cause a zero-result miss:
    ethnicity, gender, age_range. Returns a one-line hint or None.
    """
    face = filters.get("face", {})
    persons = ctx.db.get_all_persons()

    def _present(key: str) -> Dict[str, int]:
        seen: Dict[str, int] = {}
        for p in persons:
            v = p.get(key)
            if v:
                seen[v] = seen.get(v, 0) + 1
        return seen

    bits: List[str] = []
    if face.get("ethnicity"):
        vals = _present("ethnicity")
        if vals:
            listed = ", ".join(f"{k.replace('_', ' ')} ({n})"
                               for k, n in sorted(vals.items(), key=lambda x: -x[1]))
            bits.append(f"Ethnicities on record: {listed}.")
        else:
            bits.append("No ethnicity has been recorded yet — enable face analysis "
                        "to populate it.")
    if face.get("gender"):
        vals = _present("gender")
        if vals:
            bits.append("Genders on record: "
                        + ", ".join(f"{k} ({n})" for k, n in vals.items()) + ".")
    if face.get("age_range"):
        vals = _present("age_range")
        if vals:
            bits.append("Age ranges on record: "
                        + ", ".join(f"{k} ({n})" for k, n in sorted(vals.items())) + ".")
    return " ".join(bits) if bits else None


# ── Filtering ───────────────────────────────────────────────────────────────
def _face_keep(p: Dict[str, Any], f: Dict[str, Any]) -> bool:
    if f.get("status") and (p.get("status") or "") != f["status"]:
        return False
    if f.get("gender") and (p.get("gender") or "").lower() != str(f["gender"]).lower():
        return False
    age = f.get("age_range")
    if age:
        allowed = age if isinstance(age, (list, set, tuple)) else [age]
        if p.get("age_range") not in allowed:
            return False
    if f.get("ethnicity") and (p.get("ethnicity") or "").lower() != str(f["ethnicity"]).lower():
        return False
    gl = f.get("glasses")
    if gl is not None:
        pv = (p.get("glasses") or "").lower()
        has = ("glass" in pv) and ("no" not in pv)
        wants_yes = gl in ("yes", True, "glasses")
        if wants_yes and not has:
            return False
        if (not wants_yes) and has:
            return False
    if f.get("name") and str(f["name"]).lower() not in (p.get("name") or "").lower():
        return False
    return True


def search(ctx, filters: Dict[str, Any], top_k: int = 12) -> Dict[str, Any]:
    """Run a person search from extracted filters. Returns {hits, note, count}.

    Search strategy (in order):
    1. Face/status hard filter (gender, age, ethnicity, status, camera).
    2. Appearance attribute filter via search_persons_by_attributes (colour, type).
    3. Semantic re-ranking of the filtered candidates — ALWAYS when text +
       descriptions are available (not just as a fallback when empty).  This lets
       brand names (Nike, Adidas) and nuanced details float to the top even when
       a structured filter already found candidates.
    4. Pure semantic fallback (all persons) only when steps 1+2 left no hits.
    """
    face = filters.get("face", {})
    appearance = filters.get("appearance", {})
    text = filters.get("text")

    persons = ctx.db.get_all_persons()
    described_any = any(p.get("latest_description_id") for p in persons)

    base = [p for p in persons if _face_keep(p, face)]
    if face.get("camera") is not None:
        cam = face["camera"]
        base = [p for p in base if cam in ctx.db.get_cameras_for_person(p["person_id"])]
    base_ids = {p["person_id"] for p in base}
    ids = set(base_ids)

    note: Optional[str] = None
    if appearance:
        if described_any:
            attr_rows = ctx.db.search_persons_by_attributes(appearance, limit=10_000)
            ids &= {r["person_id"] for r in attr_rows}
        else:
            note = ("Clothing/appearance isn't indexed yet (no LLM descriptions) — "
                    "I matched by face attributes only. Run `--describe-all` to enable "
                    "clothing search.")

    ordered = [p for p in base if p["person_id"] in ids]

    # ── Semantic scoring (always when text + descriptions available) ─────────
    # Re-rank the structured-filtered candidates by MiniLM cosine similarity.
    # This is critical for queries like "Nike T shirt": gender=Male finds 5 people
    # but without re-ranking, they're returned in DB order instead of by relevance.
    semantic_scores: Dict[str, float] = {}
    sem_full: List[dict] = []
    if text and described_any:
        tse = engine.get_text_search_engine(ctx.db)
        # Fetch enough candidates to cover all filtered persons + some extra
        fetch_k = max(top_k * 4, 30)
        sem_full = tse.search(_clean_query(text), top_k=fetch_k)
        semantic_scores = {r["person_id"]: r.get("score", 0.0) for r in sem_full}

    if ordered:
        if semantic_scores:
            # Re-rank: persons absent from semantic results get score -1 so they
            # sink to the bottom rather than being excluded entirely.
            ordered.sort(
                key=lambda p: semantic_scores.get(p["person_id"], -1.0),
                reverse=True,
            )
        hits = [
            services.search_hit(
                ctx.db, ctx.snapshots_dir, p["person_id"],
                score=semantic_scores.get(p["person_id"]),
            )
            for p in ordered
        ][:top_k]
    else:
        hits = []

    # Pure semantic fallback: structured appearance filters matched nobody.
    # Restrict to base_ids to ensure we never violate hard constraints like Gender.
    if not hits and sem_full:
        fallback_candidates = [r for r in sem_full if r["person_id"] in base_ids]
        hits = [
            services.search_hit(
                ctx.db, ctx.snapshots_dir, r["person_id"],
                score=r.get("score"), summary=r.get("summary"),
            )
            for r in fallback_candidates[:top_k]
        ]
        if hits and note is None:
            note = "No exact attribute match — showing the closest by meaning."

    return {"hits": hits, "note": note, "count": len(hits)}


def get_person(ctx, person_id: str) -> Optional[Dict[str, Any]]:
    return services.person_detail(ctx.db, ctx.snapshots_dir, person_id, config.overlap_groups())


def person_exists(ctx, person_id: str) -> bool:
    return ctx.db.get_person(person_id) is not None


def resolve_person_id(ctx, token: str) -> Optional[str]:
    """Resolve a full UUID or a short 'P:abc123' / 'abc123' prefix to a person_id."""
    token = token.strip().lstrip("P:").lstrip("p:").strip()
    if not token:
        return None
    for p in ctx.db.get_all_persons():
        pid = p["person_id"]
        if pid == token or pid.startswith(token):
            return pid
    return None


def _read_alerts() -> List[dict]:
    try:
        data = json.loads(config.DEMO_ALERTS_LOG.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception:  # noqa: BLE001
        return []


def stats(ctx) -> Dict[str, Any]:
    return services.build_stats(ctx.db, alert_count=len(_read_alerts()))


def alerts(ctx, limit: int = 10) -> List[dict]:
    return list(reversed(_read_alerts()))[:limit]
