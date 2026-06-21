"""
webapp/api/chatbot/brain.py
---------------------------
The assistant's tiered "brain".

Tier 1 (default, instant, offline): rule-based intent routing + the engine's
rule-based QueryParser (via tools.extract_filters). Handles greetings, help,
search, multi-turn refinement, person lookup, stats/alerts, and PROPOSING write
actions.

Tier 2 (optional, Ollama): only invoked for messages Tier 1 can't classify, to
produce a friendly free-form reply. Any failure/timeout falls back to a Tier-1
clarification prompt — so the assistant always answers, even with Ollama down.

`respond()` is pure w.r.t. side effects: it reads the DB to resolve references
but never writes. Writes are returned as a ``proposed_action`` and executed by
the router only after the user confirms (and a role check passes).
"""

import re
from typing import Any, Dict, List, Optional

from . import tools

# ── Intent patterns ─────────────────────────────────────────────────────────
_GREET = re.compile(
    r"^\s*(hi+|hello+|hey+|yo|hiya|howdy|greetings|good\s+(morning|afternoon|evening)"
    r"|salam|salaam|as-?salam[ou]?[ -]?alaikum)\b",
    re.I,
)
_THANKS = re.compile(r"\b(thanks|thank you|thx|shukran|appreciate)\b", re.I)
_HELP = re.compile(r"\b(help|what can you do|capabilities|how do you work|what do you do)\b", re.I)
_STATS = re.compile(
    r"\b(stats|statistics|how many|count|overview|summar(y|ise|ize)|total (number|persons|people))\b",
    re.I,
)
_ALERTS = re.compile(r"\b(alerts?|violence|incidents?)\b", re.I)
_SEARCH_VERB = re.compile(
    r"\b(find|search|look(ing)?\s+for|show me|locate|who('?s|\s+is)|anyone|any (?:one|person|people)"
    r"|spot|identify|get me|is there|are there)\b",
    re.I,
)
_LOOKUP = re.compile(r"\b(open|show|details?|view|tell me about|info on|who is)\b", re.I)
_REFINE_PREFIX = re.compile(
    r"^\s*(only|just|also|and|but|make it|change|instead|now|what about|how about|"
    r"no |not |add |with |without |on camera|in camera)\b",
    re.I,
)
_RESET = re.compile(r"\b(reset|start over|new search|forget (that|it)|clear)\b", re.I)
# An explicit mutation verb. Attribute edits (set gender/age/ethnicity/name…) are
# only proposed when one of these is present — otherwise a *search* like
# "find anyone with age 55" was being misread as an edit of the last result.
_EDIT_VERB = re.compile(
    r"\b(set|change|changing|rename|re-?name|relabel|mark|update|edit|correct|fix|"
    r"assign|call\s+(?:it|this|that|them|him|her)|make\s+(?:it|this|that|them|him|her))\b",
    re.I,
)
_PID = re.compile(r"\bP?:?\s*([0-9a-fA-F]{6,32})\b")
_AFFIRM = re.compile(r"^\s*(y|yes|yeah|yep|yup|confirm|do it|go ahead|sure|ok(ay)?|proceed|please do)\b", re.I)
_NEGATE = re.compile(r"^\s*(n|no|nope|cancel|stop|don'?t|abort|nevermind|never mind)\b", re.I)

_ORDINALS = {
    "first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2,
    "fourth": 3, "4th": 3, "fifth": 4, "5th": 4, "last": -1,
}

# "find only one" / "show me just two" → top_k limit
_QUANTITY = re.compile(
    r"\b(?:only|just|show)\s+(?:me\s+)?(?:(?:the|a)\s+)?"
    r"(one|1|two|2|three|3|four|4|five|5|six|6|seven|7|eight|8|nine|9|ten|10)\b",
    re.I,
)
_QUANTITY_WORDS = {
    "one": 1,   "1": 1,  "two": 2,   "2": 2,  "three": 3, "3": 3,
    "four": 4,  "4": 4,  "five": 5,  "5": 5,  "six": 6,   "6": 6,
    "seven": 7, "7": 7,  "eight": 8, "8": 8,  "nine": 9,  "9": 9,
    "ten": 10,  "10": 10,
}

HELP_TEXT = (
    "Here's what I can do:\n"
    "• **Search** people by description — \"find a child about 9 in a red t-shirt\", "
    "\"a man with a backpack on camera 3\", \"women with glasses\".\n"
    "• **Refine** the last search — \"only camera 2\", \"make it blue\", \"also a hat\".\n"
    "• **Open** a result — \"open the first one\" or \"details on P:3601b2\".\n"
    "• **Stats** — \"how many people?\", \"how many on camera 2?\", \"any alerts?\".\n"
    "• **Corrections** (with your confirmation) — \"merge the first two\", "
    "\"delete the second one\", \"re-describe this person\", \"rename it to Ali\"."
)


def is_affirmative(text: str) -> bool:
    return bool(_AFFIRM.search(text))


def is_negative(text: str) -> bool:
    return bool(_NEGATE.search(text))


def _short(pid: str) -> str:
    return pid[:8] if pid else "?"


def _payload(reply: str, *, results=None, open_person_id=None, proposed_action=None,
             active_filters: Optional[Dict] = None, last_results: Optional[List[str]] = None,
             pending_action: Any = "__keep__") -> Dict[str, Any]:
    return {
        "reply": reply,
        "results": results,
        "open_person_id": open_person_id,
        "proposed_action": proposed_action,
        "active_filters": active_filters,
        "last_results": last_results,
        "pending_action": pending_action,
    }


# ── Reference resolution ─────────────────────────────────────────────────────
def _resolve_one(ctx, text: str, last: List[str]) -> Optional[str]:
    """Resolve a single person reference: explicit ID > ordinal/#N > sole context."""
    low = text.lower()
    for m in _PID.finditer(text):
        pid = tools.resolve_person_id(ctx, m.group(1))
        if pid:
            return pid
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", low):
            if -len(last) <= idx < len(last):
                return last[idx]
    m = re.search(r"\b(?:number|#)\s*(\d{1,2})\b", low)
    if m:
        i = int(m.group(1)) - 1
        if 0 <= i < len(last):
            return last[i]
    if re.search(r"\b(this|that|the|him|her|them|it|the person)\b", low) and len(last) == 1:
        return last[0]
    if len(last) == 1:
        return last[0]
    return None


def _resolve_two(ctx, text: str, last: List[str]) -> List[str]:
    ids: List[str] = []
    for m in _PID.finditer(text):
        pid = tools.resolve_person_id(ctx, m.group(1))
        if pid and pid not in ids:
            ids.append(pid)
    if len(ids) >= 2:
        return ids[:2]
    # ordinal pairs
    picked = []
    for word, idx in _ORDINALS.items():
        if re.search(rf"\b{re.escape(word)}\b", text.lower()) and -len(last) <= idx < len(last):
            picked.append(last[idx])
    picked = list(dict.fromkeys(picked))
    if len(picked) >= 2:
        return picked[:2]
    if re.search(r"\b(those|these|the|first|top)\s+two\b|\bthem\b|\bboth\b", text.lower()) and len(last) >= 2:
        return last[:2]
    if len(last) == 2:
        return last
    return ids


# ── Write detection (proposes; never executes) ──────────────────────────────
def _detect_write(ctx, text: str, last: List[str]) -> Optional[Dict[str, Any]]:
    low = text.lower()

    if re.search(r"\bmerge\b", low):
        ids = _resolve_two(ctx, text, last)
        if len(ids) >= 2:
            keep, remove = ids[0], ids[1]
            return _payload(
                f"Merge {_short(remove)} into {_short(keep)}? All of {_short(remove)}'s "
                f"embeddings move to {_short(keep)} and {_short(remove)} is removed. "
                f"Confirm? (yes/no)",
                proposed_action={"type": "merge", "args": {"keep_id": keep, "remove_id": remove},
                                 "summary": f"Merge {_short(remove)} → {_short(keep)}"},
                pending_action={"type": "merge", "args": {"keep_id": keep, "remove_id": remove}},
            )
        return _payload("Which two people should I merge? Run a search first, then say "
                        "\"merge the first two\", or give me two IDs.")

    if re.search(r"\b(delete|remove)\b", low) and not _NEGATE.match(text):
        pid = _resolve_one(ctx, text, last)
        if pid:
            return _payload(
                f"Delete person {_short(pid)} and all their data (embeddings, history, "
                f"snapshots)? This can't be undone. Confirm? (yes/no)",
                proposed_action={"type": "delete", "args": {"person_id": pid},
                                 "summary": f"Delete {_short(pid)}"},
                pending_action={"type": "delete", "args": {"person_id": pid}},
            )
        return _payload("Which person should I delete? Open one, or say \"delete the first one\".")

    if re.search(r"\bsplit\b", low):
        return _payload("Splitting a person needs you to pick which gallery rows go to the new "
                        "ID — open the person in the People page and use Split there. I can't "
                        "pick the rows from chat yet.")

    if re.search(r"\b(re-?describe|describe (again|him|her|them|this|the person))\b", low):
        pid = _resolve_one(ctx, text, last)
        if pid:
            return _payload(
                f"Queue person {_short(pid)} for re-description with the VLM (needs Ollama)? (yes/no)",
                proposed_action={"type": "redescribe", "args": {"person_id": pid},
                                 "summary": f"Re-describe {_short(pid)}"},
                pending_action={"type": "redescribe", "args": {"person_id": pid}},
            )
        return _payload("Which person should I re-describe? Open one or say \"re-describe the first one\".")

    # Attribute edits require an explicit edit verb (set/change/rename/mark…).
    # Without one, phrases like "age 55" or "a man" are SEARCH terms, not edits.
    fields = _parse_attr_edits(text) if _EDIT_VERB.search(text) else {}
    if fields:
        pid = _resolve_one(ctx, text, last)
        if pid:
            human = ", ".join(f"{k}={v}" for k, v in fields.items())
            return _payload(
                f"Set {human} on person {_short(pid)}? Confirm? (yes/no)",
                proposed_action={"type": "edit_attributes", "args": {"person_id": pid, "fields": fields},
                                 "summary": f"Edit {_short(pid)}: {human}"},
                pending_action={"type": "edit_attributes", "args": {"person_id": pid, "fields": fields}},
            )
        return _payload("Which person should I edit? Open one first, or say \"rename the first one to …\".")

    return None


def _parse_attr_edits(text: str) -> Dict[str, str]:
    """Extract editable persons-row fields from a 'set/rename' instruction."""
    low = text.lower()
    out: Dict[str, str] = {}
    m = re.search(r"\b(?:rename|name|call)\b.*?\bto\s+([A-Za-z][\w .'-]{0,40})", text, re.I)
    if not m:
        m = re.search(r"\b(?:rename|name|call)\s+(?:it|this|that|them|him|her|the (?:first|one))\s+([A-Za-z][\w .'-]{0,40})", text, re.I)
    if m:
        out["name"] = m.group(1).strip().strip(".")
    g = re.search(r"\b(?:gender|sex)\s+(?:to\s+)?(male|female|man|woman)\b", low)
    if g:
        out["gender"] = "Male" if g.group(1) in ("male", "man") else "Female"
    e = re.search(r"\bethnicity\s+(?:to\s+)?([a-z_ ]+)\b", low)
    if e:
        out["ethnicity"] = e.group(1).strip().title().replace(" ", "_")
    a = re.search(r"\bage(?:\s*range)?\s+(?:to\s+)?(\d{1,3})\b", low)
    if a:
        b = tools.bucket_for_age(int(a.group(1)))
        if b:
            out["age_range"] = b
    if re.search(r"\b(set|mark|give)\b.*\bglasses\b", low):
        out["glasses"] = "No Glasses" if re.search(r"\bno\b", low) else "Glasses"
    return out


# ── Lookup ───────────────────────────────────────────────────────────────────
def _detect_lookup(ctx, session, text: str, last: List[str]) -> Optional[Dict[str, Any]]:
    if not _LOOKUP.search(text):
        return None
    # A lookup opens ONE specific, already-known person. If the message instead
    # carries search filters (glasses, ethnicity, age, colour…) or a search verb
    # ("show me anyone…", "find…"), it's a new search — don't let the sole-result
    # fallback in _resolve_one hijack it into opening a leftover person. An
    # explicit P:id is always an unambiguous lookup, so it bypasses this guard.
    if not _PID.search(text) and (
        _SEARCH_VERB.search(text) or tools.has_any_filter(tools.extract_filters(text))
    ):
        return None
    pid = _resolve_one(ctx, text, last)
    if not pid:
        return None
    detail = tools.get_person(ctx, pid)
    if not detail:
        return _payload(_generate_reply(session, text, f"I couldn't find person {_short(pid)}."))
    cams = ", ".join(str(c) for c in detail.get("cameras", [])) or "—"
    desc = (detail.get("description") or {}).get("summary")
    bits = [b for b in (detail.get("gender"), detail.get("age_range"),
                        detail.get("ethnicity"), detail.get("glasses")) if b]
    line = f"Person {_short(pid)}: " + (", ".join(bits) if bits else "no face attributes") + "."
    line += f" Seen on camera(s) {cams}, {detail.get('gallery_size') or 0} body embeddings."
    if desc:
        line += f" Description: {desc}"
    return _payload(_generate_reply(session, text, line), open_person_id=pid,
                    results=[tools_summary(ctx, pid)], last_results=[pid])


def tools_summary(ctx, pid: str):
    from .. import services  # noqa: PLC0415

    return services.search_hit(ctx.db, ctx.snapshots_dir, pid)


# ── Stats / alerts ───────────────────────────────────────────────────────────
def _stats_reply(ctx, session, text: str, active, last) -> Dict[str, Any]:
    s = tools.stats(ctx)
    low = text.lower()
    m = re.search(r"\b(?:camera|cam)\s*#?\s*(\d+)\b", low)
    if m:
        cam = m.group(1)
        n = s["per_camera_sightings"].get(cam, 0)
        return _payload(_generate_reply(session, text, f"Camera {cam} has {n} sighting(s) on record."),
                        active_filters=active, last_results=last)
    by_status = ", ".join(f"{k}: {v}" for k, v in s["by_status"].items() if k)
    reply = (f"{s['persons']} people tracked — {s['multi_camera']} seen on multiple cameras, "
             f"{s['described']} described ({s['undescribed']} not), "
             f"{s['total_body_embeddings']} body embeddings, {s['alerts']} alert(s)."
             + (f" Status — {by_status}." if by_status else ""))
    return _payload(_generate_reply(session, text, reply), active_filters=active, last_results=last)


def _alerts_reply(ctx, session, text: str, active, last) -> Dict[str, Any]:
    items = tools.alerts(ctx, limit=5)
    if not items:
        return _payload(_generate_reply(session, text, "No violence/anomaly alerts on record."), active_filters=active, last_results=last)
    lines = [f"{it.get('label') or it.get('level') or 'ALERT'} on cam {it.get('cam_id')} "
             f"({it.get('score')}) at {it.get('timestamp')}" for it in items]
    reply = "Most recent alerts:\n• " + "\n• ".join(lines)
    return _payload(_generate_reply(session, text, reply),
                    active_filters=active, last_results=last)


# ── Filter merge ─────────────────────────────────────────────────────────────
def _merge_filters(active: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    out = {"face": dict(active.get("face", {})), "appearance": dict(active.get("appearance", {}))}
    out["face"].update(new.get("face", {}))
    out["appearance"].update(new.get("appearance", {}))
    out["text"] = new.get("text") or active.get("text")
    return out


def _looks_like_refinement(text: str, has_filters: bool) -> bool:
    return bool(_REFINE_PREFIX.match(text)) or (has_filters and not _SEARCH_VERB.search(text))


# ── Tier 2 — Ollama free-form fallback (optional) ───────────────────────────
# Strong, role-based system prompt. The brain is deterministic and authoritative
# about WHAT to say; the LLM only controls HOW it is phrased. The hard rules below
# keep it from dropping facts, inventing data, or apologising its way out of a
# correct answer.
_ASSISTANT_ROLE = (
    "You are SURVEILLANT's command-center assistant — a precise, professional, "
    "operator-facing helper inside a multi-camera person re-identification and "
    "surveillance system. The system re-identifies people across cameras (OSNet + "
    "ByteTrack body Re-ID, InsightFace face attributes, and LLM appearance "
    "descriptions). Through you the operator can search people by description, open "
    "a person's profile and cross-camera journey, report statistics and alerts, and "
    "— only after explicit confirmation — merge, delete, re-describe or relabel "
    "identities.\n"
    "Voice: concise, factual, confident, security-professional. No chit-chat, no "
    "apologetic padding, no emojis. At most one short paragraph."
)


def _llm_assist(session, text: str, tool_context: Optional[str] = None) -> Optional[str]:
    """Single short Ollama call. Two modes:
      • polish (``tool_context`` set): rephrase a deterministic, authoritative
        reply in a natural voice WITHOUT altering any fact, number, ID or
        suggestion it contains.
      • fallback (no ``tool_context``): answer a message Tier-1 couldn't classify.
    Returns a reply string, or None on any failure/timeout/disabled (the caller
    then uses the deterministic text)."""
    try:
        import requests  # noqa: PLC0415
        from surveillant.config.settings import (  # noqa: PLC0415
            OLLAMA_HOST, OLLAMA_QUERY_MODEL, CHAT_LLM_POLISH, CHAT_LLM_TIMEOUT_SEC,
        )
    except Exception:  # noqa: BLE001
        return None

    # Polishing a deterministic reply is optional (off → instant, fully offline).
    if tool_context is not None and not CHAT_LLM_POLISH:
        return None

    history = session.get("_recent_msgs") or []
    if tool_context is not None:
        sys_prompt = (
            _ASSISTANT_ROLE + "\n\n"
            "The system just executed the operator's request and produced this exact, "
            f"authoritative result:\n[{tool_context}]\n\n"
            "Relay it in your own natural voice. ABSOLUTE RULES:\n"
            "1. Preserve EVERY fact, number, count, ID, camera and name EXACTLY — "
            "never change, round, drop or add one.\n"
            "2. Keep any guidance the result gives (what to try next, which values "
            "exist) — do not delete the helpful part.\n"
            "3. Never invent data the result does not state. If it found nothing, say "
            "so plainly and pass along any suggested alternatives.\n"
            "4. Do not claim or imply any action the result did not actually take.\n"
            'Reply with JSON: {"reply": "..."}.'
        )
    else:
        sys_prompt = (
            _ASSISTANT_ROLE + "\n\n"
            "The operator said something the deterministic router could not map to a "
            "known action. Answer briefly and helpfully, steering them toward a "
            "concrete capability (a description search, opening a person, stats, or a "
            "correction). You have NO search results here — never claim to have "
            "searched or found anyone, and never invent people, counts or IDs.\n"
            'Reply with JSON: {"reply": "..."}.'
        )
    msgs = [{"role": "system", "content": sys_prompt}]
    for m in history[-4:]:
        msgs.append({"role": m["role"], "content": m["content"]})
    msgs.append({"role": "user", "content": text})

    try:
        r = requests.post(
            f"{OLLAMA_HOST.rstrip('/')}/api/chat",
            json={"model": OLLAMA_QUERY_MODEL, "stream": False, "format": "json",
                  "keep_alive": "10m", "messages": msgs,
                  "options": {"temperature": 0.2, "num_ctx": 2048}},
            timeout=CHAT_LLM_TIMEOUT_SEC,
        )
        r.raise_for_status()
        import json  # noqa: PLC0415

        content = (r.json().get("message") or {}).get("content") or ""
        data = json.loads(content)
        reply = data.get("reply")
        return reply.strip() if isinstance(reply, str) and reply.strip() else None
    except Exception:  # noqa: BLE001
        return None


def _generate_reply(session, text: str, default_reply: str) -> str:
    """Try to generate a conversational response using the LLM given the tool output.
    Falls back to the raw hardcoded string if the LLM fails."""
    llm_reply = _llm_assist(session, text, tool_context=default_reply)
    return llm_reply if llm_reply else default_reply


# ── Main entry ───────────────────────────────────────────────────────────────
def respond(ctx, session: Dict[str, Any], text: str) -> Dict[str, Any]:
    active = session.get("active_filters") or {}
    last = session.get("last_results") or []
    text = (text or "").strip()
    if not text:
        return _payload("Say something and I'll help — try \"find a man with a backpack\".",
                        active_filters=active, last_results=last)

    # 1. Smalltalk
    if _GREET.search(text) and len(text.split()) <= 5:
        reply = (
            "Hi! How can I assist you? I can find people by description (age, clothing "
            "colour, camera, gender…), open a person's details and journey, report stats, "
            "or make corrections like merging two IDs. "
            "Try: \"find a child about 9 wearing a red t-shirt\"."
        )
        return _payload(_generate_reply(session, text, reply), active_filters=active, last_results=last)
    if _HELP.search(text):
        return _payload(HELP_TEXT, active_filters=active, last_results=last)
    if _THANKS.search(text) and not _SEARCH_VERB.search(text) and not tools.has_any_filter(tools.extract_filters(text)):
        return _payload(_generate_reply(session, text, "You're welcome! Anything else?"), active_filters=active, last_results=last)

    # 2. Stats / alerts
    if _STATS.search(text):
        return _stats_reply(ctx, session, text, active, last)
    if _ALERTS.search(text) and not _SEARCH_VERB.search(text):
        return _alerts_reply(ctx, session, text, active, last)

    # 3. Writes (propose, don't execute)
    wr = _detect_write(ctx, text, last)
    if wr is not None:
        # keep active/last unchanged on a proposal
        wr["active_filters"] = active
        wr["last_results"] = last
        return wr

    # 4. Lookup
    lk = _detect_lookup(ctx, session, text, last)
    if lk is not None:
        lk["active_filters"] = active
        return lk

    # 5. Search / refine
    extracted = tools.extract_filters(text)
    has_filters = tools.has_any_filter(extracted)
    is_refine = bool(active) and _looks_like_refinement(text, has_filters)

    if has_filters or _SEARCH_VERB.search(text) or is_refine:
        if _RESET.search(text):
            active = {}
            is_refine = False
        filters = _merge_filters(active, extracted) if is_refine else extracted
        # ── Honour "only one" / "just two" quantity hints ─────────────────────
        _top_k = 12
        _qty_m = _QUANTITY.search(text)
        if _qty_m:
            _top_k = _QUANTITY_WORDS.get(_qty_m.group(1).lower(), 12)
        res = tools.search(ctx, filters, top_k=_top_k)
        hits = res["hits"]
        summary = tools.describe_filters(filters)
        if hits:
            reply = f"I found {len(hits)} {'person' if len(hits) == 1 else 'people'} matching {summary}."
            if res.get("note"):
                reply += " " + res["note"]
        else:
            reply = f"I couldn't find anyone matching {summary}."
            # Tell the operator which values ARE on record (e.g. they asked for an
            # ethnicity the demo data doesn't contain), instead of a flat "no".
            hint = tools.facet_hint(ctx, filters)
            if res.get("note"):
                reply += " " + res["note"]
            elif hint:
                reply += " " + hint
            else:
                reply += " Want to adjust the age, colour, or camera?"

        reply = _generate_reply(session, text, reply)
        return _payload(reply, results=hits, active_filters=filters,
                        last_results=[h["person_id"] for h in hits])

    # 6. Unknown -> Tier 2, else clarify
    reply = _llm_assist(session, text)
    if reply:
        return _payload(reply, active_filters=active, last_results=last)
    return _payload(
        "I can search for people, open a person, or report stats. "
        "Try: \"find a man with a backpack on camera 3\", or type \"help\".",
        active_filters=active, last_results=last)
