"""LLM-backed pick-one-choice for Aiko's game selfplay.

Replaces the retired Jev decision API: Aiko now thinks her game moves with
the chat LLM (Ministral on LLM_BASE_URL) instead of a hosted classifier.

choice(state, instructions, criteria) -> (pick, {}, 0.0):
    Asks the LLM to reply with exactly one of the candidate keys.
    Raises LLMError when the reply names no candidate or the LLM is
    unreachable — callers must void the turn/game rather than guess.

decide(call, *, label): retry wrapper; raises LLMUnavailable on exhaustion.
"""
from __future__ import annotations

import json
import os
import re
import time

from system.log import get_logger
from .common import chat_completions_create

log = get_logger(__name__)

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "ministral")
_LLM_CLIENT = None


class LLMError(Exception):
    """Fatal LLM choice failure (bad reply or unreachable backend)."""


class LLMUnavailable(LLMError):
    """LLM could not decide after retries. Callers must void the turn/game —
    a guessed move would pollute learning with fiction."""


def _get_client():
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        from openai import OpenAI

        _LLM_CLIENT = OpenAI(
            base_url=LLM_BASE_URL,
            api_key=os.getenv("LLM_API_KEY", "") or "not-needed",
        )
    return _LLM_CLIENT


def choice(state: dict, instructions: str, criteria: dict, **kw) -> tuple[str, dict, float]:
    """Pick one option via the chat LLM. Returns (option, {}, 0.0)."""
    if not criteria:
        raise LLMError("choice needs at least one candidate")
    options = "\n".join(f"- {key}: {desc}" for key, desc in criteria.items())
    prompt = (
        f"{instructions}\n\n"
        f"Game state: {json.dumps(state, ensure_ascii=False)[:2000]}\n\n"
        f"Reply with EXACTLY ONE of these option keys and nothing else:\n{options}"
    )
    try:
        resp = chat_completions_create(
            _get_client(),
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": "You are Aiko choosing a game move. Reply with exactly one option key and nothing else."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=32,
            temperature=0.0,
        )
        raw = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        raise LLMError(f"LLM choice call failed: {exc}") from exc
    pick = _extract_pick(raw, criteria)
    if pick is None:
        raise LLMError(f"LLM reply named no candidate: {raw[:200]!r}")
    return (pick, {}, 0.0)


def _extract_pick(raw: str, criteria: dict) -> str | None:
    """Find a candidate key in the LLM reply: exact match first, then a
    standalone token match, so 'take2' is found in 'I choose take2.'."""
    text = raw.strip()
    if text in criteria:
        return text
    for key in criteria:
        if re.search(r"(?<![A-Za-z0-9_])" + re.escape(key) + r"(?![A-Za-z0-9_])", text):
            return key
    return None


def _decide_retries() -> int:
    try:
        return max(0, int(os.getenv("SELFPLAY_LLM_RETRIES", "3")))
    except Exception:
        return 3


def decide(call, *, label: str = "decision"):
    """Run a zero-arg LLM choice call with retries; raise LLMUnavailable on
    exhaustion. Everything gets SELFPLAY_LLM_RETRIES attempts (default 3)
    with capped backoff."""
    tries = max(1, _decide_retries())
    last: Exception | None = None
    for attempt in range(tries):
        try:
            return call()
        except Exception as exc:
            last = exc
            log.debug("llm %s failed (attempt %d/%d): %s", label, attempt + 1, tries, exc)
            if attempt + 1 < tries:
                time.sleep(min(30.0, 2.0 ** attempt))
    raise LLMUnavailable(f"llm {label} failed after {tries} attempts: {last}")
