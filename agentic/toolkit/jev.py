"""Jev decision-model client (TypeSafe API) for Aiko.

Jev (https://docs.typesafe.ai/api) is a fast classification/decision model:
ask yes/no (noul), pick-an-option (choice), or rate (score) questions about
any state. Aiko uses it for shogi move decisions (choice over candidate
moves) and position/game review (score/noul).

Auth: ``JEV_API_KEY`` from the environment (production: add it to .env.age
via ``./util/edit_dotenv.sh`` — never commit the key). ``JEV_MODEL``
overrides the model (default ``jev-latest``).

Stdlib only (urllib). Bounded timeouts + exponential backoff on 429/529.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from system.log import get_logger

log = get_logger(__name__)

JEV_API_URL = "https://api.typesafe.ai/v1/systemone"
JEV_MODEL_DEFAULT = "jev-latest"
JEV_TIMEOUT_S = 30.0
JEV_MAX_RETRIES = 3


class JevError(Exception):
    """Fatal Jev failure (auth, validation, or exhausted retries)."""


class JevAuthError(JevError):
    """Missing/invalid API key — fix config, don't retry."""


def _api_key() -> str:
    try:
        key = (os.getenv("JEV_API_KEY", "") or "").strip()
    except Exception:
        key = ""
    if not key:
        raise JevAuthError("JEV_API_KEY is not set (add it via ./util/edit_dotenv.sh)")
    return key


def _model() -> str:
    try:
        return (os.getenv("JEV_MODEL", "") or "").strip() or JEV_MODEL_DEFAULT
    except Exception:
        return JEV_MODEL_DEFAULT


def evaluate(state, questions: dict, *, model: str | None = None,
             timeout: float = JEV_TIMEOUT_S) -> dict:
    """POST one evaluation. Returns {"answers": {...}, "usage": {...}, "model": ...}."""
    if not questions:
        raise ValueError("questions must not be empty")
    body = json.dumps({
        "state": state,
        "model": model or _model(),
        "questions": questions,
    }).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {_api_key()}",
        "Content-Type": "application/json",
    }
    last_err: Exception | None = None
    for attempt in range(JEV_MAX_RETRIES):
        req = urllib.request.Request(JEV_API_URL, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if not isinstance(payload.get("answers"), dict):
                raise JevError("malformed Jev response: missing answers map")
            return payload
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")[:500]
            except Exception:
                detail = ""
            if exc.code == 401:
                raise JevAuthError(f"Jev rejected the API key (401): {detail}")
            if exc.code == 422:
                raise JevError(f"Jev rejected the request (422): {detail}")
            if exc.code in (429, 529):
                last_err = JevError(f"Jev throttled/overloaded ({exc.code}), retrying")
                log.debug("jev %s, attempt %d", last_err, attempt + 1)
                time.sleep(min(8.0, 2.0 ** attempt))
                continue
            raise JevError(f"Jev HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_err = JevError(f"Jev transport failed: {exc}")
            log.debug("jev transport failed, attempt %d: %s", attempt + 1, exc)
            time.sleep(min(8.0, 2.0 ** attempt))
    raise last_err or JevError("Jev failed without a recorded error")


def choice(state, instructions: str, criteria: dict, **kw) -> tuple[str, dict, float]:
    """Pick one option. Returns (option, probabilities, confidence)."""
    out = evaluate(state, {"q": {"type": "choice", "instructions": instructions,
                                 "criteria": criteria}}, **kw)
    ans = out["answers"]["q"]
    if ans.get("type") != "choice" or "choice" not in ans:
        raise JevError(f"unexpected choice answer: {str(ans)[:200]}")
    return (str(ans["choice"]), dict(ans.get("probabilities") or {}),
            float(ans.get("confidence", 0.0)))


def noul(state, instructions: str, criteria: dict | None = None, **kw) -> float:
    """Yes/no as probability. Returns float in [0, 1]."""
    q: dict = {"type": "noul", "instructions": instructions}
    if criteria:
        q["criteria"] = criteria
    out = evaluate(state, {"q": q}, **kw)
    ans = out["answers"]["q"]
    if ans.get("type") != "noul" or "noul" not in ans:
        raise JevError(f"unexpected noul answer: {str(ans)[:200]}")
    return max(0.0, min(1.0, float(ans["noul"])))


def score(state, instructions: str, criteria: list, **kw) -> tuple[float, dict, float]:
    """Rate along rubric levels. Returns (value, probabilities, confidence)."""
    if not criteria or len(criteria) < 2:
        raise ValueError("score criteria need at least two levels")
    out = evaluate(state, {"q": {"type": "score", "instructions": instructions,
                                 "criteria": list(criteria)}}, **kw)
    ans = out["answers"]["q"]
    if ans.get("type") != "score" or "score" not in ans:
        raise JevError(f"unexpected score answer: {str(ans)[:200]}")
    return (float(ans["score"]), dict(ans.get("probabilities") or {}),
            float(ans.get("confidence", 0.0)))
