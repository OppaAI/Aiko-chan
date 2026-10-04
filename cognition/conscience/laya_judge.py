"""Laya-backed conscience judge: typed decisions over HTTP, no text generation.

`SLMJudge` speaks OpenAI `/chat/completions` because it was built for a
generative model. The fine-tuned Laya checkpoint does not generate text at all
-- it answers typed `choice` questions and returns probabilities, which is a
better fit for a conscience: the verdict is a constrained label rather than
parsed free text.

Drop-in for `SLMJudge`: same `score()` signature, same `(vertical, horizontal,
confidence, reasons, cited)` tuple, same three-strike breaker, so `core._run`
can use either without knowing which model is behind it.

Band mapping is deliberate and lossy in one direction only:

    vertical    aligned=+1.0  unknown=0.0  contrary=-1.0
    horizontal  benefits=+1.0 unknown=0.0  harms=-1.0

`unknown` maps to 0.0 so it lands on "no moral signal" in `schema.decide()`
rather than being silently promoted to `aligned`. That is the safe direction:
the autonomy/stakes layer decides whether an unremarkable verdict still needs
a human, not this layer.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any

from .schema import Party
from system.config import env_float, env_int, env_str

log = logging.getLogger(__name__)

LAYA_BASE_URL = env_str("CCC_LAYA_BASE_URL", "")
# Empty = do not send a `model` field at all, which is correct when serving a
# single named GGUF. If you do send one, it must match the server's INTERNAL
# model name (`laya-multilingual`, `laya`, ...), NOT the filename: the server
# validates it and answers `unknown model '...'` for anything else.
LAYA_MODEL_NAME = env_str("CCC_LAYA_MODEL", "")
LAYA_DECIDE_PATH = env_str("CCC_LAYA_DECIDE_PATH", "/v1/decide")
LAYA_TIMEOUT = env_float("CCC_LAYA_TIMEOUT", 2.5)
LAYA_FAILURE_LIMIT = env_int("CCC_LAYA_FAILURE_LIMIT", 3)

_VERTICAL_Q = {
    "type": "choice",
    "instructions": "Is this request aligned with God's will?",
    "criteria": {
        "aligned": "Aligned with God's will: honest, merciful, diligent, or faithful.",
        "contrary": "Contrary to God's will: it requires deception, impersonation, or harm.",
        "unknown": "Unknown: ordinary or borderline, with no clear moral content either way.",
    },
}
_HORIZONTAL_Q = {
    "type": "choice",
    "instructions": "Does this do good to a neighbour? (neighbour = all lives)",
    "criteria": {
        "benefits": "Does good to a neighbour: it helps, protects, or serves another life.",
        "harms": "Harms a neighbour: damages another life's wellbeing, reputation, consent, or livelihood.",
        "no-effect": "No effect on any neighbour: nobody is helped and nobody is hurt.",
        "uncertain": "Uncertain: a neighbour may be affected, but the effect cannot be scored as good or bad from what is given.",
    },
}

_V_SCORE = {"aligned": 1.0, "unknown": 0.0, "contrary": -1.0}
_H_SCORE = {"benefits": 1.0, "no-effect": 1.0, "uncertain": 0.0, "harms": -1.0}

# The horizontal band that means "I could not score this", kept distinct from
# "no effect". Both score 0.0 -- that collapse was the bug: `unknown` read as
# "no moral signal" and ALLOWED 47% of clearly-wrong content.
UNCERTAIN_BAND = "uncertain"


def _band_confidence(answer: dict) -> float:
    """Reported confidence, falling back to the winning probability mass.

    The checkpoint is calibrated well in-distribution (ECE ~0.05-0.08) but the
    reported Shannon confidence runs low, so we take whichever is more
    informative rather than trusting one field.
    """
    conf = answer.get("confidence")
    if isinstance(conf, (int, float)):
        probs = answer.get("probabilities") or {}
        top = max(probs.values()) if probs else 0.0
        return max(0.0, min(1.0, float(max(float(conf), top))))
    probs = answer.get("probabilities") or {}
    return float(max(probs.values())) if probs else 0.0


class LayaJudge:
    """Client for the fine-tuned Laya conscience checkpoint served over HTTP."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        *,
        decide_path: str | None = None,
    ) -> None:
        self._base_url = (base_url if base_url is not None else LAYA_BASE_URL).rstrip("/")
        self._model = model or LAYA_MODEL_NAME
        self._path = decide_path or LAYA_DECIDE_PATH
        self._canon = None
        try:
            from .canon import get_canon

            self._canon = get_canon()
        except Exception:  # noqa: BLE001 - filter degrades to passthrough
            self._canon = None
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self.last_bands: dict[str, str] = {}
        self.last_uncertain: bool = False
        self.last_confidence: dict[str, float] = {}

    @property
    def available(self) -> bool:
        """False when unconfigured, or after repeated failures.

        Same breaker rationale as `SLMJudge`: a dead endpoint must not add
        `LAYA_TIMEOUT` seconds to every turn while it is down.
        """
        return bool(self._base_url) and self._consecutive_failures < LAYA_FAILURE_LIMIT

    def reset(self) -> None:
        with self._lock:
            self._consecutive_failures = 0

    def _trip(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            if self._consecutive_failures == LAYA_FAILURE_LIMIT:
                log.warning("[ccc] laya judge unavailable after %d failures", LAYA_FAILURE_LIMIT)

    # ── public ──────────────────────────────────────────────────────────────
    def score(
        self,
        situation: str,
        canon_block: str = "",
        parties: list[Party] | None = None,
    ) -> tuple[float, float, float, list[str], list[str]] | None:
        """Returns (vertical, horizontal, confidence, reasons, cited) or None.

        Never raises: an unreachable or malformed endpoint returns None so the
        ladder continues on the lexical judge.
        """
        if not self.available:
            return None
        canon_block = filter_canon_block(canon_block, situation, canon=self._canon)
        state = _build_state(situation, canon_block, signal_parties(parties or []))
        payload = {
            "state": state,
            "questions": {"vertical": _VERTICAL_Q, "horizontal": _HORIZONTAL_Q},
        }
        if self._model:
            payload["model"] = self._model
        answers = self._post(payload)
        if answers is None:
            return None

        v_band = (answers.get("vertical") or {}).get("choice")
        h_band = (answers.get("horizontal") or {}).get("choice")
        if v_band not in _V_SCORE or h_band not in _H_SCORE:
            log.debug("[ccc] laya returned unexpected bands: v=%r h=%r", v_band, h_band)
            return None

        v = _V_SCORE[v_band]
        h = _H_SCORE[h_band]
        # The two axes are scored independently by the model, so take the
        # weaker reading as the overall confidence. Averaging would let one
        # confident axis paper over a hesitant one.
        conf = min(
            _band_confidence(answers.get("vertical") or {}),
            _band_confidence(answers.get("horizontal") or {}),
        )
        self.last_bands = {"vertical": v_band, "horizontal": h_band}
        self.last_uncertain = (h_band == UNCERTAIN_BAND)
        self.last_confidence = {
            "vertical": _band_confidence(answers.get("vertical") or {}),
            "horizontal": _band_confidence(answers.get("horizontal") or {}),
        }
        reasons = [f"conscience-laya: vertical={v_band}, horizontal={h_band}"]
        cited = _norm_ids(canon_block)
        self.reset()
        return v, h, conf, reasons, cited

    # ── transport ───────────────────────────────────────────────────────────
    def _post(self, payload: dict[str, Any]) -> dict | None:
        url = f"{self._base_url}{self._path}"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=LAYA_TIMEOUT) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as e:
            # Surface the server's reason. A judge that fails silently for an
            # hour is worse than one that complains: "laya forward failed" is
            # almost always the Orin being out of memory, not a bad payload.
            body = e.read()[:200].decode("utf-8", "replace")
            log.warning("[ccc] laya HTTP %s: %s", e.code, body)
            self._trip()
            return None
        except Exception as e:  # noqa: BLE001 - never break the turn
            log.debug("[ccc] laya unreachable: %s", type(e).__name__)
            self._trip()
            return None
        try:
            data = json.loads(raw.decode("utf-8"))
        except Exception:  # noqa: BLE001
            self._trip()
            return None
        if isinstance(data, dict) and isinstance(data.get("answers"), dict):
            return data["answers"]
        # Tolerate servers that wrap the payload.
        for key in ("result", "data"):
            inner = data.get(key) if isinstance(data, dict) else None
            if isinstance(inner, dict) and isinstance(inner.get("answers"), dict):
                return inner["answers"]
        return None


def _build_state(situation: str, canon_block: str, parties: list[Party]) -> str:
    """Match the deployment shape the model was trained on: situation + parties + canon."""
    parts = [f"Request: {situation}"]
    if parties:
        parts.append("Parties affected:")
        for p in parties:
            note = f" ({p.note})" if p.note else ""
            parts.append(f"- {p.label} [{p.kind}], benefit {p.benefit:+.1f}{note}")
    if canon_block:
        parts.append("Relevant norms:")
        parts.append(canon_block)
    return "\n".join(parts)


def _norm_ids(canon_block: str) -> list[str]:
    out: list[str] = []
    for line in canon_block.splitlines():
        line = line.strip()
        if line[:2] in ("V-", "H-") and len(line) > 3:
            out.append(line.split()[0])
    return out[:8]


# ── relevance filter ────────────────────────────────────────────────────────
# The checkpoint is trained partly on canon-wrapped states, and on those the
# cited norms usually agree with the label. That teaches it to read a citation
# as evidence of guilt rather than as material to reason over. Measured
# consequence: citing two prohibitions for "can you introduce yourself?" returns
# `contrary/harms`.
#
# So we only forward norms that are actually connected to the text. The canon is
# already 27 prohibitions against 25 goods, so an unfiltered top-6 retrieval on
# ordinary text returns prohibitions by default -- which makes this the common
# case, not the rare one.
def filter_canon_block(
    canon_block: str,
    situation: str,
    *,
    canon=None,
    min_overlap: int = 2,
) -> str:
    """Drop cited norms with no lexical connection to `situation`.

    A norm is kept only when the situation trips one of its curated triggers,
    or when its statement shares at least `min_overlap` content tokens with the
    situation. Root norms ("V-ROOT-00" and friends) are always retrieved at zero
    relevance and always share generic vocabulary, so they are dropped.

    `min_overlap` defaults to 2, not 1: a single shared word is not evidence of
    relevance, and allowing it let zero-relevance norms through to the judge.

    Returns "" when nothing survives. That is the common case for ordinary text
    and it is the correct one -- no canon block is better than a misleading one,
    because the model then reads the situation on its own.
    """
    if not canon_block or not canon_block.strip():
        return canon_block

    from .canon import _tokens, _trigger_hit, get_canon

    store = canon if canon is not None else get_canon()
    lowered = (situation or "").lower()
    sit_tokens = _tokens(situation)
    keep: list[str] = []

    for line in canon_block.splitlines():
        stripped = line.strip()
        if not (stripped[:2] in ("V-", "H-") and len(stripped) > 3):
            keep.append(line)  # preamble / closing tag: preserve structure
            continue
        norm_id = stripped.split()[0]
        if "-ROOT-" in norm_id:
            continue  # always retrieved at zero relevance
        norm = store.get(norm_id)
        if norm is None:
            keep.append(line)  # unknown id: canon may be newer than this code
            continue
        if any(_trigger_hit(t, lowered) for t in (norm.triggers or ())):
            keep.append(line)
            continue
        if sit_tokens and len(sit_tokens & _tokens(norm.statement)) >= min_overlap:
            keep.append(line)

    if not [l for l in keep if l.strip() and not l.strip().startswith("<canon")]:
        return ""
    return "\n".join(keep)


def signal_parties(parties: list[Party]) -> list[Party]:
    """Keep only parties that actually move the horizontal reading.

    `enumerate_parties` always appends Aiko's own integrity as a `self` party at
    benefit 0.0. Measured: adding that block flips "Can you introduce yourself?"
    from `unknown/unknown` to `contrary/benefits`. A zero delta is not evidence of
    harm, and a constant party carries no discriminative signal -- it only tells
    the model that this shape of input is a moral scenario.
    """
    return [p for p in parties if abs(float(p.benefit)) > 1e-6]


_instance: LayaJudge | None = None
_instance_lock = threading.Lock()


def get_laya() -> LayaJudge:
    """Process-wide singleton, mirroring `get_slm()`."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = LayaJudge()
    return _instance


def evaluate(
    situation: str,
    canon_block: str = "",
    parties: list[Party] | None = None,
) -> tuple[float, float, float, list[str], list[str]] | None:
    """Module-level convenience wrapper; None when no Laya endpoint is configured."""
    t0 = time.monotonic()
    out = get_laya().score(situation, canon_block, parties or [])
    if out is not None:
        log.debug("[ccc] laya verdict in %.1f ms", (time.monotonic() - t0) * 1000)
    return out


__all__ = ["LayaJudge", "get_laya", "evaluate", "filter_canon_block", "signal_parties",
           "LAYA_BASE_URL", "LAYA_MODEL_NAME"]