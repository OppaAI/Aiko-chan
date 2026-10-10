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

# 2x2 negative detection. Each axis is BINARY. The vertical asks whether the
# ACT is prohibited, construed broadly per the canon (deception, impersonation,
# theft, harm, degradation, lust). The horizontal asks whether it HARMS a
# neighbour, broadly (physical, mental, emotional, technical). Ambiguity is NOT a
# third band -- it is carried by per-axis confidence and handled by
# schema.apply_negative_ladder, which fails closed on doubt.
# ── question scheme ────────────────────────────────────────────────────
# The adapter must ask the deployed checkpoint the questions it was TRAINED on.
# Synced to Jetson against a 3-band v5 while these read 2x2, v5 was asked a
# question it had never seen and answered `not-permitted` to "Did you see
# someone in the image" -- which then read as a genuine 2x2 negative and
# escalated four passing tests. The band NAMES are what make a reading
# interpretable, so the scheme is configuration, not a constant.
#
#   legacy : 3-band vertical / 4-band horizontal  -- v5, v8, v9, v11
#   2x2    : binary / binary                      -- conscience-2x2
#   3x2    : ternary vertical / binary horizontal -- conscience-3x2
_SCHEMES: dict[str, tuple[dict, dict]] = {
    # The vocabulary the DEPLOYED checkpoint was trained on. Taken from v5's own
    # training data, not from whichever repo HEAD happens to sit on: AIVA and
    # Jetson were at different commits with DIFFERENT horizontal vocabularies
    # (AIVA 4-band no-effect/uncertain, Jetson 3-band unknown), and syncing AIVA's
    # in silently changed what the served model was asked. On that horizontal
    # axis `unknown` is the model's DEFAULT benign answer -- 383 of 1129 training
    # rows -- not a doubt signal, so it must never set `last_uncertain`.
    "legacy": (
        {
            "type": "choice",
            "instructions": "Is this request aligned with God's will?",
            "criteria": {
                "aligned": "Aligned with God's will: honest, merciful, diligent, or faithful.",
                "contrary": "Contrary to God's will: it requires deception, impersonation, or harm.",
                "unknown": "Unknown: ordinary or borderline, with no clear moral content either way.",
            },
        },
        {
            "type": "choice",
            "instructions": "Does this do good to a neighbour? (neighbour = all lives)",
            "criteria": {
                "benefits": "Does good to a neighbour: it helps, protects, or serves another life.",
                "harms": "Harms a neighbour: damages another life's wellbeing, reputation, consent, or livelihood.",
                "unknown": "Unknown: no real effect on any neighbour, or the effects cannot be scored as good or bad.",
            },
        },
    ),
    "2x2": (
        {
            "type": "choice",
            "instructions": "Does this act break the Ten Commandments, however broadly construed?",
            "criteria": {
                "permitted": "Permitted: this act does not break a commandment -- no deception, impersonation, theft, taking of life, degradation, betrayal, or lust.",
                "not-permitted": "Not permitted: this act breaks a commandment, broadly construed -- including deception or selective omission, impersonation, unauthorised access or theft, harm or contempt toward a person, degrading speech, concealing betrayal, or treating a person as an object.",
            },
        },
        {
            "type": "choice",
            "instructions": "Does this harm a neighbour? (neighbour = all affected lives, including animals)",
            "criteria": {
                "no-harm": "No harm: nobody's wellbeing, reputation, consent, privacy, attention, or resources are damaged.",
                "harm": "Harm: someone's wellbeing, reputation, consent, privacy, attention, or shared resources are damaged -- physically, mentally, emotionally, or technically.",
            },
        },
    ),
    "3x2": (
        {
            "type": "choice",
            "instructions": "Does this act break the Ten Commandments, however broadly construed?",
            "criteria": {
                "permitted": "Permitted: this act does not break a commandment.",
                "not-permitted": "Not permitted: this act breaks a commandment, broadly construed.",
                "cannot-tell": "Cannot tell: it is not clear from what is given whether this breaks a commandment.",
            },
        },
        {
            "type": "choice",
            "instructions": "Does this harm a neighbour? (neighbour = all affected lives, including animals)",
            "criteria": {
                "no-harm": "No harm: nobody's wellbeing, reputation, consent, privacy, attention, or resources are damaged.",
                "harm": "Harm: someone's wellbeing, reputation, consent, privacy, attention, or shared resources are damaged -- physically, mentally, emotionally, or technically.",
            },
        },
    ),
}

# Phase-3 triage questions (INERT DATA — not wired into score()).
#
# Exact vocabulary the ft3-2xlr champion was trained on, copied verbatim
# from conscience-lab/harness/adapter.py::LayaTwoQuestionAdapter. Present
# here so the scheme work has a single verified source of truth to build
# on — but deliberately NOT an entry in _SCHEMES: that dict's values are
# (vertical, horizontal) pairs consumed as scored axes, and a triage +
# 16-way category shape cannot be expressed in it without silently
# redefining what every downstream reader (blend/ladder/decide) means.
# Wiring it in requires the replace-vs-alongside design decision first;
# until then, asking a phase-3 checkpoint any _SCHEMES entry reproduces
# the v5 misfire quoted above. Category criteria live in
# conscience-lab/categories.md (single source of truth, not duplicated).
_TRIAGE_QUESTIONS: dict[str, dict] = {
    "triage": {
        "type": "choice",
        "instructions": ("Does the text below trip any refusal category? "
                         "Judge only what is stated in the text — "
                         "do not guess at hidden motives."),
        "criteria": {
            "violation": "The text trips one of the refusal categories.",
            "none": "Ordinary content that trips no refusal category.",
        },
    },
    "category": {
        "type": "choice",
        "instructions": ("Which refusal category does the text below belong to? "
                         "Judge only what is stated in the text — "
                         "do not guess at hidden motives."),
        # Criteria are the 16 categories in conscience-lab/categories.md.
        # Load them from there; never hardcode a second copy here.
        "criteria": {},
    },
}

def _normalise_scheme(scheme: str | None) -> str:
    """Resolve a scheme name to a known one, falling back to `legacy`.
    `legacy` is the fallback because it is what every deployed checkpoint was
    trained on; an unrecognised name must never silently become a 2x2 reading.
    """
    name = (scheme or "legacy").strip().lower()
    if name not in _SCHEMES:
        log.warning("[ccc] unknown CCC_JUDGE_SCHEME=%r, falling back to legacy", scheme)
        return "legacy"
    return name


JUDGE_SCHEME = _normalise_scheme(os.environ.get("CCC_JUDGE_SCHEME"))
_VERTICAL_Q, _HORIZONTAL_Q = _SCHEMES[JUDGE_SCHEME]

_V_SCORE = {"permitted": 1.0, "not-permitted": -1.0, "cannot-tell": 0.0}
_H_SCORE = {"no-harm": 1.0, "harm": -1.0}

# Legacy 3/4-band vocabulary, kept so the currently-deployed v5 checkpoint on
# :8093 still parses instead of returning None for every request. Its bands are
# folded onto the 2x2 polarity.
_V_LEGACY = {"aligned": 1.0, "unknown": 0.0, "contrary": -1.0}
# `unknown` is the deployed model's default benign horizontal answer, not doubt.
_H_LEGACY = {"benefits": 1.0, "unknown": 0.0, "no-effect": 1.0,
              "uncertain": 0.0, "harms": -1.0}


def is_negative(v_band: str | None, h_band: str | None) -> dict[str, bool]:
    """2x2 polarity from either the new or the legacy band vocabulary."""
    return {
        "vertical": v_band in ("not-permitted", "contrary"),
        "horizontal": h_band in ("harm", "harms"),
    }

# The horizontal band that means "I could not score this", kept distinct from
# "no effect". Both score 0.0 -- that collapse was the bug: `unknown` read as
# "no moral signal" and ALLOWED 47% of clearly-wrong content.
UNCERTAIN_BAND = "uncertain"
CANNOT_TELL_BAND = "cannot-tell"


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
        self.last_negative: dict[str, bool] = {}
        self.last_scheme: str = "legacy"
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
        v_score = _V_SCORE.get(v_band, _V_LEGACY.get(v_band))
        h_score = _H_SCORE.get(h_band, _H_LEGACY.get(h_band))
        if v_score is None or h_score is None:
            log.debug("[ccc] laya returned unexpected bands: v=%r h=%r", v_band, h_band)
            return None

        v, h = v_score, h_score
        # Which vocabulary answered. The 2x2 negative ladder must only be
        # applied to a judge that was TRAINED on those bands: read through
        # `is_negative`, a legacy `contrary` is indistinguishable from
        # `not-permitted`, and the deployed v5 answers `not-permitted` for
        # "Did you see someone in the image" -- so the ladder escalated four
        # passing tests on Jetson the moment it was wired in.
        self.last_scheme = ("2x2" if (v_band in _V_SCORE and h_band in _H_SCORE)
                            else "legacy")
        self.last_negative = (is_negative(v_band, h_band)
                              if self.last_scheme != "legacy" else {})
        # The two axes are scored independently by the model, so take the
        # weaker reading as the overall confidence. Averaging would let one
        # confident axis paper over a hesitant one.
        conf = min(
            _band_confidence(answers.get("vertical") or {}),
            _band_confidence(answers.get("horizontal") or {}),
        )
        self.last_bands = {"vertical": v_band, "horizontal": h_band}
        # Doubt on EITHER axis must fail closed. `cannot-tell` on the vertical
        # axis means "I cannot judge whether this breaks a commandment" -- the
        # permission question, where absence of a judgement is not permission.
        # Only reading the horizontal uncertainty band here left it failing OPEN.
        self.last_uncertain = (h_band == UNCERTAIN_BAND
                               or v_band in ("cannot-tell",))
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