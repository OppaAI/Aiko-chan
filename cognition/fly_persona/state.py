"""Phase 11 — Neural Personality State: per-identity persistent traits.

This is the ONLY state the persona-conditioning machinery owns. It is
deliberately separate from persona/SOUL.md, which this module never reads
as a writable config and never modifies.

Traits are floats in [0, 1]:
  curiosity, playfulness, attachment, calmness, exploration

Defaults are documented derivations from persona/SOUL.md's Personality
section (prose — there is no machine-readable trait config there, so the
derivation is recorded here instead of pretended):
  - "observant", "honest and direct", "has opinions"  → curiosity 0.70
  - "playful, occasionally teasing"                   → playfulness 0.70
  - "warm and familiar", "caring through attention"   → attachment 0.70
  - "calm"                                           → calmness 0.65
  - exploration: no direct line in SOUL.md; neutral 0.60

A future machine-readable persona config can feed in through the
AIKO_FLY_PERSONA_DEFAULTS JSON env override
(e.g. '{"curiosity": 0.8, "playfulness": 0.6}').

Drift clamp: a trait may never leave [default-0.30, default+0.30] ∩ [0, 1].
The ONLY writer of trait values is cognition.fly_persona.plasticity
(dopamine/eligibility events). Everything else reads.

Fail-soft: never raises. Persistence is best-effort JSON.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

log = logging.getLogger("aiko.fly.persona.state")

TRAITS: tuple[str, ...] = (
    "curiosity",
    "playfulness",
    "attachment",
    "calmness",
    "exploration",
)

# Documented SOUL.md derivations (see module docstring).
_SOUL_DEFAULTS: dict[str, float] = {
    "curiosity": 0.70,
    "playfulness": 0.70,
    "attachment": 0.70,
    "calmness": 0.65,
    "exploration": 0.60,
}

DRIFT_LIMIT = 0.30  # max |trait - default|


def persona_defaults() -> dict[str, float]:
    """Trait defaults: SOUL.md derivations, optionally overridden by env JSON."""
    d = dict(_SOUL_DEFAULTS)
    raw = (os.getenv("AIKO_FLY_PERSONA_DEFAULTS") or "").strip()
    if raw:
        try:
            ov = json.loads(raw)
            if isinstance(ov, dict):
                for k in TRAITS:
                    if k in ov:
                        d[k] = max(0.0, min(1.0, float(ov[k])))
        except Exception as exc:
            log.debug("persona defaults override skipped: %s", exc)
    return d


def _data_dir() -> Path:
    root = os.getenv("AIKO_FLY_PERSONA_DIR", "").strip()
    if root:
        return Path(root).expanduser()
    return Path.home() / ".aiko" / "data" / "fly_persona"


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        safe = "".join(
            c if (c.isalnum() or c in ("-", "_")) else "_"
            for c in (user_id or "").strip() or "default"
        )
        return safe[:64] or "default"


_lock = threading.RLock()
_states: dict[str, "NeuralPersonalityState"] = {}


class NeuralPersonalityState:
    """Per-identity trait vector. Mutation only via plasticity._apply_delta."""

    def __init__(self, user_id: str | None = None) -> None:
        self.user_id = user_id
        self._key = _key(user_id)
        self.defaults = persona_defaults()
        self._traits = dict(self.defaults)
        self._event_clock = 0  # credit events seen (plasticity cooldown)
        self._last_update_event = -10**9
        self._load()

    # ── persistence ────────────────────────────────────────────────
    def _path(self) -> Path:
        return _data_dir() / f"{self._key}.json"

    def _load(self) -> None:
        try:
            p = self._path()
            if not p.is_file():
                return
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return
            saved = raw.get("traits") or {}
            for k in TRAITS:
                if k in saved:
                    try:
                        self._traits[k] = self._clamp(k, float(saved[k]))
                    except Exception:
                        pass
            try:
                self._event_clock = int(raw.get("event_clock", 0))
            except Exception:
                pass
        except Exception as exc:
            log.debug("persona load skipped: %s", exc)

    def save(self) -> bool:
        try:
            p = self._path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    {
                        "user_id": self.user_id,
                        "traits": {k: round(v, 4) for k, v in self._traits.items()},
                        "defaults": {k: round(v, 4) for k, v in self.defaults.items()},
                        "event_clock": self._event_clock,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            return True
        except Exception as exc:
            log.debug("persona save skipped: %s", exc)
            return False

    # ── access ─────────────────────────────────────────────────────
    @property
    def traits(self) -> dict[str, float]:
        """A copy. External code cannot mutate state through this."""
        with _lock:
            return dict(self._traits)

    def _clamp(self, trait: str, value: float) -> float:
        d = self.defaults.get(trait, 0.5)
        lo = max(0.0, d - DRIFT_LIMIT)
        hi = min(1.0, d + DRIFT_LIMIT)
        return max(lo, min(hi, float(value)))

    def describe(self) -> dict:
        """Inspectable, JSON-serializable snapshot (future Studio data)."""
        with _lock:
            return {
                "user_id": self.user_id,
                "traits": {k: round(v, 4) for k, v in self._traits.items()},
                "defaults": {k: round(v, 4) for k, v in self.defaults.items()},
                "drift_limit": DRIFT_LIMIT,
                "bounds": {
                    k: [round(max(0.0, self.defaults[k] - DRIFT_LIMIT), 4),
                        round(min(1.0, self.defaults[k] + DRIFT_LIMIT), 4)]
                    for k in TRAITS
                },
            }

    def reset(self) -> dict:
        """Restore defaults. The only sanctioned non-plasticity mutation."""
        with _lock:
            self._traits = dict(self.defaults)
            self._last_update_event = -10**9
            self.save()
            return self.describe()


def get_personality(user_id: str | None = None) -> NeuralPersonalityState:
    key = _key(user_id)
    with _lock:
        st = _states.get(key)
        if st is None:
            st = NeuralPersonalityState(user_id)
            _states[key] = st
        return st


def clear_personality(user_id: str | None = None) -> None:
    """Drop the in-memory state (tests / identity switch). Not a reset."""
    with _lock:
        _states.pop(_key(user_id), None)
