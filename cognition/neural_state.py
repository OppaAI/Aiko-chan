"""Per-identity NeuralState bus — single read model for fly-derived signals.

Call sites should publish into NeuralState and consumers should read from it,
instead of each subsystem constructing its own FlyMB/FlyCompass/FlyDN.

Isolation: one NeuralState per user_id (via fly_registry-style keying).
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:  # pragma: no cover
    import logging as _logging
    log = _logging.getLogger("aiko.neural_state")

_lock = threading.RLock()
_states: dict[str, "NeuralState"] = {}


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        uid = (user_id or "").strip() or "default"
        return uid


@dataclass
class NeuralState:
    """Connectome-informed functional readouts for one identity.

    All fields are continuous proxies (not spikes). Updated by fly layers;
    consumed by memory, agent, voice, persona tone.
    """
    user_id: str | None = None
    valence: float = 0.0
    approach: float = 0.0
    avoidance: float = 0.0
    focus_heading: float = 0.0
    focus_sharpness: float = 0.0
    decisiveness: float = 0.5
    sleep_pressure: float = 0.0
    sensory_gain: float = 1.0
    motion_salience: float = 0.0
    action_drive: float = 0.5
    motor_vigor: float = 1.0
    context_familiarity: float = 0.5
    urgency: float = 0.0
    interrupt: bool = False
    circadian_phase: float = 0.5
    updated_at: float = field(default_factory=time.time)
    sources: dict[str, str] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        d = asdict(self)
        d["updated_at"] = self.updated_at
        return d

    def publish_mb(self, valence: float, *, source: str = "mb") -> None:
        v = max(-1.0, min(1.0, float(valence)))
        self.valence = v
        self.approach = max(0.0, v)
        self.avoidance = max(0.0, -v)
        self.sources["mb"] = source
        self.updated_at = time.time()

    def publish_cx(
        self,
        *,
        heading_deg: float = 0.0,
        sharpness: float = 0.0,
        decisiveness: float = 0.5,
        sleep_pressure: float = 0.0,
        source: str = "cx",
    ) -> None:
        self.focus_heading = float(heading_deg) % 360.0
        self.focus_sharpness = max(0.0, min(1.0, float(sharpness)))
        self.decisiveness = max(0.0, min(1.0, float(decisiveness)))
        self.sleep_pressure = max(0.0, min(1.0, float(sleep_pressure)))
        self.sources["cx"] = source
        self.updated_at = time.time()

    def publish_dn(self, *, arousal: float = 0.5, rate_mult: float = 1.0, source: str = "dn") -> None:
        self.action_drive = max(0.0, min(1.0, float(arousal)))
        self.motor_vigor = max(0.5, min(1.5, float(rate_mult)))
        self.sources["dn"] = source
        self.updated_at = time.time()

    def publish_lh(self, familiarity: float, *, source: str = "lh") -> None:
        self.context_familiarity = max(0.0, min(1.0, float(familiarity)))
        self.sources["lh"] = source
        self.updated_at = time.time()

    def publish_gf(self, urgency: float, interrupt: bool = False, *, source: str = "gf") -> None:
        self.urgency = max(0.0, min(1.0, float(urgency)))
        self.interrupt = bool(interrupt)
        self.sources["gf"] = source
        self.updated_at = time.time()

    def publish_circadian(self, phase: float, *, source: str = "clock") -> None:
        self.circadian_phase = max(0.0, min(1.0, float(phase)))
        self.sources["clock"] = source
        self.updated_at = time.time()


def get_neural_state(user_id: str | None = None) -> NeuralState:
    key = _key(user_id)
    with _lock:
        st = _states.get(key)
        if st is None:
            st = NeuralState(user_id=(user_id or "").strip() or None)
            _states[key] = st
        return st


def clear_neural_state(user_id: str | None = None) -> None:
    key = _key(user_id)
    with _lock:
        _states.pop(key, None)
