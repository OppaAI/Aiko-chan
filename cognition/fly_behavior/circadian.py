"""Real circadian clock circuit (Phase 4).

Replaces the raw wall-clock fraction previously published as "circadian".

Biology: the fly clock network (DN1/DN2/DN3, LNd, s-LNv/l-LNv, LPN —
MaleCNS v1.0 types) is entrained by light (zeitgeber) and drives
sleep/wake timing. Aiko mapping: wall clock is the zeitgeber that entrains
the oscillator; the *gain* of the rhythm comes from real clock-neuron drive
measured in the whole-brain step. High clock drive -> sharp day/night
signal; low drive -> flat rhythm (the clock is "asleep").

What is real vs synthetic:
  REAL: clock-neuron drive from the MaleCNS v1.0 full connectome.
  SYNTHETIC: the 24h oscillator abstraction, entrainment rule, nightness
    curve. The phase is entrained to wall clock — that is the honest
    zeitgeber, not a simulated free-run.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

try:
    from system.log import get_logger as _get_logger
    log = _get_logger(__name__)
except Exception:
    import logging as _logging
    log = _logging.getLogger("aiko.circadian")

# Deep-night centre ~02:00 local as a fraction of the day.
NIGHT_CENTER = 2.0 / 24.0

_lock = threading.RLock()
_circuit: "ClockCircuit | None" = None


@dataclass
class ClockCircuit:
    """24h oscillator: phase entrained by wall clock, gain by clock neurons."""

    phase: float = field(init=False, default=0.5)
    amplitude: float = field(init=False, default=0.5)
    clock_drive_ema: float = field(init=False, default=0.0)

    def tick(self, wall_phase: float, clock_drive: float | None = None) -> dict:
        """Advance one turn. wall_phase in [0,1); clock_drive >= 0 from FullBrain."""
        self.phase = max(0.0, min(1.0, float(wall_phase)))
        if clock_drive is not None:
            d = max(0.0, float(clock_drive))
            # Clock drive from the whole-brain step is O(1e-5) (measured
            # 2e-6..3.2e-5 across inputs); normalise against 2e-5 so typical
            # drive maps to ~1. EMA adapts slowly across turns.
            self.clock_drive_ema += 0.2 * (min(d / 2e-5, 2.0) - self.clock_drive_ema)
        gain = max(0.0, min(1.0, self.clock_drive_ema / 2.0))
        self.amplitude = 0.25 + 0.75 * gain
        nightness = 0.5 * (1.0 + math.cos(2.0 * math.pi * (self.phase - NIGHT_CENTER)))
        return {
            "phase": self.phase,
            "amplitude": round(self.amplitude, 3),
            "nightness": round(nightness * self.amplitude, 3),
            "clock_gain": round(gain, 3),
        }


def get_circuit() -> ClockCircuit:
    global _circuit
    with _lock:
        if _circuit is None:
            _circuit = ClockCircuit()
        return _circuit


def circadian_now(wall_phase: float, *, user_id: str | None = None,
                  clock_drive: float | None = None) -> dict:
    """Tick the clock circuit and publish to NeuralState. Never raises."""
    out = {"phase": wall_phase, "amplitude": 0.5, "nightness": 0.0, "clock_gain": 0.0}
    try:
        tick = get_circuit().tick(wall_phase, clock_drive)
        out.update(tick)
        try:
            from cognition.neural_state import get_neural_state
            st = get_neural_state(user_id)
            st.publish_circadian(tick["phase"], source="clock-circuit")
        except Exception as exc:
            log.debug("circadian publish skipped: %s", exc)
    except Exception as exc:
        log.debug("circadian tick skipped: %s", exc)
    return out
