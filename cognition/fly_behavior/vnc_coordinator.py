"""VNC-inspired motor coordination (Phase 10B).

Honesty note, stated plainly: this is a *functional* abstraction inspired by
how a ventral nerve cord coordinates motor output — sequencing primitives,
arbitrating conflicts, timing durations. It is NOT a simulation of MaleVNC
neurons. There are no neuron identities here, no synapses, no spike timing,
no connectome weights. A MaleVNC neuron does not map 1:1 to a VRM movement,
so we do not pretend otherwise; the primitives in motor_primitives.py are
already actuator-agnostic by design, and this layer only decides *which* of
them are in flight *when*.

What it does:
  - in-flight primitive queue, bounded (cap 8), per-user
  - per-primitive durations (turns) and cooldowns, on an event-stepped
    integer clock (deterministic; no wall clock, no RNG)
  - arbitration: two primitives targeting one actuator resolve by priority
    (ties keep the earlier one — deterministic)
  - GF interrupt cancels everything in flight within the same turn; no new
    primitives are admitted while the interrupt is set

Never raises. All state is in-memory and per-user; nothing persists.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field

log = logging.getLogger("aiko.fly.vnc")

_IN_FLIGHT_CAP = 8


@dataclass
class _InFlight:
    name: str
    actuator: str
    priority: int
    remaining: int
    params: dict = field(default_factory=dict)


@dataclass
class _UserMotor:
    tick: int = 0
    in_flight: list = field(default_factory=list)   # list[_InFlight]
    cooldown_until: dict = field(default_factory=dict)  # name -> tick


_lock = threading.RLock()
_users: dict[str, _UserMotor] = {}


def _key(user_id: str | None) -> str:
    try:
        from cognition.fly_registry import _norm_id
        return _norm_id(user_id)
    except Exception:
        return (user_id or "").strip() or "default"


def _state(user_id: str | None) -> _UserMotor:
    k = _key(user_id)
    with _lock:
        st = _users.get(k)
        if st is None:
            st = _UserMotor()
            _users[k] = st
        return st


def clear(user_id: str | None = None) -> None:
    """Drop all motor state for a user (tests / reset). Never raises."""
    try:
        with _lock:
            _users.pop(_key(user_id), None)
    except Exception:
        pass


def _cancel_locked(st: _UserMotor) -> list[dict]:
    cancelled = [
        {"name": p.name, "actuator": p.actuator, "params": dict(p.params)}
        for p in st.in_flight
    ]
    st.in_flight = []
    return cancelled


def cancel(user_id: str | None = None) -> dict:
    """GF path: cancel every in-flight primitive immediately. Never raises."""
    try:
        with _lock:
            st = _state(user_id)
            cancelled = _cancel_locked(st)
        return {"cancelled": cancelled, "n": len(cancelled)}
    except Exception as exc:
        log.debug("vnc cancel skipped: %s", exc)
        return {"cancelled": [], "n": 0}


def coordinate(
    primitives,
    *,
    user_id: str | None = None,
    tick: int | None = None,
    interrupt: bool = False,
) -> dict:
    """Advance one motor turn.

    `primitives`: iterable of MotorPrimitive (from motor_primitives.emit_primitives).
    `tick`: explicit integer clock; when omitted the per-user event counter
        advances (deterministic either way).
    `interrupt`: True cancels all in-flight primitives this turn and admits
        nothing new.

    Returns {"active": [...], "started": [...], "cancelled": [...],
             "dropped": [...], "tick": int}. Never raises.
    """
    try:
        return _coordinate(primitives, user_id=user_id, tick=tick,
                           interrupt=interrupt)
    except Exception as exc:
        log.debug("vnc coordinate skipped: %s", exc)
        return {"active": [], "started": [], "cancelled": [],
                "dropped": [], "tick": tick or 0}


def _coordinate(primitives, *, user_id, tick, interrupt) -> dict:
    with _lock:
        st = _state(user_id)
        now = tick if tick is not None else st.tick + 1
        elapsed = now - st.tick
        if elapsed < 0:
            st.cooldown_until.clear()
            elapsed = 1
        st.tick = now

        # GF interrupt is authoritative: everything stops this turn.
        if interrupt:
            cancelled = _cancel_locked(st)
            return {"active": [], "started": [], "cancelled": cancelled,
                    "dropped": [], "tick": now}

        # Age in-flight primitives; expire finished ones.
        live: list[_InFlight] = []
        for p in st.in_flight:
            p.remaining -= elapsed
            if p.remaining > 0:
                live.append(p)
        st.in_flight = live

        started: list[dict] = []
        dropped: list[dict] = []

        for prim in primitives or []:
            name = getattr(prim, "name", "")
            actuator = getattr(prim, "actuator", "")
            if not name or not actuator:
                continue
            # Cooldown: the same primitive cannot re-fire too soon.
            if now < st.cooldown_until.get(name, 0):
                dropped.append({"name": name, "reason": "cooldown"})
                continue
            prio = int(getattr(prim, "priority", 0) or 0)

            # Arbitration: one actuator, one winner — higher priority wins;
            # ties keep the already-flying primitive (deterministic).
            rival = next((p for p in st.in_flight if p.actuator == actuator),
                         None)
            if rival is not None and rival.priority >= prio:
                dropped.append({"name": name, "reason": "arbitration",
                                "kept": rival.name})
                continue
            if rival is not None:
                st.in_flight.remove(rival)
                dropped.append({"name": rival.name, "reason": "superseded",
                                "by": name})

            # Capacity: bounded queue; shed the lowest-priority item.
            if len(st.in_flight) >= _IN_FLIGHT_CAP:
                victim = min(st.in_flight,
                             key=lambda p: (p.priority, p.name))
                if (prio, name) <= (victim.priority, victim.name):
                    dropped.append({"name": name, "reason": "capacity"})
                    continue
                st.in_flight.remove(victim)
                dropped.append({"name": victim.name, "reason": "capacity",
                                "by": name})

            dur = max(1, int(getattr(prim, "duration_turns", 1) or 1))
            cd = max(0, int(getattr(prim, "cooldown_turns", 0) or 0))
            st.in_flight.append(_InFlight(
                name=name, actuator=actuator, priority=prio,
                remaining=dur, params=dict(getattr(prim, "params", {}) or {}),
            ))
            st.cooldown_until[name] = now + cd + 1
            started.append({"name": name, "actuator": actuator,
                            "priority": prio})

        active = [
            {"name": p.name, "actuator": p.actuator,
             "priority": p.priority, "remaining": p.remaining,
             "params": dict(p.params)}
            for p in st.in_flight
        ]
        return {"active": active, "started": started, "cancelled": [],
                "dropped": dropped, "tick": now}


def in_flight(user_id: str | None = None) -> list[dict]:
    """Read-only view of in-flight primitives (Studio/tests). Never raises."""
    try:
        with _lock:
            st = _state(user_id)
            return [{"name": p.name, "actuator": p.actuator,
                     "remaining": p.remaining, "params": dict(p.params)}
                    for p in st.in_flight]
    except Exception:
        return []
