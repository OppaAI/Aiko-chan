"""Phase 9 — FlyWorld: closed-loop world simulator for the fly brain.

Aiko acts -> environment responds -> fly brain learns -> Aiko changes.

Submodules:
  sim            deterministic, seeded model of the assistant environment
                 (requests, actions, outcomes, rewards)
  loop           the closed loop: Phase-5 action selection -> sim -> dopamine
                 + sim-local eligibility -> learning, all tagged simulated
  replay_bridge  sim episodes -> Phase 8 replay machinery (opt-in)

Everything here is synthetic. The simulator NEVER performs real tool
calls, sends real messages, writes real memory, or touches the network —
only learned policy changes (plastic weights) transfer out. A test asserts
the import boundary statically and at runtime.
"""
from __future__ import annotations

from .loop import (
    evaluate_policy,
    flyworld_mode,
    last_episode,
    clear_episodes,
    run_episode,
)
from . import sim, loop
from . import replay_bridge

__all__ = [
    "sim",
    "loop",
    "replay_bridge",
    "flyworld_mode",
    "run_episode",
    "evaluate_policy",
    "last_episode",
    "clear_episodes",
]
