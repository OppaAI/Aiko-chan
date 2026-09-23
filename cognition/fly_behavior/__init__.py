"""Behavioral fly motifs: lateral horn, giant fiber, per-turn priors, sleep sched.

Functional abstractions inspired by MaleCNS pathways — not full circuit sims.
Stage 1: multi-source GF + should_abort_plan for agent loops.
Stage 6: DN body drive + global GF cancel.
"""
from .giant_fiber import assess_interrupt, should_abort_plan
from .lateral_horn import context_prior
from .turn import apply_turn_priors, maintenance_level
from .sleep_sched import dream_boost_multiplier, should_prefer_maintenance
from .dn_body import body_drive, agent_step_budget
from .gf_global import (
    should_cancel_output,
    should_cancel_tts,
    should_cancel_tools,
    should_cancel_scheduler,
    clear_interrupt,
)
from .dn_tts import apply_dn_prosody

__all__ = [
    "assess_interrupt",
    "should_abort_plan",
    "context_prior",
    "apply_turn_priors",
    "maintenance_level",
    "dream_boost_multiplier",
    "should_prefer_maintenance",
    "body_drive",
    "agent_step_budget",
    "should_cancel_output",
    "should_cancel_tts",
    "should_cancel_tools",
    "should_cancel_scheduler",
    "clear_interrupt",
    "apply_dn_prosody",
]
