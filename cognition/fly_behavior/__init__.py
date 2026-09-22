"""Behavioral fly motifs: lateral horn, giant fiber, per-turn priors, sleep sched.

Functional abstractions inspired by MaleCNS pathways — not full circuit sims.
Stage 1: multi-source GF + should_abort_plan for agent loops.
"""
from .giant_fiber import assess_interrupt, should_abort_plan
from .lateral_horn import context_prior
from .turn import apply_turn_priors, maintenance_level
from .sleep_sched import dream_boost_multiplier, should_prefer_maintenance

__all__ = [
    "assess_interrupt",
    "should_abort_plan",
    "context_prior",
    "apply_turn_priors",
    "maintenance_level",
    "dream_boost_multiplier",
    "should_prefer_maintenance",
]
