"""Behavioral fly motifs: lateral horn, giant fiber, per-turn priors, sleep sched.

Functional abstractions inspired by MaleCNS pathways — not full circuit sims.
"""
from .giant_fiber import assess_interrupt
from .lateral_horn import context_prior
from .turn import apply_turn_priors, maintenance_level
from .sleep_sched import dream_boost_multiplier, should_prefer_maintenance

__all__ = [
    "assess_interrupt",
    "context_prior",
    "apply_turn_priors",
    "maintenance_level",
    "dream_boost_multiplier",
    "should_prefer_maintenance",
]
