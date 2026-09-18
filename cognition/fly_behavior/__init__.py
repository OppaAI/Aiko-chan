"""Behavioral fly motifs: lateral horn (context prior) + giant fiber (interrupt).

These are functional abstractions inspired by MaleCNS pathways, not full
circuit simulations. Identity isolation for LH session state is per user_id;
GF is stateless per call.
"""
from .giant_fiber import assess_interrupt
from .lateral_horn import context_prior

__all__ = ["assess_interrupt", "context_prior"]
