"""Aiko biological layer — fruit-fly mushroom-body associative memory.

Grafts a real MaleCNS v1.0 microcircuit (4,673 neurons, 121,706 synapses at
weight>=5) onto Aiko's valence system as opponent approach/avoid readouts.
See README.md in this package for the biology, the mapping, and the
honest limits (functional abstraction, not a simulation).

Stage 1: soul_teach + online_teach for SOUL.md / outcome alignment.
"""
from .circuit import FlyMB, load_circuit, text_features
from .soul_teach import ensure_soul_bootstrap, soul_episodes
from .online_teach import teach_from_user_text, teach_interrupt_honored

__all__ = [
    "FlyMB",
    "load_circuit",
    "text_features",
    "ensure_soul_bootstrap",
    "soul_episodes",
    "teach_from_user_text",
    "teach_interrupt_honored",
]
