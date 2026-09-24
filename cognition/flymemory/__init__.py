"""Aiko biological layer — fruit-fly mushroom-body associative memory.

Grafts a real MaleCNS v1.0 microcircuit (4,673 neurons, 121,706 synapses at
weight>=5) onto Aiko's valence system as opponent approach/avoid readouts.
See README.md in this package for the biology, the mapping, and the
honest limits (functional abstraction, not a simulation).

Stage 1: soul_teach + online_teach for SOUL.md / outcome alignment.
Stage 2: teach_preference / extract_topic for explicit do-or-don't teaching.
Stage 3: eligibility trail. Stage 4: dopamine PAM/PPL1 + MB sleep consolidation.
"""
from .circuit import FlyMB, load_circuit, text_features
from .fullbrain import FullBrain, get_fullbrain, load_fullbrain
from .soul_teach import ensure_soul_bootstrap, soul_episodes
from .online_teach import teach_from_user_text, teach_interrupt_honored
from .teach_api import teach_preference, extract_topic
from .eligibility import assign_credit, record_step, stats as eligibility_stats
from .dopamine import pulse as dopamine_pulse, split_channels as dopamine_channels
from .consolidate_mb import consolidate as consolidate_mb

__all__ = [
    "FlyMB",
    "load_circuit",
    "text_features",
    "FullBrain",
    "get_fullbrain",
    "load_fullbrain",
    "ensure_soul_bootstrap",
    "soul_episodes",
    "teach_from_user_text",
    "teach_interrupt_honored",
    "teach_preference",
    "extract_topic",
    "assign_credit",
    "record_step",
    "eligibility_stats",
    "dopamine_pulse",
    "dopamine_channels",
    "consolidate_mb",
]
