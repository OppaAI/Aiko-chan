"""Aiko biological layer II — fruit-fly central-complex compass.

Ring-attractor focus/heading plus R5 sleep-drive pressure, wired with real
MaleCNS v1.0 connectivity (1,282 neurons, 50,515 synapses at weight>=5).
See README.md for biology, mapping, and honest limits.
"""
from .compass import FlyCompass, load_circuit
from .temporal import (
    TemporalCX,
    clear_temporal_cx,
    cx_mode,
    drive_bias_for_candidate,
    get_temporal_cx,
    get_urgency_trace,
    reset_urgency,
    teach_gain_for,
    tick_cx_temporal,
)

__all__ = [
    "FlyCompass",
    "load_circuit",
    "TemporalCX",
    "clear_temporal_cx",
    "cx_mode",
    "drive_bias_for_candidate",
    "get_temporal_cx",
    "get_urgency_trace",
    "reset_urgency",
    "teach_gain_for",
    "tick_cx_temporal",
]
