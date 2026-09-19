"""Aiko biological layer II — fruit-fly central-complex compass.

Ring-attractor focus/heading plus R5 sleep-drive pressure, wired with real
MaleCNS v1.0 connectivity (1,282 neurons, 50,515 synapses at weight>=5).
See README.md for biology, mapping, and honest limits.
"""
from .compass import FlyCompass, load_circuit

__all__ = ["FlyCompass", "load_circuit"]
