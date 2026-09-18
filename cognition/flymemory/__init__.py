"""Aiko biological layer — fruit-fly mushroom-body associative memory.

Grafts a real MaleCNS v1.0 microcircuit (4,673 neurons, 121,706 synapses at
weight>=5) onto Aiko's valence system as opponent approach/avoid readouts.
See README.md in this package for the biology, the mapping, and the
honest limits (functional abstraction, not a simulation).
"""
from .circuit import FlyMB, load_circuit

__all__ = ["FlyMB", "load_circuit"]
