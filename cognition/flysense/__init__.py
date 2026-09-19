"""Aiko sensory grafts — fruit-fly antennal lobe, motion vision, DN output.

- al: divisive normalization from real MaleCNS AL gain stats (49 glomeruli).
- motion: T4/T5 Reichardt energy with real subtype convergence + LPTC pooling.
- dn: descending-neuron-style output drive mapping (functional, stats-backed).
See README.md for real-vs-synthetic accounting.
"""
from .al import FlyAL
from .dn import FlyDN
from .motion import FlyMotion, emd_energy

__all__ = ["FlyAL", "FlyDN", "FlyMotion", "emd_energy"]
