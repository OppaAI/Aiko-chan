"""Aiko sensory grafts — fruit-fly antennal lobe, motion vision, DN output.

- al: divisive normalization from real MaleCNS AL gain stats (49 glomeruli).
- motion: T4/T5 Reichardt energy with real subtype convergence + LPTC pooling.
- dn: descending-neuron-style output drive mapping (functional, stats-backed).
- pathways (Phase 6): per-turn sensory encoders — voice prosody, motion, and
  visual frames become neural activity published into NeuralState.
See README.md for real-vs-synthetic accounting.
"""
from .al import FlyAL
from .dn import FlyDN, get_flydn
from .motion import FlyMotion, emd_energy
from .pathways import (
    MotionPathway,
    encode_turn_senses,
    encode_voice_prosody,
    get_motion_pathway,
    note_visual_frame,
)

__all__ = [
    "FlyAL",
    "FlyDN",
    "FlyMotion",
    "emd_energy",
    "get_flydn",
    "MotionPathway",
    "encode_turn_senses",
    "encode_voice_prosody",
    "get_motion_pathway",
    "note_visual_frame",
]
