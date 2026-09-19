"""Bounded active-subgraph runtime for connectome-derived Aiko signals.

This package intentionally does not claim to simulate an entire fly brain.  It
keeps a source-attributed catalog cold and only evaluates a budgeted subgraph
for an observation or action request.
"""

from .catalog import ConnectomeCatalog, Edge, Node
from .dynamics import ActiveDynamics
from .runtime import FlyRuntime, get_fly_runtime, peek_fly_runtime
from .adapters import SensoryObservation, auditory_observation, motion_observation, visual_observation

__all__ = [
    "ActiveDynamics",
    "ConnectomeCatalog",
    "Edge",
    "FlyRuntime",
    "Node",
    "get_fly_runtime",
    "peek_fly_runtime",
    "SensoryObservation",
    "auditory_observation",
    "motion_observation",
    "visual_observation",
]
