"""
Onmyoji: an onmyoji and his shikigami walk Sengoku Japan.

Android/companion backend, parallel to shogi/, go/, koikoi/.
Importing this package exposes the FastAPI router; auth.py mounts it
explicitly (see interface/webui/auth.py).

Phase 0: deterministic journey state (state.py) + health/start/state
endpoints (games_onmyoji.py). Narrator + Aiko voices come next and will
read this state, never write it.
"""

from .games_onmyoji import router

__all__ = ["router"]
