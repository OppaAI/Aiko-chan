"""
Koi-Koi: play Koi-Koi (hanafuda) vs Aiko.

Android-app backend, parallel to shogi/ and go/.
Importing this package exposes the FastAPI router; auth.py mounts it
explicitly (see interface/webui/auth.py).

Rules engine: cards.py (pure stdlib). AI: ai.py heuristic ("aiko"),
always available — no external engine binary exists for Koi-Koi
comparable to YaneuraOu/KataGo, and none is needed on the Nano.
"""

from .games_koikoi import router

__all__ = ["router"]
