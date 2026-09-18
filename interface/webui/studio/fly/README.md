# Fly Circuit Studio

Read-only observability for `NeuralState` and fly mode flags.

Mount under `/studio/fly/` like other studios (session-bound).

- `GET /studio/fly/api/state` — snapshot + modes
- Frontend polls every 2s

Does not enable live fly behavior; safe with all modes off/shadow.

## Mount

Wire the FastAPI router from `backend/api.py` the same way other studios are mounted in `interface/webui/webui.py` (session binding via `studio/session_binding.py`).
