# Fly Stage 6.1 — Front-end DN body wiring

Wires Stage 6 `body_drive()` into Studio + VRM companion.

| Piece | Where |
|-------|--------|
| `/studio/fly/api/body` | Studio API → body packet + avatar_intents |
| DN body panel | Studio sidebar meters |
| `window.aikoApplyDnBody` | `vrm.js` expression / gesture amplitude |
| Companion poll | `/static/companion_dn_body.js` every ~2.5s |

## Observe

1. Open Fly Studio → **DN body drive** panel updates while chatting
2. Avatar expression soft-tracks expression_intensity
3. `STOP` / GF interrupt → neutral + cancelled flag on body panel

Requires `MEMORY_FLYDN_MODE=live` (and GF live for cancel).
