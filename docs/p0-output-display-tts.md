# P0 — Output display / TTS alignment

## Goal

Keep leading **emoji** and **karaoke typewriter**. Stop showing/speaking useless `ACTION: none`. Align TTS with dialogue only.

## Files (upload from `artifacts/p0/`)

| Path | Change |
|------|--------|
| `sensory/speak.py` | `format_for_display`, `dialogue_for_stream`; stream path never types raw chrome |
| `cognition/think.py` | `_emit` / karaoke `_emit_finalized_response` use those helpers |

## Upload

```bash
git checkout fix/output-display-tts-p0
cp artifacts/p0/sensory/speak.py sensory/speak.py
cp artifacts/p0/cognition/think.py cognition/think.py
git add sensory/speak.py cognition/think.py
git commit -m "P0: display drops ACTION:none; karaoke/TTS dialogue-only (keep emoji)"
git push
```

## Smoke

1. `ACTION: none` → not on screen; TTS is dialogue
2. Real action → still shown
3. Karaoke → typewriter tracks voice; emoji kept
