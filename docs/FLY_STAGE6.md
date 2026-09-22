# Fly Stage 6 — Embodied Aiko

DN expands beyond TTS rate/volume; GF cancel is global for speech/tools/jobs.

| Piece | Module |
|-------|--------|
| DN drive (speech + body) | `flysense/dn.py` |
| Body packet | `fly_behavior/dn_body.py` |
| TTS + GF mute | `fly_behavior/dn_tts.py` |
| Global cancel | `fly_behavior/gf_global.py` |

## Config

```yaml
MEMORY_FLYDN_MODE: "live"
MEMORY_FLYGF_MODE: "live"
```

## Observe

1. Causal trail: `dn` with expression/gesture/gaze; `dn_body`; `gf_clear`
2. High sleep / low energy → lower `action_vigor` / softer prosody
3. `STOP` → TTS cancel + agent abort + body cancelled

## Still later

LIF spikes (Stage 7), full MaleCNS experimental mode (Stage 8).
