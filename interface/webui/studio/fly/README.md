# Fly Circuit Studio

Session-bound observability for fly-motif circuits and live `NeuralState`.

## What it looks like

A **simplified neural-net / connectome motif** graph (not a full MaleCNS EM render):

```
Sensory → AL / T4-T5 → MB · LH · CX → GF → DN → Aiko
```

Nodes glow from live proxies (valence, focus, urgency, sleep, …). Edges brighten when both ends are active.

## Routes

| Path | Role |
|------|------|
| `/studio/fly/` | Frontend |
| `/studio/fly/api/state` | NeuralState + modes |
| `/studio/fly/api/circuit` | Motif graph + activation |

Mounted from `interface/webui/auth.py` like other studios. Safe with all fly modes off.
