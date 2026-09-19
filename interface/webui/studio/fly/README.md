# Fly Circuit Studio

Session-bound observability for fly-motif circuits and live `NeuralState`.

## What it looks like

A polished observability dashboard with a compact live-state sidebar and a
bounded projection of the highest-rate cells/edges from a real active trace.
When no trace is available it falls back to the simplified neural-net motif:

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
| `/studio/fly/api/trace?node_limit=30&edge_limit=80` | Identity-scoped, bounded active trace projection |

Mounted from `interface/webui/auth.py` like other studios. Safe with all fly modes off.
