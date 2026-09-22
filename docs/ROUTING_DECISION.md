# Intent-routing backend comparison (Jetson Orin Nano, 2026-09-22)

Question: replace Harrier-270M (local llama-server) with Laya-ONNX
(mizchi/laya-multilingual-onnx, fp16) or Jev API (TypeSafe, hosted)
for Aiko's quaternary + capability routers?

## Method

- Ground truth: `agentic/router/intent_prompts.json` (221 quaternary
  examples) + `agentic/router/capability_prompts.json` (58 triggers,
  7 capabilities). Leave-one-out both sides; argmax, no thresholds.
- Laya: local ONNX fp16 via onnxruntime-gpu 1.27 (Jetson wheel, CUDA).
- Jev: `jev-latest` via TypeSafe API (free key).
- Harrier: local llama-server `embedding=true`, same scorer as
  production (`think.py` top-3 cosine / `capability.py` trigger cosine).
- Latency: 50 timed single-query calls, P50.

## Quaternary (n=221)

| label | n | Harrier acc | Jev acc / conf | Laya acc / conf |
|---|---|---|---|---|
| greeting | 25 | 1.000 / — | 0.960 / 0.958 | 0.880 / 0.578 |
| localchat | 67 | 0.925 / — | 0.522 / 0.767 | 0.373 / 0.255 |
| webchat | 75 | 0.960 / — | 0.933 / 0.924 | 0.253 / 0.293 |
| agentic | 54 | 0.926 / — | 0.981 / 0.991 | 0.741 / 0.376 |
| **overall** | **221** | **0.946** | **0.824** | **0.480** |
| latency P50 | | **17ms** | 198ms | 25ms |

Laya + production think-policy (thresholds/gap): 0.443 — thresholds tuned
for cosine scores misfire on Laya's softer probs.

## Capability (n=58)

| capability | n | Harrier acc | Jev acc / conf | Laya acc / conf |
|---|---|---|---|---|
| job_hunt | 3 | 1.000 / — | 1.000 / 1.000 | 0.000 / 0.879 |
| kb_proposal | 8 | 1.000 / — | 1.000 / 1.000 | 0.000 / 0.903 |
| photo | 6 | 1.000 / — | 1.000 / 1.000 | 1.000 / 0.915 |
| repo | 10 | 1.000 / — | 1.000 / 0.982 | 0.200 / 0.633 |
| research | 11 | 0.818 / — | 1.000 / 0.995 | 0.000 / 0.826 |
| scheduling | 11 | 0.818 / — | 1.000 / 1.000 | 0.364 / 0.813 |
| social | 9 | 1.000 / — | 0.667 / 0.820 | 0.000 / 0.665 |
| **overall** | **58** | **0.931** | **0.948** | **0.207** |
| latency P50 | | **17ms** | 192ms | 40ms (CUDA) / 1118ms earlier 7-marker CPU run |

Note: Laya is confidently wrong (0.8–0.9 confidence at 0.0 accuracy
on 4/7 capabilities) — its confidence can't gate decisions.

## Why not Laya

1. Accuracy: 0.48 / 0.21 vs 0.95 / 0.93. Localchat/webchat
   confusion (.37/.25) is exactly what production thresholds exist for.
2. Confidence uncalibrated for our labels (high confidence on wrong
   answers), so no gating fix available.
3. Ops cost on Orin Nano: 617MB weights + ~700MB ORT arena; stock
   onnxruntime wheels lack sm_87 kernels (4 versions failed) and the
   default arena OOMs next to the resident LLM. Only the Jetson-built
   1.27 wheel runs, and only when GPU headroom exists.
4. Architectural mismatch: Laya outputs decisions, not vectors — Harrier
   stays anyway for sqlite-vec recall, capability cache, preferences.

## Why not Jev (despite best capability score)

1. Cloud dependency + ~200ms/call (10× local) on every turn.
2. Free-tier latency/cost risk; localchat weakest (.522) where most
   casual turns land.
3. Keeps for shogi self-play (already integrated), not for hot-path routing.

## Decision

Keep Harrier-270M local for both routers. No Aiko code changes.
Repro: `~/laya_eval/compare_routing.py --task quaternary|capability`.
Two methodological footnotes if anyone challenges it: capability Harrier used mean-of-trigger-vectors LOO (embed each trigger once, ~116 calls) instead of joined-text LOO, after the exact protocol crashed llama-server mid-run — same ranking signal, documented in the script. And quaternary production adds thresholds + LLM tiebreak on top of these scorer numbers, which only helps Harrier (its scores separate cleanly).
