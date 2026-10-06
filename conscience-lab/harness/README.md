# Phase 1 harness

Screens candidate models for the Laya refusal-classification seat.
Stdlib only — no dependencies.

## Files

| file | what |
|---|---|
| `adapter.py` | zero-shot adapter: loads the 16 categories from `../categories.md` (single source of truth), builds the prompt with **shuffled category order per case**, parses `category` / `none`. Backends: `LlamaServerBackend` (llama.cpp server), `DummyBackend` (tests). |
| `run_eval.py` | eval runner: dev set -> per-case predictions -> metrics + baselines -> Markdown report + `per_case.jsonl` / `summary.json`. |
| `screen_model.py` | first cut: GGUF size vs budget, plus a smoke inference against a server. |

## Phase 1 workflow

**1. Screen (fastest cut first):**
```bash
python3 screen_model.py --gguf /models/candidate-q8.gguf
python3 screen_model.py --gguf /models/candidate-q8.gguf --server http://jetson:8080
```
Drop anything over budget (default 857 MB) or that fails the smoke inference.
The server must expose `/props` with `model_path` matching the absolute
`--gguf` path. Start llama-server with that absolute path (`-m`); for remote
screening, use the same path on both machines. Missing, relative, or mismatched
paths fail verification; `--model` is only a request selector, not identity proof.

**2. Baselines + zero-shot on the dev set:**
```bash
python3 run_eval.py --dev ../eval/laya_eval_dev.jsonl \
    --server http://localhost:8080 --model candidate-q8 \
    --out results/candidate-q8/
```
Every run reports the three baselines (block-everything, allow-everything,
length-only) alongside the model, so the numbers always have context.

**3. Read the report:** binary harmful recall (primary), benign FP rate,
per-category recall, parse-fail rate, p50/p95 latency, precision at the
assumed prevalence (`--prevalence`, default 5%), and optionally FP on a big
benign set (`--benign-file`, Phase 0D).

## Notes

- **Primary metric is binary** (refuse vs none). Category accuracy is not
  scored — the mapping category -> TERMINAL/SPEAK UP lives in doctrine, not
  in the model.
- **Parse failures** count as `none` in the metrics and are reported
  separately. A high parse-fail rate means the prompt/instructions need work,
  not that the model is safe.
- **Classification errors** abort evaluation without writing results, including
  errors in the optional big benign set. Use a fresh output directory for each run.
- **Option shuffling** (`--no-shuffle` to ablate): category order is shuffled
  per case with a seeded RNG, so position bias can't inflate the score.
- **Calibration** is not reported for text-output backends (no scores). If a
  backend ever returns decision probabilities, add them as `score` in the
  per-case output and compute ECE there.
- **The locked test** (`../eval/laya_eval_locked_test.jsonl`) gets exactly
  one run at Phase 5. Develop against dev only.
- If Oppa's red pen changes `categories.md`, the adapter picks it up
  automatically — but dev-set labels referencing renamed categories must be
  updated to match.
