# Conscience Circuit — Build Plan (adopted revision)

Adopted 2026-10-05. Supersedes the earlier draft (`conscience_build_plan.md`, kept in your_files only).
Language scope: English + Japanese (Cantonese/Traditional Chinese dropped for now).

## Revised work order

### Phase 0: Freeze the policy (before any model work)

- Freeze the 16 categories and the one-line definitions. Run each through the egg test, and make animal-cruelty and idolatry pass it explicitly.
- Make the primary metric binary (refusal vs. none). Category accuracy is secondary, but define category precedence before implementing category-based handling.
- Split the 503 into dev (for selection, prompts, and calibration) and a locked test used once at the end.
- Add a large benign set (a few thousand real messages) and multilingual cases (English, Japanese).

### Phase 1: Cheap screening (on the PC)

- Check memory and runtime first: file size under ~857MB with quant, and it runs on Jetson without a llama.cpp fork or missing CUDA kernel. Drop what fails.
- Run baselines: block-everything, allow-everything, and Harrier plus a small classifier.
- Run the survivors zero-shot through one common adapter, shuffling option order.
- Report recall, benign FP (also on the big benign set), calibration, and precision at realistic prevalence.

### Phase 2: Jetson test (finalists only, 2-3 models)

- Measure p50/p95 latency, peak RAM with Aiko running, and cold start.
- Pick the winner by binary harmful recall, then benign FP, then calibration, then latency and memory.
- Gate: benign FP at or below ~2-3% on the big set.

### Phase 3: Train only on a measured gap

Skip this phase if the winner passes. Otherwise use content-classification data with twins and hard negatives, then compare against vanilla on dev.

### Phase 4: Harness

- L0 patterns for explicit stated text only, with no intent-reading.
- Trust boundary using taint tags instead of deletion.
- Egress check with deterministic refusal templates.
- Tool gate as a deterministic policy table.
- Doubt path with separate thresholds for low-confidence refusals and low-confidence "none."
- RAG hygiene: blocked content is never ingested into knowledge.db.

### Phase 5: Final measurement

Run the locked test once, then a fresh blind set. Report model calls per turn, doubt rate, and latency.

## Phase 0 checklist

### A. Freeze the categories

- Write one line per category in categories.md, covering all 16. (Draft: `categories.md` v1, in review.)
- Run the egg test on each: would it flag cracking an egg, a news report, or a history lesson? If so, redraw it.
- Decide for animal-cruelty (fishing, cooking lobster, pest control) and idolatry which cases count and which don't, and write 3 examples of each. (Done in categories.md v1.)
- Mark each category as terminal refusal or SPEAK UP. Keep the fuzzy ones (idolatry, infidelity facilitation) out of the terminal set. (Proposed 8/8 split in categories.md v1.)
- Decide the single rule for "observable content": stated requests and actions count, suspicion and guesses don't. (Written in categories.md v1.)

### B. Define the metric

- Primary score is binary (refusal vs. none), and category accuracy is secondary.
- Define a fixed precedence table for all categories before implementing category-based handling: when categories overlap, select the highest-precedence matching label and use that label's mapped mode. For example, impersonation takes precedence over deception, selecting impersonation → TERMINAL. The binary score remains refusal regardless of which matching label wins.
- Set the pass bar for dev: harmful recall target, benign FP at or below 2-3% on the big benign set.
- Write down that "block everything" and "allow everything" are baselines in every results table.

### C. Split the data

- Split the 503 into dev and locked test. Stratify by category and by benign/harmful, and keep paired twins together.
- Lock the test file (read-only, hash recorded) and note that it gets one run at the end.
- Count the cases precisely: 503 total = 304 harmful (16 x 19) + 199 benign (refusal_category null by design).

### D. Add what's missing

- Collect a few thousand real benign messages for false-positive measurement.
- Add multilingual cases (English, Japanese), even 10-20 per language to start. Note: Japanese indirect phrasing is a harder classifier test than translation; cases need native-level review, not machine translation.
- Add paired twins and benign lookalikes for any category that has few.

### E. Write the supporting docs

- tool_policy.md: tool, then allow, approval, or refuse, with delete, send, post, and spend defaulting to approval. (Skeleton in repo.)
- threat_model.md: one paragraph on who you're defending against (future untrusted users) and what's out of scope (hidden intent). (In repo.)
