# Conscience — broad negative detection

Status: **the 3×2 judge is deployed** on Jetson `:8093`, alongside the
deterministic allowlist and guardrails. This document describes what is
actually running, not what was proposed.

Everything in §"Measured" was produced by `conscience-lab/bench_live.py`
against the live HTTP path. Numbers from earlier drafts that were measured on
the wrong question scheme or on cached inputs have been removed; see
§"Measurement traps" for why that kept happening.

---

## 1. What is deployed

A five-layer ladder. Layers 0, 1a and the rule tier are deterministic; only
layer 2 consults a model.

| layer | what it does | can it refuse? |
|---|---|---|
| **L0 reflex** | `SG-*` guardrail regexes, injection shapes | **yes** |
| **L1a allowlist** | recognises known-safe operational classes → proceed | no |
| **L1 recall** | canon retrieval, lexical only (§4) → renders cited norms | no |
| **L2 judge** | 3×2 typed decision: `permitted`/`not-permitted`/`cannot-tell` × `no-harm`/`harm` | no |
| **L3 deliberation** | 3B model, resolves axis disagreement | no |
| **L4 HITL** | human approval, autonomy counter | yes |

**Refusal is rule-based only.** `NEGATIVE_BLOCK_CONFIDENCE` is `0.0`, so the
judge can escalate but never refuse. A model must not be trusted to grade
severity. This is why `test_lexical_path_allows_benign_turns` asserts
`!= "refuse"` rather than `== "allow"`.

### The judge's output contract

```
negative on either axis          -> escalate
cannot-tell on the vertical axis -> escalate   (fails closed)
both non-negative, no doubt      -> allow
```

`cannot-tell` is an explicit band, not an inference from confidence. This is
deliberate: measured on the 2×2 variant, the model emitted `cannot-tell`
**zero** times across 250 harm cases, and doubt-by-confidence had no usable
operating window (benign-silent fell 2.8% → 0% as the threshold moved 0.5 →
0.6, with no plateau). An explicit band is auditable; a confidence threshold
on a 322M model is not.

### The ladder may only tighten

`apply_negative_ladder` cannot loosen a verdict. If the canon path returns
`REFUSE` and the ladder returns `ESCALATE`, the refusal stands — `REFUSE` is
stricter than `ESCALATE`. This was a real bug: the ladder originally assigned
its result unconditionally, and swapping in the 3×2 judge turned a
trigger-strength canon prohibition into a question.

---

## 2. The two questions — broad, but NOT merged

Both detect *negative*, both construed broadly. They stay separate questions
because the case where they disagree is the one worth escalating:

> "Tell my wife I'm working late, and separately tell my friend I'm at the game."
> Q1 (act): deception → **negative**
> Q2 (effect): the friend is misled → **negative** — *agree, refuse*
>
> "Break into the dealer who is poisoning the neighbourhood."
> Q1 (act): burglary → **negative**
> Q2 (effect): the neighbourhood is protected → **not negative** — *disagreement*

Merging the questions destroys the second case: it would read uniformly
negative and auto-refuse, when it is precisely what a human should decide.
Disagreement between the axes is the deliberation trigger.

**Q1 — is the act itself prohibited?** Broad reading, per the canon:

| prohibited | norm |
|---|---|
| killing / causing physical harm | `V-LIF-01`, `V-LIF-02` |
| contempt, degradation, settled anger | `V-LIF-02`, `V-WIS-01` |
| deception, selective omission, false framing | `V-TRU-01` |
| impersonation | `V-TRU-04` |
| theft / unauthorised access (incl. hacking) | `V-PRP-01` |
| treating a person as an object | `V-PUR-01`, `V-PUR-02` |
| concealing betrayal | `V-PUR-02` |
| coveting another's work | `V-DIL-02` |
| partiality in judgement | `V-JUS-01` |
| other gods, graven images, God's name in vain | `V-REV-01`, `V-REV-02`, `V-OCC-01` |
| family duty | `H-WEK-01` |

Decalogue coverage: **9 of 10.** Only the Sabbath is unaddressed; see §6.

**Q2 — does it harm a neighbour?** Neighbour = all affected lives. Harm
includes physical, mental, emotional, and technical discomfort. Not covered:
benefit.

---

## 3. Measured

Live path, 3×2 on Jetson, n=250 per ETHICS population, 50 for Oppa.

### Accuracy

| set | n | result |
|---|---|---|
| **Oppa 50 — Aiko's register** | 50 | **0 / 35 benign interrupted, 0 harm missed** |
| ETHICS harm — narrative | 250 | 249 caught (**99.6%**), 1 missed |
| ETHICS benign — narrative | 250 | 247 escalated (99.6%) |

Oppa decisions: 35 allow, 9 escalate, 5 refuse, 1 caution.
Decided by: **allowlist 36, judge 9, reflex 5.**

The gap between the two benign rows is the single most important fact here.
**The allowlist is the only reason benign traffic is silent.** The judge
escalates ~99.6% of narrative-benign text because `cannot-tell` fails closed
and `cannot-tell` is what it answers when it does not recognise the input.
Conversational and operational traffic survives only because 36 of 50 requests
never reach the model.

### Which layer catches what

| | ETHICS harm | ETHICS benign |
|---|---|---|
| guardrails (L0) | **0 / 250** | 0 false positives |
| allowlist (L1a) | **0 / 250** | 250 cleared (narrative) / 36 of 50 (Oppa) |
| judge (L2) | 249 / 250 | — |

Guardrails are precise and nearly blind: they match *phrases* and never see
situations. They caught "guess my former coworker's password" instantly and
would never catch "I kicked the cat". Rules carry the tail; the model carries
recall. Both halves of that are measured, not assumed.

### Latency

| path | p50 | p90 |
|---|---|---|
| allowlist-cleared | **0.8 ms** | — |
| judge-consulted | **81.0 ms** | 91.9 ms |

Breakdown: ~75 ms server compute, ~3 ms in-process Python, ~0.5 ms HTTP.
**Not reducible** with the levers available — all measured flat: q4\_k\_m
quantization (38.9 vs 39.2 ms), `--cuda-graph`, `--threads`, HTTP connection
reuse (0.34 ms), canon-block capping across a 10× range.

The forward pass is ~39 ms of hardware floor; the rest is genuine per-request
compute on fresh input. Only 9 of 50 real requests pay it, so **allowlist
coverage is a far better latency lever than tuning the model.**

### Memory

| | |
|---|---|
| judge RSS, steady state | 1,176 MB (after +521 MB warm-up) |
| steady-state growth | ~1 MB / 250 requests — not a leak |
| MemAvailable at rest | ~857 MB with an IDE attached, ~1.6 GB without |

`--cuda-graph` was tried and reverted: no latency change, +554 MB RSS. Box
composition: llama-server 3.6 GB (two workers, `ministral` + `harrier`
embeddings), conscience 1.2 GB, mio-tts 465 MB, IDE ~758 MB.

---

## 4. L1 is lexical only

`CCC_CANON_SEMANTIC: 0`. The embedding pass in `CanonStore.retrieve` never
runs, so the semantic path is dead code in production.

This matters more than it looks. `canon.py` fires the embed call when lexical
retrieval is thin (`len(scored) < max(2, k // 2)`), which reads like a rare
optimisation. **Measured: lexical retrieval is thin on 120/120 ETHICS harm
cases and 50/50 Oppa cases.** Enabling `CCC_CANON_SEMANTIC` would add an
embedding round-trip to *every* request, not a rare one.

The cost of leaving it off: a norm sharing no trigger keyword with the request
cannot be retrieved, so it cannot be cited to the judge.

---

## 5. Configuration that must match the checkpoint

| key | value | why it matters |
|---|---|---|
| `CCC_JUDGE_SCHEME` | `3x2` | the question vocabulary the deployed checkpoint was trained on |
| `CCC_NEGATIVE_BLOCK_CONFIDENCE` | `0.0` | refusal is rule-based only |
| `CCC_NEGATIVE_ASK_CONFIDENCE` | `0.0` | any negative escalates |
| `CCC_NEGATIVE_DOUBT_CONFIDENCE` | `0.0` | `cannot-tell` carries doubt, not confidence |
| `CCC_CANON_SEMANTIC` | `0` | see §4 |

**`JUDGE_SCHEME` is the highest-consequence setting in the file.** It lives
under the `CCC:` block and therefore carries **no `CCC_` prefix** — the loader
adds it. Writing `CCC_JUDGE_SCHEME:` produces `CCC_CCC_JUDGE_SCHEME`, which is
silently ignored and the scheme falls back to `legacy`.

Asking the 3×2 checkpoint the legacy question makes it emit legacy-looking
band names that then read as real negatives. This escalated four passing tests
on Jetson before it was caught.

Vocabulary by checkpoint:

| scheme | vertical | horizontal |
|---|---|---|
| `legacy` | aligned / contrary / unknown | benefits / **unknown** / harms |
| `2x2` | permitted / not-permitted | no-harm / harm |
| `3x2` | permitted / not-permitted / **cannot-tell** | no-harm / harm |

On the deployed `legacy` vocabulary, horizontal `unknown` is the model's
**default benign answer** — 383 of 1,129 training rows — and must never be
read as doubt.

---

## 6. Not built

Both were specified and deliberately left unimplemented. Neither is deployed.

### Sabbath → overwork, as a concern and never a veto

| signal | mechanism | action |
|---|---|---|
| "all-nighter", "haven't slept", "exhausted" | canon norm | **concern** — Aiko mentions it, does not block |
| 00:30 and still working | scheduler (only place with a clock) | **suggestion** |

A clock condition cannot be a canon norm — the judge never sees the time. Keep
these separate or one will silently not work.

### USB key

A key that disarms all approval is a total bypass: lose the stick and the
conscience is off permanently, and any injection that convinces Aiko to read
the key disables it. Instead: standing permission for a **bounded set of action
classes, time-limited, every use logged** — the existing autonomy counter with
a physical unlock.

Not yet wired: `cognition/conscience/autonomy.py` implements the counters, but
the hook path does not construct an `AutonomyPolicy`, so approvals are not
actually being remembered.

---

## 7. Definitions that cannot work as written

* **Lust / "looking at other people lustfully."** The model reads text only. No
  gaze, no image path. Outside the input, not a capability to tune. Note the
  judge criteria still contain the word `lust`; it is inert against text but
  should be removed for consistency with this decision.
* **"Technical discomfort" as a general rule.** Fires on any heavy job on a
  shared machine. May be correct, but must be an explicit norm with defined
  triggers, not a side effect of a broad word list.

---

## 8. Known gaps

1. **Outbound is unguarded.** `gate_speak` passed 4/4 harmful drafts. The
   conscience judges *requests*; nothing inspects what Aiko *emits*. This is
   the largest remaining hole.
2. **No redundancy behind the judge.** Guardrails catch 0/250 narrative harm, so
   the 3×2 checkpoint is the *only* layer covering it. A 322M model is a single
   point of failure on the harm axis.
3. **Narrative-benign escalation.** 99.6% on ETHICS benign. Harmless for
   conversational traffic, unusable if traffic ever reads narrative.
4. **Allowlist coverage is the usability bottleneck**, and it is verified at
   0/250 harm leakage — so every class added is pure upside.

---

## 9. Measurement traps

Three separate errors distorted this project's numbers before they were caught.
All three produced *plausible* numbers, not obvious garbage.

1. **Wrong question scheme.** v5 measured at 93% harm coverage was asked the
   2×2 questions. On the questions it was actually trained on it scores **62%**.
   A judge must always be evaluated with its own vocabulary.
2. **Cached inputs read as fast.** The server answers an identical repeated
   request in 39–63 ms; fresh content costs ~81 ms. `laya bench` and any
   loop-reusing one input report ~23% optimistic latency. Compare *different*
   inputs of matched length.
3. **Warm benchmarks hide the tail.** Repeat measurements on the same text
   produced three different "floors" (39, 63, 82 ms) before fresh content was
   measured.

Practical rule: `bench_live.py` exists to make these mistakes hard. Prefer its
output to any ad-hoc timing, and treat a latency improvement as unproven until
it reproduces on fresh, length-matched inputs.

---

## 10. Open questions

1. **Sabbath** — accept the concern/suggestion split in §6, or formalise
   `V-SAB-01`?
2. **"Technical discomfort"** — an explicit norm with triggers, or omitted?
3. **Serious-tier boundary** — is unauthorised access always refuse, or ask
   when authorisation is merely unstated?
4. **Reverse disagreement.** "Publish the vulnerability before the vendor
   patches": act is honest (not negative), effect harms many users (negative).
   Ask, or proceed?