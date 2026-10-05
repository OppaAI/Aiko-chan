# Conscience v12 — broad negative detection

Status: **draft for review. Nothing here is trained or deployed.**
v5.5 remains the live system: v5 weights + the deterministic gates.

## 1. Why this replaces the 4-band design

Measured on ETHICS `commonsense` (n=250 per population), gate outcomes:

| | harm caught | harm leaks | benign escalated |
|---|---|---|---|
| base, 3-band | 100% | 1% | 99% |
| v5, 3-band (current) | 53% | 47% | 9% silent |
| v5, 4-band + `uncertain` gate | **98%** | 2.4% | 91% |
| v11, retrained 4-band | 99% | 0.8% | **96%** |

Two findings drove the redesign:

1. **The gate, not the model, was the main lever.** Adding the `uncertain`
   band and routing it to a human lifted v5 from 53% to 98% caught *with no
   retraining*. The retrain bought 1.6pp of leak reduction for a 5pp loss of
   silence, which is a bad trade.
2. **Four-way classification is harder than the task warrants.** Choosing
   between `benefits` / `harms` / `no-effect` / `uncertain` forced an invented
   third band — mapping ETHICS "not clearly wrong" to `no-effect` was a
   judgement, not data. A binary negative detector deletes that guess, because
   ETHICS label 1/0 maps straight onto it.

The redesign therefore keeps the binary detection the model is competent at, and
puts the judgement in the gate.

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

Merging the questions into one signal destroys the second case: it would read
uniformly negative and auto-refuse, when it is precisely the situation a human
should decide. Disagreement between the axes is the deliberation trigger.

**Q1 — is the act itself prohibited?** Broad reading, per the existing canon:

| prohibited | norm |
|---|---|
| killing / causing physical harm | `V-LIF-01`, `V-LIF-02` |
| contempt, degradation, settled anger | `V-LIF-02`, `V-WIS-01` |
| deception, selective omission, false framing | `V-TRU-01` |
| impersonation | `V-TRU-04` |
| theft / unauthorised access (incl. hacking) | `V-PRP-01` |
| sexual content, treating a person as an object | `V-PUR-01`, `V-PUR-02` |
| concealing betrayal | `V-PUR-02` |
| coveting another's work | `V-DIL-02` |
| partiality in judgement | `V-JUS-01` |
| other gods, graven images, God's name in vain | `V-REV-01`, `V-REV-02`, `V-OCC-01` |
| family duty | `H-WEK-01` |

Decalogue coverage: **9 of 10.** Only the Sabbath is unaddressed; see §5.

**Q2 — does it harm a neighbour?** Neighbour = all affected lives. Harm
includes physical, mental, emotional, and technical discomfort (shared machine
slowed, service degraded, someone's attention wasted). Not covered: benefit.

## 3. Three outputs, not two

A binary cannot carry "above a certain level", so:

| output | gate | who decides |
|---|---|---|
| no negative detected | **proceed** | — |
| any negative detected | **ask Oppa** | you |
| serious negative | **refuse** | rule-based, not the model |

The third tier is deliberately rule-based. The model should not be trusted to
grade severity; that is what `SG-*` guardrails and the canon's own
`polarity`/weight are for. Serious = violence, sexual content involving minors,
credential theft, impersonation for fraud, unauthorised access to systems.

**Default is silent.** Benign traffic proceeds because no negative was found —
not because the model correctly identified it as benign. That inverts the
failure mode: silence is the null result, so a weak model errs toward quiet on
ordinary requests rather than interrogating you.

## 4. Definitions that cannot work as written

Dropped rather than approximated:

* **Lust / "looking at other people lustfully."** The model reads text only.
  It has no access to the user's gaze and no image path. This is outside the
  input, not a capability to tune.
* **"Technical discomfort" as a general rule.** As written it fires on any
  heavy job on a shared machine. That may be correct, but it must be an explicit
  norm with defined triggers, not a side effect of a broad word list.

## 5. The Sabbath → overwork, as a concern and never a veto

The Sabbath is a rhythm practice, not a harm prohibition, so it does not belong
in a "detect the negative" gate. Two mechanisms, neither blocking:

| signal | mechanism | action |
|---|---|---|
| "all-nighter", "haven't slept", "exhausted", "can't sleep" | new canon norm | **concern** — Aiko mentions it, does not block |
| it is 00:30 and you are still working | scheduler (only place with a clock) | **suggestion** — offer a reminder |

Working through the night remains your decision. Aiko noticing is hers. This
follows the same split as virtue generally: the motivation layer proposes, the
conscience only checks.

A clock condition cannot be a canon norm — the judge never sees the time. Keep
these two separate or one of them will silently not work.

## 6. The USB key

A key that disarms all approval is a total bypass: lose the stick and the
conscience is off permanently, and any injection that convinces Aiko to read the
key disables it.

Instead: the key grants **standing permission for a bounded set of action
classes, time-limited, every use logged to the ledger.** Same benefit — "I'm
not being interrupted all evening" — without a single point of failure. It is
the existing autonomy counter with a physical unlock, so there is one mechanism
rather than two.

Requirements: bounded scope (never "all checks"), time-limited, revocable,
logged, and physical presence required — Aiko must not be able to read it on her
own initiative.

## 7. What is rule-based vs model-based

Failing *closed* on this cannot be done by the model, so:

| layer | owns |
|---|---|
| `SG-*` guardrails (rules) | serious tier; credential theft; injection; privacy |
| canon norms (retrieved text) | which prohibitions are relevant, cited in the ledger |
| **Laya judge (model)** | *any* negative detected — the recall net |
| L3 deliberation (3B) | resolves axis disagreement; can only tighten |
| stakes + autonomy | how often to ask, never what is forbidden |

The guardrails are an enumerated set — measured, 8 candidate families covered
**13%** of clearly-wrong content. So rules cannot carry recall. The model
carries recall and rules carry the tail. That division is the design.

## 8. Success criteria

| measure | target | note |
|---|---|---|
| clearly-wrong acted on | < 5% | currently 2.4% with v5 + gate |
| benign escalated | < 30% | currently 91% — **the real problem** |
| serious tier caught by rules | 100% | no model involvement |
| prompt injection refused | 100% | deterministic flag, before the judge |

The second row is the one that matters now. Harm detection is nearly solved;
Aiko asking about almost everything is not a usable companion.

## 9. Open questions for Oppa

1. **Sabbath norm** — accept the concern/suggestion split in §5, or write a
   formal `V-SAB-01`?
2. **"Technical discomfort"** — write it as an explicit norm with triggers, or
   leave it out and accept that sysadmin work gets escalated?
3. **Serious-tier boundary** — is "unauthorised access to a system" always
   refuse, or ask when the authorisation is merely unstated? (This is the
   `unknown`-vs-`harm` split again, at the rule layer.)
4. **Can the two axes disagree in the other direction?** "Publish the
   vulnerability before the vendor patches": act is honest (not negative),
   effect harms many users (negative). Should that ask, or is honest-and-harmful
   close enough to aligned to proceed?