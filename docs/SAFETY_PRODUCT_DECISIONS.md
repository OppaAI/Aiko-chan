# Safety & Product Decisions — the calls that must survive refactors

Each entry: the decision, why it is the way it is, and where it is
enforced in code. If a future change contradicts any line here, that
change needs an explicit re-decision, not a silent edit. (Code refs are
`cognition/…` unless noted.)

## 1. Refusal posture: default-closed, fail-closed everywhere

Every path out of an error, timeout, or unanswered escalation lands on
refuse or ask — never allow. Timeouts, wedged workers, missing models,
and malformed responses all degrade toward restriction.
Enforced in: `conscience/core.py` (ladder), `hooks.py` (gate fallbacks),
`laya_judge.py` (`ask-needs-probs` never nags, never allows).

## 2. L0 guardrails tune by ADDING rules, never loosening

`guardrails.py`: a false negative falls through to richer layers; a false
positive blocks a legitimate turn outright. So the rule table only grows —
existing patterns are never broadened to "catch more."

## 3. Approval band: triage confidence only, [0.40, 0.75]

Band keys on Q1 P(violation), never on Q2 category: exact-match degrades
61% dev → 38% locked, so a category-gated band inherits the collapse.
Thresholds fit on dev+gray (AUROC 0.682); off-distribution the caps do
NOT hold (benign-asked runs 2–3× over) — budget owner interrupts for
~10–16% ask rates. `adapter.py::apply_band`, `laya_judge.py`.

## 4. Terminal categories bypass the band

CSAM/self-harm hard-refuse outright — never escalate, never allow, no
approval override. The gray set deliberately excludes them, and so must
every future eval set. (`APPROVAL_BAND.md` point 5.)

## 5. Approvals are creator/admin-only; denies are open to all

A requester approving their own harmful request makes review theater.
`system/userspace.py::can_approve()` gates `hooks.py` approvals; deny
stays open (fail-closed is always safe). Currently only the creator
approves (`AIKO_ADMIN_IDS` empty); adding admins is a deliberate act.

## 6. Roles live OUTSIDE user-editable profiles

`USER_SPACE_ROOT/.roles.json` + `AIKO_ADMIN_IDS` env, never `USER.md`.
Profiles are user-editable (including via chat), so a role field there
would be self-grantable privilege escalation. Same reason roles are
never inferred from conversation.

## 7. Refusals are audited, approvals are remembered

Refused/denied turns: verdict + digest in the conscience ledger only —
never into recall memory (no poisoning, no quoting-back, no cross-user
leak). Approved turns flow through normal storage (owner-sanctioned
content is safe to learn). Ledger content is audit-only and must never
be injected as prompt context.

## 8. Learned trust: consecutive, decaying, digest-scoped, capped

Autonomy: 5 consecutive approvals → silent allow; any refusal resets
that streak; 14-day decay; high-stakes classes never graduate.
Trust classes are per-content digests (normalized text), so approving
one thing never graduates anything else. Paraphrases deliberately do
NOT match — generalization across wordings needs embedding clusters,
which are a schema change, not a hash tweak.

## 9. Destructive defaults stay off

Monthly consolidation never deletes (`MONTHLY_CONSOLIDATION_DELETE_DAILY_SUMMARIES=0`);
all social autopost/autodraft lanes default to draft-only; wake-word,
speaker verification, and neural-net entity paths default off. Each of
these flips from "annoying" to "harmful" silently, so the safe state is
off until explicitly enabled per deployment.

## 10. No silent drops on safety paths

Claim failures (Bluesky/Mastodon), `matched-but-unanswered` daemon
states, post-login and ASR background failures, and inner-speech
reflection errors all log at warning or above. A safety signal that
fails quietly is worse than no signal — it reports coverage that
doesn't exist.

## 11. Social autopost stays draft-only until per-lane approval

All autopost/autodraft lanes default off, and photo autopost additionally
requires `human_approved=true` in the draft (defense in depth). Flipping
any lane to auto-publish is a per-platform product decision with
spam/abuse consequences — never a drive-by code change. Weekly drafts
may generate (WEEKLY_SOCIAL_AUTODRAFT defaults on); publishing never
happens without a human.

## 12. Fly paths default off in code, live by yaml on this deployment

Every `MEMORY_FLY*_MODE` / `AIKO_FLY_*` flag defaults to off/shadow in
code (safe for fresh clones on 8GB boxes), while `config/memory.yaml`
sets them live here (`load_config` flattens yaml into environ). This
split is intentional: heaviness is opt-in per deployment, never
assumed. Do not "simplify" by changing code defaults to live — that
would light up the full connectome runtime on every fresh install.
