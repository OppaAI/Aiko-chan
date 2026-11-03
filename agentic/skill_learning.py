"""Machine-written skill proposal helpers for observe -> distill -> reuse loops.

Trusted skillsets remain human-owned. This module only writes reviewable drafts
under the active user's workspace so successful/failing workflows can be
promoted deliberately later.

Promotion (promote_skill_proposal) is human-gated by design: it never writes
into the trusted agentic/skillsets/ tree. It parses an existing proposal draft
and writes a candidate skillset JSON into workspace/skillsets_staging/ for a
person to review and copy into agentic/skillsets/ by hand (or via CLI).
"""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

from system.log import get_logger
from system.userspace import user_workspace_root

log = get_logger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Per-step args are capped independently of final_preview, since a step's
# args can carry a full tool-call payload (report bodies, social post text,
# ingested document text). Without this cap, repeated observed runs against
# the same goal slug make the proposal file grow without bound.
_STEP_ARGS_MAX_CHARS = 500

# Bound the proposal file itself: a goal slug that gets hit repeatedly
# (e.g. a recurring scheduled job) would otherwise append one "Observed
# run" block per invocation forever. Keep only the most recent N blocks —
# older observed-run history has diminishing review value once a newer
# run against the same goal exists, and an unbounded file is both a disk
# and a review-burden risk.
_MAX_OBSERVED_RUNS_PER_FILE = 20


def _slug(text: str) -> str:
    return _SLUG_RE.sub("-", (text or "skill").lower()).strip("-")[:80] or "skill"


def skill_proposal_dir(user_id: str | None = None) -> Path:
    path = user_workspace_root(user_id) / "skill_proposals"
    path.mkdir(parents=True, exist_ok=True)
    return path


def skillset_staging_dir(user_id: str | None = None) -> Path:
    """Human-review staging area for promoted skill proposals. Never written
    to automatically by the agent loop — only promote_skill_proposal (below)
    writes here, and only when explicitly invoked (never as an implicit
    side effect of a normal chat/agentic turn)."""
    path = user_workspace_root(user_id) / "skillsets_staging"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _truncated_step(step: dict[str, Any]) -> dict[str, Any]:
    """Copy a step record with its args stringified and capped so a large
    tool-call payload can't blow up the proposal file size."""
    return {**step, "args": str(step.get("args"))[:_STEP_ARGS_MAX_CHARS]}


_OBSERVED_RUN_BLOCK_RE = re.compile(
    r"\n## Observed run \((?:success|failure)\)\n.*?(?=\n## Observed run \(|\Z)",
    re.DOTALL,
)


def _trim_observed_runs(text: str, max_runs: int = _MAX_OBSERVED_RUNS_PER_FILE) -> str:
    """Keep only the most recent `max_runs` '## Observed run (...)' blocks.

    Everything before the first such block (the header written on file
    creation) is preserved untouched.
    """
    blocks = _OBSERVED_RUN_BLOCK_RE.findall(text)
    if len(blocks) <= max_runs:
        return text
    head_end = text.find("\n## Observed run (")
    head = text[:head_end] if head_end != -1 else text
    kept = blocks[-max_runs:]
    return head + "".join(kept)


def propose_skill_from_run(
    goal: str,
    steps: list[dict[str, Any]],
    final_text: str,
    *,
    verified_ok: bool,
    score: float,
    user_id: str | None = None,
) -> Path | None:
    """Draft or refine a reviewable skill proposal from an agentic run.

    Successes capture reusable tool order. Failures append avoid-notes to the
    same proposal path so repeated mistakes refine the draft without modifying
    trusted `agentic/skillsets/` files.
    """
    if len(steps or []) < 2:
        return None

    path = skill_proposal_dir(user_id) / f"{_slug(goal)}.md"
    tools = [str(s.get("tool")) for s in steps if s.get("tool")]
    status = "success" if verified_ok and score >= 0.7 else "failure"
    header = f"# Skill Proposal: {goal[:120]}\n\n"

    if not path.exists():
        path.write_text(header + "Status: draft_review_required\n\n", encoding="utf-8")

    block = {
        "ts": time.time(),
        "status": status,
        "score": score,
        "tools": tools,
        "steps": [_truncated_step(s) for s in steps],
        "final_preview": (final_text or "")[:1000],
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## Observed run ({status})\n\n")
        if status == "success":
            f.write("Reusable tool order: " + " -> ".join(tools) + "\n\n")
        else:
            f.write("Avoid / failed approach notes: review this trace before promotion.\n\n")
        f.write("```json\n" + json.dumps(block, ensure_ascii=False, indent=2, default=str) + "\n```\n")

    # Rotate: cap the number of observed-run blocks kept on disk so a
    # recurring goal slug can't grow this file without bound.
    try:
        current = path.read_text(encoding="utf-8")
        trimmed = _trim_observed_runs(current)
        if trimmed != current:
            path.write_text(trimmed, encoding="utf-8")
    except OSError as exc:
        log.warning("Failed trimming skill proposal %s: %s", path, exc)

    return path


# ── promotion ──────────────────────────────────────────────────────────────
# Human-gated: this function is never called automatically by the agentic
# loop. It's exposed for a person (via CLI, or an explicit needs_approval
# agent tool wrapper elsewhere) to turn a reviewed proposal into a candidate
# skillset. It never writes into agentic/skillsets/ — only into
# workspace/skillsets_staging/, where a person still has to copy it into the
# trusted tree by hand.
#
# Explicit validation rules for promotion to a VALIDATED DAG live in
# validate_dag_promotion below. A proposal becomes a validated DAG only when
# every rule passes: the path must be repeated (several successful runs with
# the same tool order), recent (latest run a success), and clean (every
# success scored >= 0.7). One-off or shaky runs stay as drafts no matter who
# asks.
_DAG_PROMOTE_MIN_SUCCESSES = int(os.getenv("DAG_PROMOTE_MIN_SUCCESSES", "3"))
_DAG_PROMOTE_MIN_SCORE = float(os.getenv("DAG_PROMOTE_MIN_SCORE", "0.7"))

_SUCCESS_BLOCK_RE = re.compile(
    r"## Observed run \(success\)\n\nReusable tool order: (?P<order>.+?)\n\n"
    r"```json\n(?P<json>.+?)\n```",
    re.DOTALL,
)


# Matches every observed-run block (success or failure) so the validator can
# see the full recent history, not just successes.
_OBSERVED_RUN_RE = re.compile(
    r"## Observed run \((?P<status>success|failure)\)\n\n"
    r"(?:Reusable tool order: (?P<order>.+?)\n\n|Avoid / failed approach notes:[^\n]*\n\n)"
    r"```json\n(?P<json>.+?)\n```",
    re.DOTALL,
)


def _observed_runs(proposal_text: str) -> list[dict[str, Any]]:
    """Parse every observed-run block in a proposal file, oldest first."""
    runs = []
    for m in _OBSERVED_RUN_RE.finditer(proposal_text):
        try:
            block = json.loads(m.group("json"))
        except Exception:
            block = {}
        runs.append({
            "status": m.group("status"),
            "order": (m.group("order") or "").strip(),
            "score": block.get("score"),
            "tools": block.get("tools") or [],
        })
    return runs


def validate_dag_promotion(slug: str, user_id: str | None = None) -> tuple[bool, list[str]]:
    """Explicit validation rules for promoting a proposal to a validated DAG.

    Returns (ok, reasons). A proposal becomes a validated DAG only when:
      1. it exists and has observed runs;
      2. at least _DAG_PROMOTE_MIN_SUCCESSES successful runs were observed
         (the path is *repeated*, not a one-off);
      3. every successful run scored >= _DAG_PROMOTE_MIN_SCORE (clean);
      4. all successful runs share the same tool order (a stable path);
      5. the latest observed run is a success (recent — no unresolved
         failure after the last success).
    """
    reasons: list[str] = []
    proposal_path = skill_proposal_dir(user_id) / f"{slug}.md"
    if not proposal_path.exists():
        return False, [f"no skill proposal found for slug={slug!r}"]
    try:
        text = proposal_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, [f"could not read proposal: {exc}"]
    runs = _observed_runs(text)
    if not runs:
        return False, ["proposal has no observed runs yet"]
    successes = [r for r in runs if r["status"] == "success"]
    if len(successes) < _DAG_PROMOTE_MIN_SUCCESSES:
        reasons.append(
            f"only {len(successes)} successful observed run(s); "
            f"need >= {_DAG_PROMOTE_MIN_SUCCESSES} for a validated DAG"
        )
    low = [r for r in successes
           if not isinstance(r["score"], (int, float)) or float(r["score"]) < _DAG_PROMOTE_MIN_SCORE]
    if low:
        reasons.append(
            f"{len(low)} successful run(s) scored below {_DAG_PROMOTE_MIN_SCORE:.2f}"
        )
    orders = {r["order"] for r in successes if r["order"]}
    if len(orders) > 1:
        reasons.append(
            f"successful runs disagree on tool order ({len(orders)} variants); "
            "not a stable repeated path"
        )
    if runs[-1]["status"] != "success":
        reasons.append("latest observed run was a failure — resolve it before promotion")
    return (not reasons), reasons


def _latest_success_block(proposal_text: str) -> dict[str, Any] | None:
    matches = list(_SUCCESS_BLOCK_RE.finditer(proposal_text))
    if not matches:
        return None
    match = matches[-1]
    try:
        return json.loads(match.group("json"))
    except json.JSONDecodeError:
        return None


def _promotion_args_for_step(tool: str, step: dict[str, Any]) -> dict[str, Any]:
    """Best-effort placeholder args for a promoted node — mirrors
    agentic.graph_engine's promotion helper of the same name. Kept
    independent (not imported from graph_engine) to avoid a promotion-time
    dependency on the graph engine module; the shapes are intentionally
    aligned so a promoted skillset "reads" the same as a promoted playbook.
    """
    generic_args_by_tool = {
        "make_plan": {"goal": "$prompt"},
        "create_checklist": {"title": "$title", "items": "$heuristic_items"},
        "save_note": {"title": "$title", "content": "$prompt", "folder": "notes"},
        "deep_research": {"query": "$prompt"},
        "adaptive_search": {"query": "$prompt"},
        "synthesize_report": {"evidence": "$prompt", "prompt": "$prompt", "style": "auto"},
        "polish_text": {"evidence": "$prompt", "prompt": "$prompt", "style": "auto"},
        "combine_evidence": {"parts": ["$prompt"], "separator": "\n\n---\n\n"},
        "condense_text": {"text": "$prompt", "query": "$prompt"},
        "kb_search": {"query": "$prompt"},
        "learn_report": {"title": "$title", "text": "$prompt"},
        "write_report": {"title": "$title", "content": "$prompt"},
    }
    if tool in generic_args_by_tool:
        return generic_args_by_tool[tool]
    args_str = step.get("args")
    if isinstance(args_str, str):
        try:
            parsed = json.loads(args_str)
            if isinstance(parsed, dict) and parsed:
                return {str(k): "$prompt" for k in parsed}
        except json.JSONDecodeError:
            log.debug("skill_learning: args_str is not JSON, treating as literal")
    return {}


def promote_skill_proposal(
    slug: str,
    *,
    user_id: str | None = None,
    dry_run: bool = True,
    validated: bool = False,
) -> Path:
    """Read a proposal draft and build a candidate skillset in staging.

    `slug` is the proposal filename stem (without .md), i.e. whatever
    `_slug(goal)` produced when the proposal was written.

    dry_run=True (default): validate and return the WOULD-BE staging path
    without writing it, so a caller (CLI, review UI) can preview first.
    dry_run=False: actually write the candidate skillset JSON to
    workspace/skillsets_staging/<slug>.json and return that path.

    validated=False (default): the candidate is staged as a draft with
    ``review_status: pending_human_review`` — it is NOT a validated DAG.
    validated=True: the proposal must first pass :func:`validate_dag_promotion`;
    on success the candidate is staged with ``review_status: validated_dag``
    and the validation outcome recorded in the payload.

    Raises FileNotFoundError if no proposal exists for `slug`, and
    ValueError if the proposal has no successful observed run yet (a
    proposal with only failures has nothing safe to promote) or, when
    validated=True, if the promotion rules do not pass.
    """
    proposal_path = skill_proposal_dir(user_id) / f"{slug}.md"
    if not proposal_path.exists():
        raise FileNotFoundError(f"no skill proposal found for slug={slug!r} at {proposal_path}")

    text = proposal_path.read_text(encoding="utf-8")
    validation_outcome: dict[str, Any] | None = None
    if validated:
        ok, reasons = validate_dag_promotion(slug, user_id=user_id)
        if not ok:
            raise ValueError(
                f"skill proposal {slug!r} does not meet validated-DAG rules: "
                + "; ".join(reasons)
            )
        validation_outcome = {"validated_dag": True, "rules": reasons or ["all rules passed"]}
    block = _latest_success_block(text)
    if block is None:
        raise ValueError(
            f"skill proposal {slug!r} has no successful observed run yet; "
            "nothing safe to promote"
        )

    goal_line = text.splitlines()[0] if text else f"# Skill Proposal: {slug}"
    goal = goal_line.replace("# Skill Proposal:", "").strip() or slug

    steps = block.get("steps") or []
    nodes: list[dict[str, Any]] = []
    for idx, step in enumerate(steps, start=1):
        tool = str(step.get("tool") or "").strip()
        if not tool or tool in {"final_answer", "llm_call"}:
            continue
        node = {"id": f"step_{idx}", "tool": tool, "args": _promotion_args_for_step(tool, step)}
        if nodes:
            node["depends_on"] = [nodes[-1]["id"]]
        nodes.append(node)

    if not nodes:
        raise ValueError(f"skill proposal {slug!r} success block had no promotable tool steps")

    candidate = {
        "id": f"proposed_{slug}",
        "name": goal[:120],
        "source_proposal": str(proposal_path),
        "promoted_at": time.time(),
        "score": block.get("score"),
        "triggers": [],
        "requires_any": [],
        "nodes": nodes,
        "review_status": "validated_dag" if validation_outcome else "pending_human_review",
    }
    if validation_outcome:
        candidate["validation"] = validation_outcome

    staging_path = skillset_staging_dir(user_id) / f"{slug}.json"
    if dry_run:
        return staging_path

    staging_path.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Promoted skill proposal %r to staging: %s", slug, staging_path)
    return staging_path