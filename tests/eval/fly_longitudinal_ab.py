"""Longitudinal A/B: Fly OFF vs LIVE over a scripted 40-turn conversation.

Unlike `fly_stage3_ab.py` (one-shot path check), this simulates a realistic
arc — teach avoid, multi-turn recall with paraphrases, praise/correction,
STOP interrupt — and reports behavioral metrics:

  tart_shown_rate      fraction of recall turns with a tart candidate shown
  paraphrase_shown     non-literal tart variants shown (needs embeddings;
                       reported, relaxed when embedder unreachable)
  inversions           turns where a tart outranks every unrelated item
  false_suppression    unrelated items scoring below baseline
  diversity            unique shown motifs / shown slots
  interrupt_ok         STOP flagged by GF
  praise_taught / correction_taught / eligibility_credited

Run (separate processes, like the stage-3 harness):
  MEMORY_FLYMB_MODE=off MEMORY_FLYGF_MODE=off python -m tests.eval.fly_longitudinal_ab
  MEMORY_FLYMB_MODE=live MEMORY_FLYGF_MODE=live python -m tests.eval.fly_longitudinal_ab

Hermetic: user state + plasticity DB are redirected to a temp dir, and all
fly identities use a fresh uid, so runs never contaminate the device or
each other. No LLM, network (except best-effort local embedder), or
Needle servers required.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
import uuid

TURNS = 40

TART_LITERAL = [
    "we ate fruit tarts yesterday",
    "I remember the fruit tarts",
]
TART_PARAPHRASE = [
    "the birthday pastry was delicious",
    "that June dessert you bought",
    "the pastry from your birthday",
]
UNRELATED = [
    "the garden was quiet",
    "meeting notes for Thursday",
    "the train arrives at noon",
    "she repaired the old radio",
]


def _setup_hermetic() -> str:
    tmp = tempfile.mkdtemp(prefix="fly-long-ab-")
    uid = f"long-ab-{uuid.uuid4().hex[:8]}"
    os.environ["USER_STATE_ROOT"] = tmp
    os.environ["AIKO_USER_STATE_ROOT"] = tmp
    os.environ["USER_SPACE_ROOT"] = tmp
    try:
        from system import userspace

        _real = userspace.user_state_dir

        def _tmp_state(user_id=None):
            import pathlib
            p = pathlib.Path(tmp) / (str(user_id or "default"))
            p.mkdir(parents=True, exist_ok=True)
            return str(p)

        userspace.user_state_dir = _tmp_state  # type: ignore[attr-defined]
    except Exception:
        pass
    return uid


def main() -> int:
    uid = _setup_hermetic()

    from cognition.flymemory.online_teach import teach_from_user_text
    from cognition.flymemory.teach_api import teach_preference
    from cognition.memory.diversity import diversify
    from cognition.memory.fly_rank import adjust_recall_score
    from cognition.memory.preference_store import preference_delta
    from cognition.fly_behavior.giant_fiber import assess_interrupt

    modes = {k: os.getenv(k, "off") for k in ("MEMORY_FLYMB_MODE", "MEMORY_FLYGF_MODE")}
    mb_live = modes.get("MEMORY_FLYMB_MODE") == "live"

    # Embedder probe (one short call): paraphrase suppression via cosine
    # only fires when the local embedding server is reachable. Report it
    # so the verdict stays honest on boxes without the server.
    embed_available = False
    try:
        from cognition.memory.vecstore import HarrierEmbedder
        list(HarrierEmbedder(timeout=2.0).embed(["harness probe"]))
        embed_available = True
    except Exception:
        embed_available = False

    metrics: dict = {
        "turns": TURNS,
        "tart_turns": 0,
        "tart_shown": 0,
        "paraphrase_turns": 0,
        "paraphrase_shown": 0,
        "inversions": 0,
        "unrelated_below_baseline": 0,
        "unrelated_total": 0,
        "shown_motifs": [],
        "shown_slots": 0,
    }

    # Turns 1-2: explicit avoid teaching (chat phrasing + direct API).
    teach_from_user_text("please avoid fruit tarts", user_id=uid)
    teach_preference("fruit tarts", direction="avoid", user_id=uid)

    for turn in range(3, TURNS + 1):
        lits = [TART_LITERAL[(turn + i) % len(TART_LITERAL)] for i in range(1)]
        paras = [TART_PARAPHRASE[(turn + i) % len(TART_PARAPHRASE)] for i in range(2)]
        cands = lits + paras + list(UNRELATED)
        scored = []
        for c in cands:
            s, _ = adjust_recall_score(1.0, c, user_id=uid, log_influence=False)
            scored.append({"memory": c, "score": s})
        scored.sort(key=lambda r: r["score"], reverse=True)
        # NOTE: diversify() already applies the freshness anti-loop once
        # internally — do not apply it a second time (double application
        # inverts the signal by penalizing whatever the first pass showed).
        shown = [r["memory"] for r in diversify(scored, user_id=uid)[:4]]
        metrics["shown_slots"] += len(shown)
        metrics["shown_motifs"].extend(shown)

        is_tart = [m for m in shown if "tart" in m.lower()]
        is_para = [m for m in shown if m in TART_PARAPHRASE]
        metrics["tart_turns"] += 1
        metrics["paraphrase_turns"] += 1
        if is_tart:
            metrics["tart_shown"] += 1
        if is_para:
            metrics["paraphrase_shown"] += 1
        unrelated_scores = [r["score"] for r in scored if r["memory"] in UNRELATED]
        tart_scores = [r["score"] for r in scored if r["memory"] not in UNRELATED]
        metrics["unrelated_total"] += len(unrelated_scores)
        metrics["unrelated_below_baseline"] += sum(1 for s in unrelated_scores if s < 1.0 - 0.05)
        if tart_scores and unrelated_scores and min(tart_scores) > max(unrelated_scores):
            metrics["inversions"] += 1

        # Mid-arc outcomes: praise + correction exercise delayed credit.
        if turn == 20:
            teach_from_user_text("thanks, perfect!", user_id=uid)
        if turn == 30:
            teach_from_user_text("no, that's wrong, stop saying that", user_id=uid)

    gf = assess_interrupt("STOP right now")
    praise = teach_from_user_text("thanks, perfect!", user_id=uid)
    correction = teach_from_user_text("no, that's wrong", user_id=uid)
    try:
        from cognition.flymemory.eligibility import stats as elig_stats
        elig = elig_stats(uid)
    except Exception:
        elig = {}
    try:
        pdelta = preference_delta("we ate fruit tarts yesterday", user_id=uid)
    except Exception:
        pdelta = 0.0

    tart_rate = metrics["tart_shown"] / max(1, metrics["tart_turns"])
    para_rate = metrics["paraphrase_shown"] / max(1, metrics["paraphrase_turns"])
    uniq = len({m.lower() for m in metrics["shown_motifs"]})
    diversity = uniq / max(1, metrics["shown_slots"])
    report = {
        "modes": modes,
        "embed_available": embed_available,
        "tart_shown_rate": round(tart_rate, 3),
        "paraphrase_shown_rate": round(para_rate, 3),
        "inversions": metrics["inversions"],
        "false_suppression": round(
            metrics["unrelated_below_baseline"] / max(1, metrics["unrelated_total"]), 3
        ),
        "diversity": round(diversity, 3),
        "interrupt_ok": bool(gf.get("interrupt")),
        "praise_taught": bool(praise.get("taught")),
        "correction_taught": bool(correction.get("taught")),
        "eligibility": elig,
        "preference_delta": pdelta,
        "ok": True,
    }
    if mb_live:
        reasons = []
        if tart_rate > 0.35:
            reasons.append(f"tart_shown_rate {tart_rate:.2f} > 0.35")
        if metrics["inversions"] > 0:
            reasons.append(f"inversions {metrics['inversions']} > 0")
        if not gf.get("interrupt"):
            reasons.append("STOP not flagged")
        # Paraphrase suppression via cosine needs the embedder; enforce
        # only when it actually participated, otherwise report the rate.
        if embed_available and para_rate > 0.9:
            reasons.append(f"paraphrase_shown_rate {para_rate:.2f} > 0.9")
        if reasons:
            report["ok"] = False
            report["reason"] = "; ".join(reasons)
    print(json.dumps(report, default=str))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
