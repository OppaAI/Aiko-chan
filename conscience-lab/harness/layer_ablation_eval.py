#!/usr/bin/env python3
"""Layer-ablation eval: L0+L1+Laya vs Laya-only vs L0+L1-only.

Answers: is the deterministic part of the ladder (L0 reflex + L1
allowlist/canon) too strict on its own? Measures FP% (benign refused) and
FN% (harmful allowed) per configuration on a labeled situation set.

Configs:
  full      L0 + L1 + L2 Laya judge (production ladder; L3 needs an llm_client
            so it never runs here)
  laya_only L0/L1a/L1b neutralized; the Laya SLM decides alone
  det_only  tier-2 made unavailable; deterministic layers decide
            (L0 reflex, L1a allowlist, L1b lexical via the L2 blend)

Layer control is script-local monkeypatching; production code is untouched.

Needs the Laya server (default http://localhost:8093) for `full` and
`laya_only`. The script probes it up front and refuses to run rather than
silently measuring lexical-only.

Usage (on the host where :8093 is live, from the repo root):
  python3 conscience-lab/harness/layer_ablation_eval.py \
      --set conscience-lab/data/laya_phase3_round2_variants_v3.jsonl

Output: per-config FP/FN, decision distribution, refusing-layer
attribution, and a per-case JSONL for drill-down.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from cognition.conscience import conscience_for  # noqa: E402
from cognition.conscience import allowlist as allowlist_mod  # noqa: E402
from cognition.conscience import canon as canon_mod  # noqa: E402
from cognition.conscience import judge as judge_mod  # noqa: E402
from cognition.conscience.laya_judge import get_laya  # noqa: E402
from cognition.conscience.schema import ALLOW, Verdict  # noqa: E402


def _probe_laya():
    """Probe the Laya server once; return the live singleton.

    The returned instance is pinned as the tier-2 judge for the Laya
    configs so a mid-run outage cannot silently swap in the SLM fallback.
    """
    laya = get_laya()
    if not laya.available:
        sys.exit("REFUSING: Laya judge not configured (no base URL).")
    probe = laya.score("<situation>probe: is it raining?</situation>", "", [])
    if probe is None:
        sys.exit(
            "REFUSING: Laya server not answering at "
            f"{laya._base_url} — refusing to measure 'Laya' configs "
            "that would silently degrade to lexical-only."
        )
    print(f"[probe] Laya answering at {laya._base_url}", flush=True)
    return laya


class _Patches:
    """Apply/revert script-local monkeypatches for one config."""

    def __init__(self):
        self._saved: list[tuple] = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def revert(self):
        for obj, name, value in reversed(self._saved):
            setattr(obj, name, value)
        self._saved.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.revert()


def _neutralize_l0_l1(circuit, patches: _Patches) -> None:
    """Config `laya_only`: L0/L1a/L1b contribute nothing; the SLM decides."""

    def _allow_l0(act, text, ctx):
        return Verdict(decision=ALLOW, gate="eval-skip-l0", layers_run=["layer0"])

    patches.set(circuit, "_layer0", _allow_l0)
    patches.set(allowlist_mod, "known_safe_class", lambda text: None)
    store = canon_mod.get_canon()
    patches.set(store, "retrieve", lambda text, embedder=None: [])


def _disable_tier2(patches: _Patches) -> None:
    """Config `det_only`: deterministic layers decide; no SLM."""

    class _Unavailable:
        available = False

    patches.set(judge_mod, "get_tier2", lambda: _Unavailable())


def _load_cases(path: str) -> list[dict]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            expect = str(d.get("expect", "")).lower()
            if expect not in ("refuse", "allow"):
                continue
            cases.append(
                {
                    "id": d.get("id", f"case-{len(cases)}"),
                    "situation": d.get("situation", ""),
                    "expect_refuse": expect == "refuse",
                }
            )
    return cases


class _LayaDown(Exception):
    """Laya became unavailable mid-run; the config cannot be measured."""


def _run_config(circuit, cases: list[dict], config: str, act: str, laya=None) -> list[dict]:
    results = []
    needs_laya = config in ("full", "laya_only")
    with _Patches() as patches:
        # Never pollute the production ledger with eval traffic.
        patches.set(circuit, "_record", lambda *a, **k: None)
        if config == "laya_only":
            _neutralize_l0_l1(circuit, patches)
        elif config == "det_only":
            _disable_tier2(patches)
        elif config != "full":
            raise ValueError(f"unknown config {config!r}")

        scored = {"called": False, "ok": False}
        if needs_laya:
            # Pin the tier-2 judge to the probed Laya instance. Without this,
            # a mid-run Laya outage makes get_tier2() silently return the SLM
            # fallback and the config would measure the wrong judge.
            patches.set(judge_mod, "get_tier2", lambda: laya)
            orig_score = laya.score

            def _tracked_score(*a, **k):
                scored["called"] = True
                res = orig_score(*a, **k)
                scored["ok"] = res is not None
                return res

            patches.set(laya, "score", _tracked_score)

        for case in cases:
            scored["called"] = False
            scored["ok"] = False
            t0 = time.monotonic()
            try:
                verdict = circuit.evaluate(
                    act=act,
                    content=case["situation"],
                    context={"surface": "eval-layer-ablation"},
                    surface="eval",
                )
                # The circuit converts internal crashes to CAUTION verdicts
                # with fail_mode set; those are unmeasured cases, not
                # allow/refuse outcomes.
                err = verdict.fail_mode if verdict and verdict.fail_mode else ""
            except Exception as exc:  # noqa: BLE001 - record, don't stop the run
                verdict = None
                err = f"{type(exc).__name__}: {exc}"
            dt_ms = int((time.monotonic() - t0) * 1000)
            if needs_laya and not laya.available:
                # Laya died mid-run: every later case would measure the
                # lexical fallback, so the config is unmeasurable. Abort it;
                # main() still runs configs that don't need Laya.
                raise _LayaDown(
                    f"Laya became unavailable mid-run (config={config}, "
                    f"case={case['id']})"
                )
            if needs_laya and scored["called"] and not scored["ok"]:
                # Laya was consulted but returned no score: the ladder fell
                # back to lexical for this case. Unmeasured, not a valid
                # FP/FN outcome.
                err = err or "laya-no-score"
            results.append(
                {
                    "id": case["id"],
                    "config": config,
                    "expect_refuse": case["expect_refuse"],
                    "decision": verdict.decision if verdict else "error",
                    # blocked covers REFUSE and ESCALATE; a None verdict
                    # (harness-level exception) is never a refusal.
                    "refused": verdict.blocked if verdict else False,
                    "gate": verdict.gate if verdict else "",
                    "layers_run": list(verdict.layers_run) if verdict else [],
                    "latency_ms": dt_ms,
                    "error": err,
                }
            )
    return results


def _summarize(results: list[dict]) -> dict:
    # Failed evaluations (circuit errors, harness exceptions, Laya no-score)
    # are unmeasured cases: they must not dilute FP/FN either way.
    valid = [r for r in results if not r["error"]]
    failed = [r["id"] for r in results if r["error"]]
    harm = [r for r in valid if r["expect_refuse"]]
    ben = [r for r in valid if not r["expect_refuse"]]
    fn = sum(1 for r in harm if not r["refused"])
    fp = sum(1 for r in ben if r["refused"])
    decisions: dict[str, int] = {}
    refusing_gates: dict[str, int] = {}
    for r in valid:
        decisions[r["decision"]] = decisions.get(r["decision"], 0) + 1
        if r["refused"]:
            refusing_gates[r["gate"]] = refusing_gates.get(r["gate"], 0) + 1
    lat = [r["latency_ms"] for r in valid]
    lat.sort()
    p50 = lat[len(lat) // 2] if lat else 0
    return {
        "n": len(valid),
        "n_errors": len(failed),
        "error_case_ids": failed,
        "n_harmful": len(harm),
        "n_benign": len(ben),
        "FN_pct": round(100.0 * fn / len(harm), 1) if harm else None,
        "FP_pct": round(100.0 * fp / len(ben), 1) if ben else None,
        "decisions": decisions,
        "refusing_gates": refusing_gates,
        "latency_ms_p50": p50,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", required=True, help="labeled jsonl (situation/expect)")
    ap.add_argument(
        "--configs",
        default="full,laya_only,det_only",
        help="comma-separated subset of configs to run",
    )
    ap.add_argument("--act", default="respond", help="conscience act under test")
    ap.add_argument("--limit", type=int, default=0, help="only first N cases (smoke test)")
    ap.add_argument("--out", default="", help="write per-case JSONL here")
    args = ap.parse_args()

    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    laya = _probe_laya() if any(c in ("full", "laya_only") for c in configs) else None

    cases = _load_cases(args.set)
    if args.limit:
        cases = cases[: args.limit]
    n_h = sum(1 for c in cases if c["expect_refuse"])
    print(f"[eval] {len(cases)} cases ({n_h} harmful, {len(cases)-n_h} benign), "
          f"configs={configs}, act={args.act}", flush=True)

    circuit = conscience_for("eval-layer-ablation")
    all_results: list[dict] = []
    summaries: dict[str, dict] = {}
    laya_down = False
    for config in configs:
        if laya_down and config in ("full", "laya_only"):
            print(f"[eval] config={config} SKIPPED (Laya down)", flush=True)
            summaries[config] = {"aborted": "laya-down"}
            continue
        t0 = time.monotonic()
        try:
            results = _run_config(circuit, cases, config, args.act, laya=laya)
        except _LayaDown as exc:
            print(f"[eval] config={config} ABORTED: {exc}", flush=True)
            summaries[config] = {"aborted": str(exc)}
            laya_down = True
            continue
        summaries[config] = _summarize(results)
        summaries[config]["wall_s"] = round(time.monotonic() - t0, 1)
        all_results.extend(results)
        print(f"[eval] config={config} done in {summaries[config]['wall_s']}s", flush=True)

    print("\n" + json.dumps(summaries, indent=2))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"[eval] per-case results -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
