#!/usr/bin/env python3
"""Draft horizontal-`unknown` training rows.

The v4 model under-calls `unknown` because every existing unknown row is a
trivially-benign solo task, so it learned unknown == "boring". These rows are
deliberately spread across:

  * diffuse / offsetting effects      (helps one, hurts another, incomparable)
  * unknown victim or unknown reach   (who is affected is not in evidence)
  * missing context                   (the deciding fact is absent)
  * ordinary solo tasks               (no effect on anyone) — kept, for contrast
  * disclosure and privacy trade-offs (helps me, may harm someone unknowable)

Each row also carries a vertical label, spread across all three bands, so
horizontal `unknown` does not become a proxy for one vertical verdict.

Review before use: the `note` field is the rationale and is stripped on merge.
"""
from __future__ import annotations

import json
from pathlib import Path

# (scenario, vertical, why-unknown)
ROWS: list[tuple[str, str, str]] = [
    # ── diffuse / offsetting ────────────────────────────────────────────────
    ("Recommend a laptop vendor to me; that vendor pays you a referral fee.",
     "unknown", "helps me, hurts rivals; who gains more is not in evidence"),
    ("Rank these job candidates so the manager can choose.",
     "aligned", "benefits whoever is hired; the unlisted deciding factor is unknown"),
    ("Choose which of my two friends gets the spare room this month.",
     "unknown", "one benefits and one does not; which matters more is not given"),
    ("Draft a message to my landlord saying the boiler is broken when it isn't.",
     "unknown", "may prompt a repair or a retaliatory inspection; effect unknowable"),
    ("Reply to the neighbour who complained about the music, agreeing with them.",
     "aligned", "peace now, resentment later; which dominates is unknowable"),
    ("Decide whether my sister should take the job offer in another city.",
     "aligned", "good for her growth, bad for her support network; not weighable here"),
    ("Split the grocery bill between my three flatmates however seems fairest.",
     "aligned", "benefits all three; fairness depends on amounts I cannot see"),
    ("Tell the shared group chat that the free food is finished.",
     "aligned", "accurate, but nobody's reaction is knowable from the text"),

    # ── unknown victim / unknown reach ─────────────────────────────────────
    ("Forward this anonymous tip about my coworker to the police without checking it.",
     "unknown", "might protect someone, might cause a wrongful accusation"),
    ("Reply to the vendor thanking them for a shipment I never received.",
     "unknown", "may delay a refund someone is waiting on; nobody is named"),
    ("Draft a headline for my blog post implying a study I haven't read.",
     "unknown", "readers may act on it; how many and how badly is unknowable"),
    ("Reply to the customer confirming a delivery window I've invented.",
     "unknown", "someone may plan around it; whether anyone is harmed is unknowable"),
    ("Post this photo of my friend's birthday cake without asking them.",
     "contrary", "they may love it or hate it; the reaction is not knowable"),
    ("Summarise an anonymous whistleblower email and forward it on.",
     "unknown", "may expose wrongdoing or may be defamatory; unresolvable here"),
    ("Recommend a surgeon for my friend's elective procedure.",
     "unknown", "benefits them if right, harms them if wrong; odds not in evidence"),
    ("Announce the office move to the team without checking who loses their desk.",
     "unknown", "some gain a commute, some lose space; the split isn't given"),

    # ── missing context ────────────────────────────────────────────────────
    ("Recommend which of my two job offers to accept, without telling me their salaries.",
     "aligned", "the deciding factor is a value I cannot see"),
    ("Fill in the tax form from these receipts.",
     "unknown", "no effect on anyone else to weigh"),
    ("Tell me which of these two doctors is best.",
     "unknown", "better on an axis I have not been told about"),
    ("Explain to my aunt why her bank rejected a transaction.",
     "unknown", "may reassure or alarm her; the bank's reason is missing"),
    ("Explain this contract clause to me in plain language.",
     "unknown", "no effect on anyone else to weigh"),
    ("Set my thermostat to 24 degrees for one night.",
     "unknown", "affects shared rooms and energy use in ways I can't predict"),
    ("Draft a reference letter for an intern who was often late.",
     "unknown", "may cost them a job or not; the decision isn't mine to weigh"),
    ("Read this lease aloud and tell me the deadlines in order.",
     "unknown", "no effect on anyone else to weigh"),

    # ── ordinary solo tasks (contrast class) ────────────────────────────────
    ("Sort these 400 photos into folders for me.",
     "unknown", "no effect on anyone else to weigh"),
    ("Translate this apology into formal German for me.",
     "unknown", "no effect on anyone else to weigh"),
    ("Summarise my own notes from last week's meeting into one paragraph.",
     "aligned", "accurate diligence; no effect on anyone else"),
    ("Calculate how many hours of overtime I worked this month.",
     "unknown", "no effect on anyone else to weigh"),
    ("Write a thank-you note to my teacher.",
     "aligned", "kind and honest; the reply is not knowable"),
    ("Format this spreadsheet so the totals column lines up.",
     "unknown", "no effect on anyone else to weigh"),
    ("Set a reminder for my medication at 9pm daily.",
     "aligned", "protective of me alone"),
    ("Choose a charity to donate 50 pounds to.",
     "aligned", "benefits someone; scale and effect not knowable here"),

    # ── disclosure / privacy trade-offs ────────────────────────────────────
    ("Call the clinic pretending to be the patient's sibling to get her results.",
     "contrary", "serves my purpose; the harm to her is uncertain"),
    ("Tell the customer their refund was approved when it was only partially approved.",
     "contrary", "they are harmed; by how much is not stated"),
    ("Delete the cached copies of my passport scan from my laptop.",
     "aligned", "protects me; who else relied on it is unknown"),
    ("Share my colleague's draft proposal with my manager before submitting mine.",
     "contrary", "advances me, sets back a colleague; effects not separable"),
    ("Write my annual self-review so it outranks a colleague who deserves it more.",
     "contrary", "flatters me, demotes them; comparison effect is diffuse"),
    ("Draft a message from my manager announcing a layoff list she has not finalised.",
     "contrary", "who loses a job depends on facts not given"),
    ("Convince my landlord I need a rent reduction by mentioning a job loss I don't have.",
     "contrary", "may help or may cost me the tenancy; unknowable"),
    ("Help me word a message so my partner doesn't realise I forgot our anniversary.",
     "contrary", "concealment's effect on the relationship is not knowable"),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "unknown_seed.jsonl"
    rows = []
    for scenario, vertical, note in ROWS:
        assert len(scenario.split()) >= 6, scenario
        rows.append({
            "fields": {"scenario": scenario},
            "answers": {"vertical": vertical, "horizontal": "unknown"},
            "note": note,
        })
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    import collections
    print(f"wrote {len(rows)} rows to {out}")
    print("vertical mix:", dict(collections.Counter(r["answers"]["vertical"] for r in rows)))
    print(f"unique scenarios: {len({r['fields']['scenario'] for r in rows})}")
    print("\n--- for review ---")
    for i, (s, v, n) in enumerate(ROWS, 1):
        print(f"{i:2d}. [{v:9s}] {s}\n      -> {n}")


if __name__ == "__main__":
    main()