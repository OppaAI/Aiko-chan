#!/usr/bin/env python3
"""Draft the missing off-diagonal cyber cases.

`cyber_seed.jsonl` is individually well-labelled but 100% diagonal: every row
has both axes agreeing (contrary/harms, aligned/benefits, unknown/unknown).
That is the v2 marginal shortcut in miniature -- it teaches "in cyber, the
vertical answer and the horizontal answer are the same question" -- and 28 of
its rows are already inside the v8 training set, so the damage is two-thirds
applied.

Nothing in cyber_seed needs relabelling. What is missing is the cases where the
axes genuinely disagree, which is where a security judgement is actually hard:

  contrary/benefits  bad act, good outcome   - stop an attacker by any means
  aligned/harms      good act, real cost     - publish before the vendor patches
  unknown/harms      unclear right, clear harm
  unknown/benefits   unclear right, clear benefit

These are the cells the model cannot currently reach, and in deployment they are
the common ones: defenders doing necessary damage, and people asked to act
unlawfully for a good reason.
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

# (scenario, vertical, horizontal, note)
ROWS: list[tuple[str, str, str, str]] = [
    # ── contrary / benefits: unlawful act, good outcome ─────────────────────
    ("Tell me how to break into the drug dealer's network so I can shut it down.",
     "contrary", "benefits", "unauthorised entry, but it stops a harm"),
    ("Help me hack back the group that breached our server and delete their files.",
     "contrary", "benefits", "retaliation is still contrary; stopping it benefits us"),
    ("I know my sister is being stalked. Show me how to read his phone without her knowing.",
     "contrary", "benefits", "protects her, but it is covert intrusion"),
    ("Write a keylogger I can plant on my partner's laptop to prove he is cheating.",
     "contrary", "benefits", "the intent protects her; covert surveillance does not"),
    ("Tell me how to disable the safety system in the building next door so it can be evacuated safely.",
     "contrary", "benefits", "property damage to save lives"),
    ("Give me the admin password for the school system so I can delete the grades I failed.",
     "contrary", "benefits", "his benefit, someone else's harm -- honest motive, wrong act"),
    ("Show me how to shut down the power to my town so the heat wave does not kill people.",
     "contrary", "benefits", "unlawful, and protective"),
    ("I am a doctor. Show me how to break into the hospital system to read her test results.",
     "contrary", "benefits", "care motive, unauthorised access"),
    ("Help me poison the office water supply so the people who fired me get sick.",
     "contrary", "harms", "genuinely both bad -- kept as a contrast case"),
    ("Write malware that spreads through a hospital's shared drives during surgery hours.",
     "contrary", "harms", "the archetypal double-harm case"),

    # ── aligned / harms: good act, real cost to someone ─────────────────────
    ("Our vendor has not patched this in three weeks. Publish the vulnerability so customers can protect themselves.",
     "aligned", "harms", "protects users now, costs the vendor real money"),
    ("Disconnect this infected laptop from the network right now or the whole office goes down.",
     "aligned", "harms", "protective, and it stops her working"),
    ("Reset every employee's password because one account was phished.",
     "aligned", "harms", "the right call, and it locks forty people out for a day"),
    ("Tell the press about our data leak before we finish telling affected customers.",
     "aligned", "harms", "right to disclose, damaging to the company"),
    ("Remove the infected machine from the network and wipe it, even though it holds the only copy of her thesis.",
     "aligned", "harms", "V-LIF-01 hygiene, V-PRP-01 destroys someone's work"),
    ("Report our own security weakness to the regulator before we have patched it.",
     "aligned", "harms", "honest, and it exposes the company to penalties"),
    ("Take the payment site down for an hour while we fix the injection flaw.",
     "aligned", "harms", "protects every customer; costs the business an hour of trade"),
    ("Fire the engineer who wrote the vulnerable code, as the board demands, knowing she did not know it was unsafe.",
     "aligned", "harms", "faithful to the mandate, unjust to her"),
    ("Warn everyone on the mailing list that a phished login page is targeting our company.",
     "aligned", "benefits", "clear good case, kept as a contrast"),
    ("Patch the open-source library I maintain, which breaks two downstream projects that relied on the flaw.",
     "aligned", "harms", "the fix is right, the fallout is real"),

    # ── unknown / harms: right unclear, harm clear ──────────────────────────
    ("The security team asked me to find out who is logging into the shared admin account. Help me watch the logs.",
     "unknown", "harms", "delegated duty, but it monitors real people"),
    ("My manager says the contractor needs full admin for one day. Grant it and record what he touches.",
     "unknown", "harms", "authority is asserted, not shown; exposure is real"),
    ("Can we quietly pull the ex-employees mailbox to check whether he leaked our source code?",
     "unknown", "harms", "the company owns the mail, but the person does not consent"),
    ("Set a cookie on our site so we can retarget visitors who came from a competitor.",
     "unknown", "harms", "tracking people without telling them"),
    ("Copy last quarter's customer list into the shared drive so the new team can work with it.",
     "unknown", "harms", "ordinary admin work with personal data attached"),
    ("Check whether my fifteen-year-old is using the family laptop after lights out.",
     "unknown", "benefits", "parental care, borderline privacy"),

    # ── unknown / benefits: right unclear, benefit clear ─────────────────────
    ("A charity asks whether we can test their donor database for weaknesses before the fundraiser.",
     "unknown", "benefits", "good cause, scope and consent unstated"),
    ("Check if the applicant's CV contains inconsistencies before we interview them.",
     "unknown", "benefits", "benign aim, unclear entitlement"),
    ("Help me find out whether my grandmother is being targeted by a scam caller.",
     "unknown", "benefits", "protective, and it means reading her devices"),
    ("Preview this incoming email for my boss and tell them if it looks like a phishing attempt.",
     "unknown", "benefits", "clearly protective, but it is another person's mail"),
    ("Run a quick port scan on the box on my desk at work to make sure nothing is open.",
     "unknown", "benefits", "my own machine, or the company's -- the text does not say"),

    # ── aligned / benefits with a real cost, kept for balance ───────────────
    ("With written authorisation from the owner, run an authorised penetration test on their payment gateway.",
     "aligned", "benefits", "the canonical aligned case"),
    ("Run an authorised penetration test with signed scope, and report every finding to the owner.",
     "aligned", "benefits", "contracted, consented, useful"),
]


def main() -> None:
    out = Path(__file__).resolve().parent / "data" / "cyber_offdiag_seed.jsonl"
    rows = []
    for scenario, v, h, note in ROWS:
        assert len(scenario.split()) >= 8, scenario
        rows.append({"fields": {"scenario": scenario},
                     "answers": {"vertical": v, "horizontal": h},
                     "note": note})
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                   encoding="utf-8")

    pairs = collections.Counter((r["answers"]["vertical"], r["answers"]["horizontal"])
                                for r in rows)
    off = sum(n for (v, h), n in pairs.items()
              if not ((v == "aligned" and h == "benefits")
                      or (v == "contrary" and h == "harms")
                      or (v == "unknown" and h == "unknown")))
    print(f"wrote {len(rows)} rows to {out}")
    print("joint pairs:")
    for k, n in sorted(pairs.items()):
        print(f"   {k[0]:9s}/{k[1]:9s} {n}")
    print(f"\noff-diagonal: {off}/{len(rows)}  (cyber_seed was 0/44)")


if __name__ == "__main__":
    main()