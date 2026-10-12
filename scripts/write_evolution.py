#!/usr/bin/env python3
"""Write Aiko's monthly evolution lore from git history.

Backfill a past month:
    python3 scripts/write_evolution.py --month 2026-05

Monthly consolidate (run in the first days of a new month, writes the
month before):
    python3 scripts/write_evolution.py --last-month

Gathers the month's merged PRs and feature commits, asks Aiko's LLM to
pick the historically significant features and changes (not minor
fixes), and writes persona/lore/evolution-YYYY-MM.jsonl.

History is immutable: refuses to overwrite an existing month file
unless --force is given.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LORE_DIR = REPO / "persona" / "lore"

PROMPT_TEMPLATE = """You are writing Aiko's history. Below is the git history for {month_label} from the Aiko-chan repo (merged pull requests and feature commits).

Write 2-5 lorebook entries capturing what was historically significant that month: features shipped, systems built, major changes. Skip minor fixes, typo corrections, doc tweaks, dependency bumps, and merge noise.

Each entry is one JSON object per line with:
- "keys": 2-6 trigger words or phrases someone would say when asking about this (e.g. "june 2026", "sqlite-vec", "memory backend")
- "content": 1-2 sentences phrased as a dated historical claim (e.g. "June 4, 2026: ..."). It must stay true forever even if the system is later replaced or removed.
- "enabled": true

Rules:
- Brief and factual. No speculation beyond what the history shows.
- At least one entry must carry the month name as a key (e.g. "june 2026") so "what happened in June?" finds it.
- Output ONLY the JSON lines, no markdown fences, no commentary.

Git history for {month_label}:
```
{history}
```"""


def _git(args: list[str]) -> str:
    out = subprocess.run(
        ["git", "-C", str(REPO)] + args,
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        raise RuntimeError(f"git failed: {out.stderr.strip()}")
    return out.stdout


def gather_month(year: int, month: int, cap: int = 120) -> list[str]:
    """Merged PRs first (best summaries), then feature commits. Deduplicated."""
    since = f"{year}-{month:02d}-01"
    if month == 12:
        until = f"{year + 1}-01-01"
    else:
        until = f"{year}-{month + 1:02d}-01"
    fmt = ["log", f"--since={since}", f"--until={until}",
           "--format=%ad %s", "--date=short", "--reverse"]
    merges = _git(fmt + ["--merges"]).splitlines()
    feats = _git(fmt + ["--no-merges", "--grep=^feat", "--regexp-ignore-case"]).splitlines()
    seen: set[str] = set()
    lines: list[str] = []
    for line in merges + feats:
        # Strip the date prefix for dedup; keep full line for the prompt.
        norm = re.sub(r"^\d{4}-\d{2}-\d{2}\s+", "", line).strip().lower()
        norm = re.sub(r"^merge pull request #\d+ from \S+ ", "", norm)
        if not norm or norm in seen:
            continue
        # Skip noise: branch sync merges carry no information.
        if re.match(r"merge branch ", norm):
            continue
        seen.add(norm)
        lines.append(line.strip())
        if len(lines) >= cap:
            break
    return lines


def call_llm(prompt: str) -> str:
    from openai import OpenAI

    base_url = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
    model = os.getenv("LLM_MODEL", "ministral")
    client = OpenAI(base_url=base_url, api_key=os.getenv("LLM_API_KEY", "") or "not-needed")
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500,
    )
    return resp.choices[0].message.content or ""


def parse_entries(text: str) -> list[dict]:
    # Tolerate markdown fences around the JSONL.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    entries = []
    for line in text.splitlines():
        line = line.strip().rstrip(",")
        if not line.startswith("{"):
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            continue
        keys = [str(k).strip() for k in rec.get("keys", []) if str(k).strip()]
        content = str(rec.get("content", "")).strip()
        if not keys or not content:
            continue
        entries.append({"keys": keys, "content": content, "enabled": True})
    if not entries:
        raise ValueError("LLM returned no usable lore entries")
    return entries


def resolve_month(args: argparse.Namespace) -> tuple[int, int]:
    if args.last_month:
        today = _dt.date.today().replace(day=1)
        prev = today - _dt.timedelta(days=1)
        return prev.year, prev.month
    m = re.fullmatch(r"(\d{4})-(\d{2})", args.month or "")
    if not m:
        raise SystemExit("--month must look like 2026-05 (or use --last-month)")
    return int(m.group(1)), int(m.group(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--month", help="YYYY-MM to write")
    g.add_argument("--last-month", action="store_true",
                   help="write the month before the current one")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing month file")
    ap.add_argument("--dry-run", action="store_true",
                    help="print entries without writing")
    args = ap.parse_args()

    year, month = resolve_month(args)
    label = f"{year}-{month:02d}"
    out_path = LORE_DIR / f"evolution-{label}.jsonl"
    if out_path.exists() and not args.force and not args.dry_run:
        raise SystemExit(f"{out_path.name} already exists (history is immutable); "
                         "use --force to rewrite or --dry-run to preview")

    history = gather_month(year, month)
    if not history:
        raise SystemExit(f"no git history found for {label}")
    month_name = _dt.date(year, month, 1).strftime("%B %Y")
    prompt = PROMPT_TEMPLATE.format(
        month_label=month_name, history="\n".join(history))
    print(f"gathered {len(history)} history lines for {month_name}; asking LLM…",
          file=sys.stderr)
    entries = parse_entries(call_llm(prompt))
    print(f"got {len(entries)} entries", file=sys.stderr)

    if args.dry_run:
        for e in entries:
            print(json.dumps(e))
        return
    LORE_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
