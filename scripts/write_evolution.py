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
import ast
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

Write 2-8 lorebook entries capturing what was historically significant that month: features shipped, systems built, major changes. Scale to the month: 2-3 for a quiet month, up to 8 for a busy one — one entry per significant feature or change. Skip minor fixes, typo corrections, doc tweaks, dependency bumps, and merge noise.

Each entry is one JSON object per line with:
- "keys": 2-6 trigger words or phrases someone would say when asking about this (e.g. "june 2026", "sqlite-vec", "memory backend")
- "content": 1-2 sentences phrased as a dated historical claim (e.g. "June 4, 2026: ..."). It must stay true forever even if the system is later replaced or removed.
- "enabled": true

Rules:
- Brief and factual. No speculation beyond what the history shows.
- At least one entry must carry the month name as a key (e.g. "june 2026") so "what happened in June?" finds it.
- Output ONLY the JSON lines, no markdown fences, no commentary.

Example of one entry (exactly this shape, double quotes, one per line):
{"keys": ["june 2026", "sqlite-vec"], "content": "June 4, 2026: the memory backend moved from Qdrant to a custom sqlite-vec.", "enabled": true}

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
    """Merged PRs + feature commits, sampled evenly across the whole month.

    Records use committer date (%cd) so displayed dates match the
    --since/--until selection boundaries. Merge commits carry their PR
    title (from the commit body) instead of the bare "Merge pull request"
    subject. When history exceeds the cap, stratified sampling keeps
    end-of-month changes eligible instead of truncating them.
    """
    since = f"{year}-{month:02d}-01"
    if month == 12:
        until = f"{year + 1}-01-01"
    else:
        until = f"{year}-{month + 1:02d}-01"
    # NUL-separated records: hash, committer date, subject, body.
    fmt = ["log", f"--since={since}", f"--until={until}",
           "--format=%H%x00%cd%x00%s%x00%b%x1e", "--date=short"]
    raw = (_git(fmt + ["--merges"])
           + _git(fmt + ["--no-merges", "--grep=^feat", "--regexp-ignore-case"]))
    seen: set[str] = set()
    records: list[tuple[str, str]] = []
    for chunk in raw.split("\x1e"):
        parts = chunk.strip().split("\x00")
        if len(parts) < 4:
            continue
        _sha, date, subject, body = (p.strip() for p in parts[:4])
        # --until is inclusive in git; keep the upper bound exclusive so a
        # month's history never borrows the next month's first commit.
        if not (since <= date < until):
            continue
        if re.match(r"merge pull request #\d+ from \S+", subject, re.I):
            # GitHub puts the PR title on the first body line.
            title = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
            display = title or subject
        else:
            display = subject
        norm = display.lower()
        if not norm or norm in seen:
            continue
        # Skip noise: branch sync merges carry no information.
        if re.match(r"merge branch ", norm):
            continue
        seen.add(norm)
        records.append((date, display))
    records.sort(key=lambda r: r[0])  # chronological
    lines = [f"{d} {t}" for d, t in records]
    if len(lines) > cap:
        step = len(lines) / cap
        lines = [lines[int(i * step)] for i in range(cap)]
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


def _parse_line(line: str):
    """Parse one candidate entry, tolerating small-model output quirks."""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        pass
    try:
        # Single-quoted dicts, True/False/None literals.
        rec = ast.literal_eval(line)
        return rec if isinstance(rec, dict) else None
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return None


def parse_entries(text: str, month_key: str) -> list[dict]:
    # Tolerate markdown fences around the JSONL.
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    recs: list = []
    s = text.strip()
    if s.startswith("["):
        # Whole response is a JSON array instead of JSONL.
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                recs = arr
        except json.JSONDecodeError:
            pass
    if not recs:
        for line in s.splitlines():
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            rec = _parse_line(line)
            if rec is None:
                print(f"warning: skipping unparseable entry: {line[:100]}",
                      file=sys.stderr)
                continue
            recs.append(rec)
    entries = []
    for rec in recs:
        if not isinstance(rec, dict):
            continue
        # keys must be a list of nonempty strings; a bare string like
        # "memory" would otherwise splinter into single-character keys.
        raw_keys = rec.get("keys")
        if not isinstance(raw_keys, list):
            continue
        keys = [k.strip() for k in raw_keys
                if isinstance(k, str) and k.strip()]
        content = str(rec.get("content", "")).strip()
        if not keys or not content:
            continue
        entries.append({"keys": keys, "content": content, "enabled": True})
    if not entries:
        raise ValueError("LLM returned no usable lore entries")
    # The prompt requires at least one entry discoverable by month name.
    # If the LLM omitted it, attach it to the first entry rather than
    # saving a month file that can't be found by its month-name trigger.
    if not any(month_key in (k.lower() for k in e["keys"]) for e in entries):
        entries[0]["keys"].insert(0, month_key)
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
    # Plain replace, not str.format: the template contains literal JSON
    # braces (and history lines may too), which format() would misparse.
    prompt = (PROMPT_TEMPLATE
              .replace("{month_label}", month_name)
              .replace("{history}", "\n".join(history)))
    print(f"gathered {len(history)} history lines for {month_name}; asking LLM…",
          file=sys.stderr)
    entries = parse_entries(call_llm(prompt), month_name.lower())
    print(f"got {len(entries)} entries", file=sys.stderr)

    if args.dry_run:
        for e in entries:
            print(json.dumps(e))
        return
    if args.force and out_path.exists():
        # --force is an escape hatch for a failed/garbled run of the open
        # month; closed-month history stays immutable.
        if _dt.date(year, month, 1) < _dt.date.today().replace(day=1):
            print(f"WARNING: --force on closed month {label}; "
                  "history should stay immutable", file=sys.stderr)
    LORE_DIR.mkdir(parents=True, exist_ok=True)
    # Write to a temp file and replace atomically: a failed --force run
    # must not truncate the existing history.
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(LORE_DIR),
                               prefix=out_path.stem + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        os.replace(tmp, out_path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
