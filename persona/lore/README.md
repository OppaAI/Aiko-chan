# Lorebook (World Info)

Static facts that never change live here instead of in vector recall:
names, places, running jokes, hard rules. One `*.jsonl` file per topic,
one object per line:

```json
{"keys": ["odessa", "weather station"], "content": "The old weather station on Odessa hill ...", "enabled": true}
```

- `keys`: trigger phrases, matched whole-word, case-insensitive, against
  the current user input.
- `content`: injected verbatim inside a `<lore>` block when any key hits.
- `enabled`: `false` skips without deleting.

Budgets: each entry capped at ~1200 chars, the lorebook block at ~200
tokens per turn (`RECALL_TOKEN_CAP_LOREBOOK`, env-tunable). Keep entries
short and factual — this is not a second persona file (that's SOUL.md).

## Evolution files (history convention)

`evolution.jsonl` holds timeless entries (origin, builders). Each
`evolution-YYYY-MM.jsonl` holds that month's history, written by Aiko
herself from git history — never by hand.

Rule: history is immutable — once a month ends, its file is frozen and
never edited, even if systems get replaced or removed later.

Workflow (`scripts/write_evolution.py`, runs on Aiko's box with her LLM):
- Backfill: `python3 scripts/write_evolution.py --month 2026-05`
- Monthly consolidate (first days of a new month, writes the month
  before): `python3 scripts/write_evolution.py --last-month`
- Preview without writing: add `--dry-run`. Existing month files refuse
  to overwrite unless `--force` is given.

The script feeds the month's merged PRs and feature commits to the LLM,
which picks the historically significant features and changes (not minor
fixes) and writes 2-5 lore entries as dated claims.
