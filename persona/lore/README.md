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
