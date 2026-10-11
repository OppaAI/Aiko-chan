"""World Info lorebooks (SillyTavern-inspired, minimal v1).

Static lore lives here instead of in vector recall: entries trigger on
keywords in the current input and inject verbatim. Cheaper than RRF
retrieval for facts that never change (names, places, running jokes,
hard rules), and editable as plain files — no re-embedding, no DB.

Entry format (JSONL, one object per line, `*.jsonl` under this dir):
    {"keys": ["odessa", "weather station"], "content": "...", "enabled": true}

- `keys`: trigger phrases, matched whole-word case-insensitively.
- `content`: injected verbatim when any key hits.
- `enabled`: false skips without deleting (default true).

v1 scope: current user input only, file order, first N that fit the
lorebook token cap. Later: constant entries, scan depth, per-entry
budgets, ST-style recursion/selective logic.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

LORE_DIR = Path(__file__).resolve().parents[2] / "persona" / "lore"
ENTRY_BUDGET_CHARS = 1200  # per-entry sanity cap before the token budget runs


def _lore_dir() -> Path:
    override = os.getenv("Aiko_LORE_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return LORE_DIR


def load_entries(lore_dir: str | Path | None = None) -> list[dict]:
    """All enabled entries from every *.jsonl in the lore dir, in file order."""
    d = Path(lore_dir) if lore_dir else _lore_dir()
    entries: list[dict] = []
    if not d.is_dir():
        return entries
    for path in sorted(d.glob("*.jsonl")):
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if not rec.get("enabled", True):
                        continue
                    keys = [str(k).strip() for k in rec.get("keys", []) if str(k).strip()]
                    content = str(rec.get("content", "")).strip()
                    if not keys or not content:
                        continue
                    entries.append({"keys": keys, "content": content,
                                    "source": f"{path.name}"})
        except OSError:
            continue
    return entries


def _key_re(keys: list[str]) -> "re.Pattern[str] | None":
    parts = []
    for k in keys:
        k = k.strip()
        if not k:
            continue
        parts.append(r"(?<!\w)" + re.escape(k) + r"(?!\w)")
    if not parts:
        return None
    return re.compile("|".join(parts), re.IGNORECASE)


def match_entries(text: str, entries: list[dict]) -> list[dict]:
    """Entries whose keys hit `text` (whole-word, case-insensitive)."""
    text = text or ""
    hits = []
    for e in entries:
        rx = _key_re(e["keys"])
        if rx is not None and rx.search(text):
            hits.append(e)
    return hits


def lorebook_context_for(text: str, *, lore_dir: str | Path | None = None,
                         max_chars: int = 600) -> str:
    """Render matched lore as a prompt block ("" when nothing matches)."""
    hits = match_entries(text, load_entries(lore_dir))
    if not hits:
        return ""
    parts = []
    used = 0
    for h in hits:
        chunk = h["content"][:ENTRY_BUDGET_CHARS]
        if used + len(chunk) > max_chars:
            break
        parts.append(chunk)
        used += len(chunk)
    if not parts:
        return ""
    return "<lore>\n" + "\n\n".join(parts) + "\n</lore>"
