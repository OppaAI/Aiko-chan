"""Content-hash memoization for side-effect-free graph nodes.

`_run_node` in graph_engine consults this module before executing a node:
when the same (tool, resolved args, model, user) produced a result recently,
the cached content is replayed instead of re-running the tool. This pays off
for LLM/network-heavy nodes (research, fetch, synthesis) when scheduled
playbooks repeat — e.g. a daily research playbook re-running identical nodes.

Safety rules (do not relax without review):
- Explicit allowlist only. Every entry was verified side-effect-free: no
  file/DB/network writes, no sends, no state mutation. `deep_research` is
  included ONLY when `tool_mode=True` (otherwise it internally runs
  write_report/learn_report).
- Only successful (`ok=True`) results are cached; failures and retries are
  never stored.
- The key includes the user id, so user-scoped tools (kb_search) can never
  leak results across users.
- TTLs are per tool: web content goes stale (1h), KB lookups slower (24h),
  synthesis over identical evidence is stable for hours (6h).
- Non-determinism note: synthesize_report runs at temperature=0.2, so a
  cached hit replays the first sample instead of resampling. For identical
  evidence within the TTL this is the desired consistency, not a bug.

Disable with AIKO_GRAPH_MEMO=0.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from typing import Any

from system.log import get_logger
from system.userspace import current_user_id

log = get_logger(__name__)

ENABLED = os.getenv("AIKO_GRAPH_MEMO", "1").strip().lower() not in {"0", "false", "no", "off"}
_MAX_ENTRIES = max(1, int(os.getenv("AIKO_GRAPH_MEMO_MAX", "512")))

# Tool -> TTL seconds. Keep this list minimal and verified; see module docstring.
_MEMO_TTLS: dict[str, int] = {
    "adaptive_search": 3600,
    "deep_research": 3600,  # only when tool_mode=True; see _memo_key
    "deep_read": 3600,
    "fetch_from_url": 3600,
    "kb_search": 86400,
    "synthesize_report": 21600,
}

# key -> (expires_at_monotonic, content)
_memo: dict[str, tuple[float, str]] = {}
_memo_lock = threading.Lock()


def _canonical(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def _memo_key(
    tool: str,
    args: dict[str, Any],
    llm_model: str | None,
) -> str | None:
    """Return the cache key for a node, or None if it must not be memoized."""
    if not ENABLED:
        return None
    ttl = _MEMO_TTLS.get(tool)
    if ttl is None:
        return None
    if tool == "deep_research" and args.get("tool_mode") is not True:
        # Default deep_research runs write_report/learn_report internally.
        return None
    try:
        user_id = current_user_id()
    except Exception:
        # Fail closed: an identity failure must never mint a key under a
        # shared fallback identity (e.g. "None") that could replay one
        # user's cached result to another caller.
        return None
    h = hashlib.sha256()
    h.update(tool.encode("utf-8"))
    h.update(b"\x00")
    h.update(_canonical(args).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(llm_model).encode("utf-8"))
    h.update(b"\x00")
    h.update(str(user_id).encode("utf-8"))
    return h.hexdigest()


def _memo_get(key: str) -> str | None:
    now = time.monotonic()
    with _memo_lock:
        entry = _memo.get(key)
        if entry is None:
            return None
        expires_at, content = entry
        if expires_at <= now:
            _memo.pop(key, None)
            return None
        return content


def _memo_put(key: str, tool: str, content: str) -> None:
    ttl = _MEMO_TTLS.get(tool)
    if ttl is None:
        return
    with _memo_lock:
        if len(_memo) >= _MAX_ENTRIES:
            # FIFO evict the oldest entry; dicts preserve insertion order.
            _memo.pop(next(iter(_memo)))
        _memo[key] = (time.monotonic() + ttl, content)


def memo_stats() -> dict[str, int]:
    """Current cache size (for diagnostics)."""
    with _memo_lock:
        return {"entries": len(_memo)}
