"""
toolkit/fetch_heal.py

Self-healing web-fetch strategy memory — the local translation of Bright
Data's "self-healing collector" idea, minus the vendor.

Bright Data heals server-side (their infra AI-refactors a broken collector
template). Aiko's fetch failures are dumber and local: a domain whose pages
are JS walls fails trafilatura every time; a domain that blocks the thin
downloader fails before extraction even runs. Without memory, every fetch
pays the same dead strategies in the same order, forever.

This module is that memory: a tiny per-domain ledger recording which
extraction strategy last worked and which keep failing. Callers ask for a
preferred strategy order and record outcomes; domains with a history of
trafilatura failure go straight to the escalation that works. Nothing here
fetches anything — websearch.web_fetch and research.deep_read own their
chains; they just consult this for ordering.

Ledger file: <user_state_dir>/fetch_health.json, best-effort. A missing
or corrupt ledger degrades to the default strategy order — healing is an
optimization, never a dependency.

Healing semantics (deliberate):
- Prefer recent success; deprioritize (never ban) strategies with >=3
  consecutive failures. Sites change back; a ban would need its own
  healer. Time-decay: failures older than 7 days stop counting.
- Unknown strategies in the ledger are ignored (forward-compat).
- All I/O fail-open: ledger errors log at debug and return defaults.
"""
from __future__ import annotations

import importlib.util
import json
import os
import time
from urllib.parse import urlparse

from system.log import get_logger

log = get_logger(__name__)

STRATEGIES = ("trafilatura", "tavily", "crawl4ai")

FETCH_HEAL_FILE = os.getenv("FETCH_HEAL_FILE", "fetch_health.json")
FETCH_HEAL_FAIL_THRESHOLD = int(os.getenv("FETCH_HEAL_FAIL_THRESHOLD", "3"))
FETCH_HEAL_FAIL_TTL_SECONDS = int(os.getenv("FETCH_HEAL_FAIL_TTL_SECONDS", str(7 * 86400)))
FETCH_HEAL_MAX_DOMAINS = int(os.getenv("FETCH_HEAL_MAX_DOMAINS", "500"))


def _ledger_path() -> str | None:
    try:
        from system.userspace import user_state_dir
        return str(user_state_dir() / FETCH_HEAL_FILE)
    except Exception as exc:
        log.debug("fetch_heal: state dir unavailable: %s", exc)
        return None


def _load() -> dict:
    path = _ledger_path()
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.debug("fetch_heal: ledger unreadable, using defaults: %s", exc)
        return {}


def _save(ledger: dict) -> None:
    path = _ledger_path()
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ledger, f)
    except Exception as exc:
        log.debug("fetch_heal: ledger unwritable: %s", exc)


def _domain_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.") or "unknown"
    except Exception:
        return "unknown"


def strategy_available(strategy: str) -> bool:
    """Whether a strategy can run in this environment right now."""
    if strategy == "trafilatura":
        return importlib.util.find_spec("trafilatura") is not None
    if strategy == "tavily":
        from agentic.toolkit import websearch
        return bool(websearch.TAVILY_API_KEY)
    if strategy == "crawl4ai":
        return importlib.util.find_spec("crawl4ai") is not None
    return False


def _recent_fails(entry: dict, strategy: str, now: float) -> int:
    hist = (entry.get("strategies") or {}).get(strategy) or {}
    fails = hist.get("consecutive_fails", 0)
    try:
        fails = int(fails)
    except (TypeError, ValueError):
        return 0
    last = hist.get("last_fail", 0)
    try:
        last = float(last)
    except (TypeError, ValueError):
        last = 0
    if fails > 0 and last and (now - last) > FETCH_HEAL_FAIL_TTL_SECONDS:
        return 0  # stale failure — site may have healed itself
    return fails


def preferred_order(url: str, strategies: tuple[str, ...] = STRATEGIES) -> list[str]:
    """Strategy names, best-first, for this URL's domain.

    Available strategies ordered by ledger memory: recent winners first,
    chronic failures last. Unavailable strategies are dropped entirely.
    Unknown domains (or any ledger problem) yield default order.
    """
    avail = [s for s in strategies if s in STRATEGIES and strategy_available(s)]
    if not avail:
        return [s for s in strategies if s in STRATEGIES]
    domain = _domain_of(url)
    ledger = _load()
    entry = ledger.get(domain)
    if not isinstance(entry, dict):
        return avail
    now = time.time()
    healthy = [s for s in avail if _recent_fails(entry, s, now) < FETCH_HEAL_FAIL_THRESHOLD]
    sick = [s for s in avail if s not in healthy]
    if sick:
        log.debug("fetch_heal: %s prefers %s over failed %s", domain, healthy or avail, sick)
    return healthy + sick


def record(url: str, strategy: str, ok: bool) -> None:
    """Record one strategy outcome for this URL's domain."""
    if strategy not in STRATEGIES:
        return
    domain = _domain_of(url)
    ledger = _load()
    entry = ledger.get(domain)
    if not isinstance(entry, dict):
        entry = {}
    strats = entry.get("strategies")
    if not isinstance(strats, dict):
        strats = {}
    hist = strats.get(strategy)
    if not isinstance(hist, dict):
        hist = {}
    now = time.time()
    if ok:
        hist["consecutive_fails"] = 0
        hist["last_ok"] = now
    else:
        try:
            fails = int(hist.get("consecutive_fails", 0))
        except (TypeError, ValueError):
            fails = 0
        hist["consecutive_fails"] = fails + 1
        hist["last_fail"] = now
    strats[strategy] = hist
    entry["strategies"] = strats
    entry["updated"] = now
    ledger[domain] = entry
    if len(ledger) > FETCH_HEAL_MAX_DOMAINS:
        # Prune least-recently-updated domains; the ledger is a cache,
        # not an archive — evicted domains simply heal from scratch.
        for old in sorted(ledger, key=lambda d: (ledger[d].get("updated", 0) if isinstance(ledger[d], dict) else 0))[: len(ledger) - FETCH_HEAL_MAX_DOMAINS]:
            ledger.pop(old, None)
    _save(ledger)
