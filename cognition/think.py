"""
cognition/think.py

Aiko's chat facade.
  - Routes between single-shot chat and the agentic task loop in agentic.agentic.
  - Streams llama.cpp response to console + TTS simultaneously.
  - Queues long-term memory writes (delegated to cognition.memory.memorize's async write queue).
  - Owns scheduled-job callbacks and idle learner handoff (delegated to memory.learn).
  - Owns the proactive idle check-in state machine (config/proactive.yaml),
    which is also the "is Aiko resting" signal memory.learn's idle_learner_loop
    waits on before starting autonomous quick-study top-ups.

Memory + knowledge-base fetch:
  route() runs a bounded self-assessment gate (should_attempt, mode=route)
  *before* quaternary intent routing, so soft outcomes (defer/clarify/
  degrade_chat) apply even when the turn would have been localchat.
  Then route() resolves quaternary intent (greeting/localchat/webchat/agentic).
  Greeting-only turns short-circuit directly to the LLM without memory/KB
  recall or memory writeback. All other paths start memory + KB recall only
  after intent is known, then hand the resulting future to the selected
  handler. Wiki/policy/skill/experience context is agentic-only and fetched
  separately, inside agentic.agentic.run_agentic_chat, only once intent has
  actually resolved to "agentic".

Deep-think mode:
  A human being asked a hard question doesn't just answer with the first
  idea that comes to mind — they stop, turn it over, check it against what
  they already know, and only then speak. `/think` and an explicit natural-
  language ask ("think this through more carefully", "really think about
  this") both route through chat(deep_think=True) — see _DEEP_THINK_RE and
  _is_deep_think_request() below. This is detected as a fast regex path in
  route(), BEFORE quaternary intent classification, because it's a request
  about *how* to answer (take more care), not *what kind* of task this is —
  it layers on top of localchat rather than competing with agentic/webchat
  as a fifth semantic label. Deep-think mode widens memory/knowledge recall
  (DEEP_THINK_MEMORY_LIMIT/DEEP_THINK_KNOWLEDGE_LIMIT), multiplies the token
  budget (DEEP_THINK_TOKEN_SCALE), injects the structured scaffold from
  cognition.deep_think (understand → evidence → filter → deliberate →
  verify → self-check → synthesize), reranks recall with stricter recency
  and contradiction flags, and exposes a short structured summary for the
  UI instead of raw chain-of-thought.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import json
import warnings

from system.config import env_float, env_int

import numpy as np

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from datetime import datetime
from openai import OpenAI
from pathlib import Path
import re
import contextvars
import threading
import time
import unicodedata

from agentic.tools    import web_search_context
from agentic.agentic  import run_agentic_chat
from agentic.wiki import wiki_knowledge_context_for
from cognition.knowledge import knowledge_context_for
from cognition import CONTEXT_POOL
from system.log      import get_logger
from system.schedule import DueJob, schedule_job_record, list_schedule_records, cancel_schedule_record
from system.userspace import current_user_id, current_display_name, user_profile_path
from system import brain_trace as _brain_trace
from system import bioclock
from cognition import reason
from cognition.memory import learn

# NOTE: weekly_social handler is registered by system.schedule
# (register_social_handlers via register_system_handlers_only at boot) — do
# NOT re-register here. Importing agentic.toolkit.social at think-module load
# dragged the whole social stack (OpenAI clients, requests, vision deps) into
# every cognition.think import.

log = get_logger(__name__)

_GOAL_REVIEW_TITLE = "[Aiko] Goal review"

def _sync_goal_review_schedule(state) -> None:
    """Keep one gentle recurring review reminder in sync with cognitive state."""
    try:
        uid = current_user_id()
        snap = state.snapshot()
        active = bool(snap.get("goals") or snap.get("open_loops"))
        jobs = list_schedule_records(include_disabled=True, user_id=uid)
        existing = [job for job in jobs if job.get("title") == _GOAL_REVIEW_TITLE]
        if active and not any(job.get("enabled", True) for job in existing):
            now = bioclock.local_now()
            schedule_job_record(
                _GOAL_REVIEW_TITLE,
                "Review Aiko\x27s active goals and unresolved threads; ask the user before taking consequential action.",
                now.strftime("%H:%M"),
                frequency="interval",
                interval_seconds=21600,
                action="announce",
                user_id=uid,
            )
        elif not active:
            for job in existing:
                if job.get("enabled", True):
                    cancel_schedule_record(str(job.get("id")), user_id=uid)
    except Exception as exc:
        log.debug("Goal review schedule sync skipped: %s", exc)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':    'Loading llama.cpp client + persona...',
    'think_warmup':   'Warming up language model...',
    'think_mem_wait': 'Waiting on memory system...',
    # 'think_prewarm' removed — prewarm moved to system/prepare (post-auth),
    # so the label would count a step that never fires and skew boot %.
}

# ── config ────────────────────────────────────────────────────────────────────

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:8080/v1")
LLM_MODEL    = os.getenv("LLM_MODEL",    "ministral")
ROUTER_MODEL = os.getenv("ROUTER_MODEL", LLM_MODEL)
LLM_TIMEOUT  = env_float("LLM_TIMEOUT", 120)
# Stop sequences sent on every LLM call. Model-specific — only the real
# EOS matters; the third legacy token ([INST], raw instruct formatting)
# is never emitted in chat-completions mode and was dead weight. Default
# keeps the two common EOS tokens and drops [INST]; override per model
# via LLM_STOP_SEQUENCES (comma-separated) if a different EOS is needed.
LLM_STOP_SEQUENCES = [s.strip() for s in os.getenv("LLM_STOP_SEQUENCES", "</s>,<|im_end|>").split(",") if s.strip()]

# llama-server KV-cache reuse: re-evaluate the longest common prompt prefix
# from cache instead of re-prefilling every turn. No-op on backends that
# ignore the field. Disable with LLM_CACHE_PROMPT=0 if a non-llama proxy
# rejects unknown body params.
_LLM_CACHE_PROMPT = os.getenv("LLM_CACHE_PROMPT", "1").strip().lower() not in {"0", "false", "no", "off"}

# No-think mode for hybrid-reasoning templates (MiniCPM5 `enable_thinking`,
# Qwen3-style `enable_thinking`, MiniCPM4.1 `/no_think` suffix). When set,
# chat requests both append the `/no_think` suffix to the final user turn
# (template-level, works through any server that honors the convention)
# AND send chat_template_kwargs.enable_thinking=false (honored by
# vLLM/SGLang and any llama.cpp build that forwards template kwargs; inert
# JSON elsewhere). Either mechanism firing is enough. Leave off for
# non-thinking models (ministral) — the suffix would be visible prompt
# noise to a template that doesn't strip it.
_LLM_NO_THINK = os.getenv("LLM_NO_THINK", "").strip().lower() in {"1", "true", "yes", "on"}
_NO_THINK_SUFFIX = " /no_think"


def _llm_extra_body() -> dict:
    """Extra OpenAI body params for chat completions (cache + no-think)."""
    body: dict = {
        "cache_prompt": _LLM_CACHE_PROMPT,
        "repeat_penalty": float(os.getenv("REPEAT_PENALTY", 1.15)),
        "repeat_last_n":  int(os.getenv("REPEAT_LAST_N", 64)),
        "top_k":          int(os.getenv("TOP_K", 40)),
    }
    if _LLM_NO_THINK:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _apply_no_think(messages: list[dict]) -> list[dict]:
    """Return a copy with `/no_think` appended to the last user turn.

    Idempotent (skips when already suffixed). History/store keep the raw
    text — this only touches the request payload.
    """
    if not _LLM_NO_THINK or not messages:
        return messages
    out = [dict(m) for m in messages]
    for m in reversed(out):
        if m.get("role") == "user":
            content = str(m.get("content") or "")
            if not content.rstrip().endswith("/no_think"):
                m["content"] = content.rstrip() + _NO_THINK_SUFFIX
            break
    return out
CONTEXT_WINDOW_TURNS = env_int("CONTEXT_WINDOW_TURNS", 8)

# Shared default recall/knowledge depth across all three chat paths
# (localchat/webchat/agentic) — see _fetch_memory_and_knowledge below.
MEMORY_RECALL_LIMIT = env_int("MEMORY_RECALL_LIMIT", 3)
KNOWLEDGE_RECALL_LIMIT = env_int("KNOWLEDGE_RECALL_LIMIT", 3)
# Recall hard-timeout — a slow local embed (llama.cpp) must not block the turn.
# On expiry the memory/KB recall is skipped (empty) rather than stalling.
MEMORY_RECALL_TIMEOUT = env_float("MEMORY_RECALL_TIMEOUT", 5.0)
# Minimum recall score (see _MemoryBackend._rank_and_score's final_score in
# memory/memorize.py for the formula) a memory must clear to be included in
# context. Same numeric scale as memorize.py's MEMORY_RECALL_SCORE_THRESHOLD
# (~0.015) — that constant only decides quick-vs-wide search, this one
# actually filters weak individual results out of what gets returned.
# 0 = off (default) — no memory is ever dropped for being weak.
MEMORY_MIN_SCORE = env_float("MEMORY_MIN_SCORE", 0.0)

# ── deep-think mode ───────────────────────────────────────────────────────────
# See module docstring. Triggered by /think (orchestrate.py) or by an
# explicit natural-language ask matched by _DEEP_THINK_RE below. Scales well
# past ordinary self._reasoning mode (_REASONING_SCALE) on every axis: token
# budget, memory/knowledge recall depth, and prompt structure.
DEEP_THINK_TOKEN_SCALE = env_int("DEEP_THINK_TOKEN_SCALE", 6)
DEEP_THINK_MEMORY_LIMIT = env_int("DEEP_THINK_MEMORY_LIMIT", max(MEMORY_RECALL_LIMIT * 3, 8))
DEEP_THINK_KNOWLEDGE_LIMIT = env_int("DEEP_THINK_KNOWLEDGE_LIMIT", max(KNOWLEDGE_RECALL_LIMIT * 3, 8))

def _resolve_base_tokens() -> int:
    try:
        return int(os.getenv("LLM_MAX_TOKENS", "280"))
    except (TypeError, ValueError):
        return 280


_BASE_TOKENS     = _resolve_base_tokens()
_REASONING_SCALE = env_int("REASONING_SCALE", 3)

# Thinking-model headroom. Reasoning-capable templates (granite-4, MiniCPM-
# think, …) spend tokens in a separate think channel BEFORE any answer
# content; with only _BASE_TOKENS (280) they hit `finish=length` mid-think
# and return 0 content chars — stream and fallback both "empty" with no
# error. The extra budget is added whenever the backend is a known thinker:
# explicitly via LLM_THINKING=1, or automatically after the first turn that
# yields reasoning tokens (see _note_thinker_detected). Tune per model —
# granite-4.2 needed ~280 think + answer on a greeting; slower Jetson
# builds pay ~5-10 tok/s for every extra token.
_LLM_THINK_BUDGET = env_int("LLM_THINK_BUDGET", 512)
_LLM_THINKING_HINT = os.getenv("LLM_THINKING", "").strip().lower() in {"1", "true", "yes", "on"}
_THINKER_DETECTED = False
_THINKER_LOCK = threading.Lock()
# Set when the last stream produced reasoning but no content: the empty
# result is a burned think budget, NOT a message-shape rejection, so the
# no-system retry would just burn another full budget thinking. The
# fallback consumes (resets) it when deciding to skip that retry.
_LAST_STREAM_REASONED = False


def _note_thinker_detected() -> None:
    """Latch that the backend emits thinking-channel tokens (idempotent)."""
    global _THINKER_DETECTED
    if _THINKER_DETECTED:
        return
    with _THINKER_LOCK:
        _THINKER_DETECTED = True
    log.warning("LLM backend emits thinking tokens — adding %d headroom to max_tokens from now on", _LLM_THINK_BUDGET)


def _note_stream_reasoned(reasoned_without_content: bool) -> None:
    """Record whether the last stream thought without answering."""
    global _LAST_STREAM_REASONED
    with _THINKER_LOCK:
        _LAST_STREAM_REASONED = bool(reasoned_without_content)


def _take_stream_reasoned() -> bool:
    """Consume the flag (reset to False), returning its previous value."""
    global _LAST_STREAM_REASONED
    with _THINKER_LOCK:
        val = _LAST_STREAM_REASONED
        _LAST_STREAM_REASONED = False
        return val


def _effective_max_tokens(base: int) -> int:
    """Base prediction budget plus think headroom when the backend thinks."""
    if _LLM_THINKING_HINT or _THINKER_DETECTED:
        return int(base) + _LLM_THINK_BUDGET
    return int(base)
_ROUTE_ENABLED = os.getenv("ROUTE_ENABLED", "1").lower() in {"1", "true", "yes", "on"}

# ROUTE_MODE selects the classification METHOD only (see yaml comment for
# the four options). It does not decide whether agentic is reachable —
# that's AGENTIC_MODE_ON, applied uniformly below regardless of method.
_ROUTE_VALID_MODES = {"semantic", "semantic_only", "llm", "llm_only"}
_ROUTE_MODE = os.getenv("ROUTE_MODE", "semantic").strip().lower()
if _ROUTE_MODE not in _ROUTE_VALID_MODES:
    log.warning("[route] invalid ROUTE_MODE=%r, defaulting to 'semantic'", _ROUTE_MODE)
    _ROUTE_MODE = "semantic"

# Whether "agentic" is a reachable routing outcome at all. Off = agentic
# is excluded from scoring AND from any LLM tie-break/classify, in every
# ROUTE_MODE, so requests degrade to webchat/localchat instead.
_AGENTIC_MODE_ON = os.getenv("AGENTIC_MODE_ON", "1").lower() in {"1", "true", "yes", "on"}
# Local chat/webchat: stream tokens + TTS karaoke when True (default).
# Set CHAT_STREAM_EMIT=0 to restore old batch _emit-after-finalize behaviour.
_CHAT_STREAM_EMIT = os.getenv("CHAT_STREAM_EMIT", "1").lower() in {"1", "true", "yes", "on"}


# Three separate instruct strings, one per embedding context
_ROUTE_INSTRUCT_QUATERNARY = "What kind of task or question is this?"  # used by route() for quaternary intent routing
_ROUTE_INSTRUCT_TERNARY = _ROUTE_INSTRUCT_QUATERNARY  # backwards-compatible alias for wakeup/tests

_SEMANTIC_ROUTE_MIN_GAP = float(os.getenv("ROUTE_MIN_GAP", "0.12"))
_SEMANTIC_LABEL_TOP_K = int(os.getenv("ROUTE_LABEL_TOP_K", "3"))
_ROUTE_VECTOR_CACHE_DIR = os.getenv("ROUTE_VECTOR_CACHE_DIR", "route_vectors")

# Last-resort websearch net for plain chat: when a message explicitly asks
# for live internet info but semantic routing classified it as localchat,
# run one lightweight search and offer the results to the persona prompt.
# Mirrors the Threads monitor's _threads_research_context gate.
_CHAT_WEBSEARCH_NET_ENABLED = os.getenv("CHAT_WEBSEARCH_NET", "1").lower() in {"1", "true", "yes", "on"}
_WEBSEARCH_HINT_RE = re.compile(
    r"\b(?:internet|web|online|search|look\s+up|verify|current)\b",
    re.IGNORECASE,
)
# Personal-experience narration ("We arrived there at 11am by bus... I will
# tell you more"). These turns need listening + memory writeback, not search;
# routing can mislabel them webchat because prices/places embedding clusters.
_SHARING_RE = re.compile(
    r"\b(?:i|we)\s+(?:was|were|had|went|arrived|bought|paid|saw|ate|visited|rode|watched|walked|took)\b"
    r"|\b(?:i|we)\s+did\b"
    r"|\bi(?:'ll| will)\s+(?:tell|share|explain)\b"
    r"|\blet me tell\b",
    re.IGNORECASE,
)


def _is_personal_sharing(text: str) -> bool:
    """True for first-person experience narration with no question and no
    explicit internet request — these must not be answered from web results."""
    body = str(text or "")
    return (
        "?" not in body
        and not _WEBSEARCH_HINT_RE.search(body)
        and bool(_SHARING_RE.search(body))
    )

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "SOUL.md"
_LOCAL_KNOWLEDGE_RE = re.compile(
    r"\b("
    # NOTE: no bare "aiko" alternative — it fired on every self-addressed
    # greeting ("Hi Aiko...") and injected ~3k chars of wiki docs into
    # smalltalk prompts. Real doc questions match the phrases below.
    r"your architecture|your hardware|your features?|your functions?|"
    r"what can you do|how do you work|how are you built|"
    r"knowledge base|wiki|docs?|readme|roadmap|install|config|"
    r"SOUL\.md|USER\.md|SKILLS?\.md|SCHEDULE\.md|"
    r"repo|repository|codebase|local files|your files"
    r")\b",
    re.IGNORECASE,
)

# ── conditional persona overrides ────────────────────────────────────────────
# SOUL.md is the always-loaded steady-state
# persona; the two override files below are only appended on turns that
# actually need them (mirrors _LOCAL_KNOWLEDGE_RE / _should_use_local_knowledge
# just below — same "gate the tokens, don't pay for them every turn" pattern).
_PERSONA_DIR = _PERSONA_PATH.parent
_PERSONA_CORE_PATH = _PERSONA_DIR / "SOUL.md"
_PERSONA_JP_PATH = _PERSONA_DIR / "JAPANESE_CHAT.md"
_PERSONA_CODE_PATH = _PERSONA_DIR / "CODING_CHAT.md"

_JAPANESE_TRIGGER_RE = re.compile(r"[\u3040-\u30ff\u4e00-\u9fff]")
_CODE_TRIGGER_RE = re.compile(
    r"\b(debug|traceback|stack trace|error:|exception|refactor|"
    r"write (a|the) (function|script|class)|fix (this|my) code|"
    r"walk me through|\.py\b|\.js\b)\b",
    re.IGNORECASE,
)


_persona_core_cache: str | None = None
_persona_core_mtime: float | None = None


def _load_static_persona() -> str:
    """Read the always-loaded persona core (SOUL.md — no per-user data,
    no conditional overrides).

    Task/tool policy lives in the agentic prompt so casual chat does not pay
    for agentic/schedule tokens on every turn. Japanese/coding overrides live
    in separate files and are appended per-turn by _conditional_persona_blocks
    only when triggered — see _current_system_prompt.

    Cached with an mtime check: zero disk reads on turns where SOUL.md hasn't
    changed, but edits still picked up without a restart.
    """
    global _persona_core_cache, _persona_core_mtime
    if not _PERSONA_CORE_PATH.exists():
        raise FileNotFoundError(f"SOUL.md not found at {_PERSONA_CORE_PATH}")
    try:
        mtime = _PERSONA_CORE_PATH.stat().st_mtime
    except OSError:
        mtime = None
    if _persona_core_cache is None or mtime != _persona_core_mtime:
        _persona_core_cache = _PERSONA_CORE_PATH.read_text(encoding="utf-8").strip()
        _persona_core_mtime = mtime
    return _persona_core_cache


_persona_jp_cache: str | None = None
_persona_code_cache: str | None = None


def _conditional_persona_blocks(user_input: str) -> str:
    """Trigger-loaded persona overrides. Only paid for on turns that need them."""
    global _persona_jp_cache, _persona_code_cache
    blocks = []
    if _JAPANESE_TRIGGER_RE.search(user_input):
        if _persona_jp_cache is None:
            _persona_jp_cache = _PERSONA_JP_PATH.read_text(encoding="utf-8").strip()
        blocks.append(_persona_jp_cache)
    if _CODE_TRIGGER_RE.search(user_input):
        if _persona_code_cache is None:
            _persona_code_cache = _PERSONA_CODE_PATH.read_text(encoding="utf-8").strip()
        blocks.append(_persona_code_cache)
    return ("\n\n" + "\n\n".join(blocks)) if blocks else ""


_user_context_cache: dict[str, tuple[float, str]] = {}  # user_id -> (mtime, block)


def _load_user_context() -> tuple[str, str]:
    """Read the current turn's display name + profile block fresh, every call.

    Must be called from the turn/request context where current_user_id()
    already resolves to the real logged-in user — cached by (user_id, mtime)
    so repeated turns for the same user don't re-read an unchanged file.

    Returns (display_name, user_block) where user_block is either "" or a
    "\n\n"-prefixed profile chunk ready to append to the static persona.
    """
    display_name = current_display_name()
    user_path = user_profile_path()
    uid = current_user_id()
    if uid in _user_context_cache:
        cached_mtime, cached_block = _user_context_cache[uid]
        try:
            current_mtime = user_path.stat().st_mtime
        except OSError:
            current_mtime = 0.0
        if current_mtime == cached_mtime:
            return display_name, cached_block
    context_blocks = []
    if user_path.exists():
        raw = user_path.read_text(encoding="utf-8").strip()
        if raw:
            context_blocks.append(
                "<user_profile>\n"
                "Who you are speaking with — authoritative for identity. "
                "Never claim ignorance of this.\n\n"
                f"{raw}\n"
                "</user_profile>"
            )
    user_block = "\n\n" + "\n\n".join(context_blocks) if context_blocks else ""
    try:
        _user_context_cache[uid] = (user_path.stat().st_mtime, user_block)
    except OSError:
        log.debug("think: user context cache write failed")
    return display_name, user_block


_DEBUG_PROMPT_DUMP_PATH = os.getenv("AIKO_DEBUG_PROMPT_DUMP", "/tmp/aiko_last_prompt.txt")

def _dump_full_prompt(debug: dict) -> None:
    # Default OFF: writes the full prompt blob to SD synchronously EVERY turn
    # (latency + flash wear). Set AIKO_DEBUG_FULL_PROMPT=1 to opt in.
    if os.getenv("AIKO_DEBUG_FULL_PROMPT", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        with open(_DEBUG_PROMPT_DUMP_PATH, "w", encoding="utf-8") as f:
            f.write(f"=== mode={debug.get('mode')} @ {datetime.now().isoformat()} ===\n\n")
            f.write("----- SYSTEM PROMPT (full, untruncated) -----\n")
            f.write(debug.get("system_prompt", "") + "\n\n")
            f.write("----- MEMORY -----\n")
            f.write(debug.get("memory_prompt", "") + "\n\n")
            f.write("----- KNOWLEDGE -----\n")
            f.write(debug.get("knowledge_prompt", "") + "\n\n")
            f.write("----- WEB -----\n")
            f.write(debug.get("web_prompt", "") + "\n\n")
            f.write("----- PREVIOUS CHAT MESSAGES -----\n")
            for m in debug.get("previous_chat_messages", []):
                f.write(f"[{m.get('role')}] {m.get('content')}\n")
    except Exception:
        log.exception("Failed to dump full prompt debug")


def _should_use_local_knowledge(user_input: str) -> bool:
    """Return True for normal-chat questions about Aiko's local docs/files.

    This is separate from the general memory+KB fetch every path already
    gets (see _fetch_memory_and_knowledge) — it's an additional, narrower
    lookup specifically for wiki-authored architecture/feature docs, gated
    so casual chat doesn't pay for it on every turn.
    """
    return bool(_LOCAL_KNOWLEDGE_RE.search(user_input))

# ── semantic intent examples ──────────────────────────────────────────────────

import subprocess

def _play_beep() -> None:
    """Play a short system notification sound before a scheduled job announcement."""
    def _run():
        try:
            subprocess.run(
                ["paplay", "/usr/share/sounds/freedesktop/stereo/bell.oga"],
                check=False, timeout=6,
            )
        except Exception as e:
            log.warning("Beep playback failed: %s", e)
    threading.Thread(target=_run, daemon=True).start()

# load route examples (quaternary intent only - tools/capability moved to agentic/router)
_EXAMPLES_PATH = Path(__file__).resolve().parent.parent / "agentic" / "router" / "intent_prompts.json"

def _load_route_examples(*, include_greeting: bool = True) -> dict:
    try:
        with open(_EXAMPLES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        log.warning("[route] examples missing/unreadable at %s: %s", _EXAMPLES_PATH, e)
        return {}
    raw = dict(data.get("quaternary") or data.get("ternary") or {})
    if not include_greeting:
        raw.pop("greeting", None)
    return {k: tuple(v) for k, v in raw.items()}

_ROUTE_QUATERNARY_EXAMPLES = _load_route_examples()
_ROUTE_TERNARY_EXAMPLES = _load_route_examples(include_greeting=False)

_AGENTIC_ROUTE_RE = re.compile(
    r"\b("
    r"research|look up|search|fetch|find out|check whether|check if|"
    r"fix|debug|implement|refactor|patch|edit|modify|update tests?|inspect|open .*\.(?:py|json|md)|"
    r"write|draft|compose|save|create|prepare|"
    r"plan|roadmap|checklist|break down|schedule|remind|reminder|alarm|timer|ping me|notify me|"
    r"continue|resume|pick up where we left off|keep going|compare .* recommend|decide .* and"
    r")\b",
    re.IGNORECASE,
)

_GREETING_ONLY_RE = re.compile(
    r"^\s*(?:"
    r"hi+|hello+|hey+|hiya+|yo+|sup|"
    r"good\s+(?:morning|afternoon|evening)|morning|evening|"
    r"(?:hi|hello|hey)\s+(?:there|aiko)|"
    r"how(?:'|’)s\s+it\s+going|how\s+are\s+you(?:\s+doing)?|"
    r"what(?:'|’)s\s+up|nice\s+to\s+see\s+you|good\s+to\s+see\s+you|"
    r"just\s+saying\s+hi|thanks?|thank\s+you|okay\s+thanks|cool\s+thanks"
    r")(?:[\s!?.~、。！]*|\s+(?:lol|haha|hehe)[\s!?.~、。！]*)$",
    re.IGNORECASE,
)


def _is_greeting_only(user_input: str) -> bool:
    return bool(_GREETING_ONLY_RE.match(user_input or ""))


# ── deep-think trigger ───────────────────────────────────────────────────────
# Matches an explicit ask for careful/thorough reasoning, in natural language
# — distinct from /think (orchestrate.py), which calls chat(deep_think=True)
# directly. Kept as a fast regex path rather than a fifth semantic-routing
# label: this is a request about HOW to answer (slow down, work it through),
# which should layer on top of whatever localchat/webchat/agentic would have
# picked, not compete with them for the turn.
_DEEP_THINK_RE = re.compile(
    r"\b(?:"
    r"think (?:more |a bit )?(?:deeply|thoroughly|carefully|harder|longer)|"
    r"think (?:this |it |that )?through(?: (?:carefully|thoroughly|properly))?|"
    r"really think (?:about|through|it over)|"
    r"take (?:your|some) time (?:and |to )?think|"
    r"deep(?:er)? think(?:ing)?|"
    r"deep dive(?: into)?|"
    r"reason (?:more )?(?:deeply|carefully|thoroughly)|"
    r"give (?:this|it) (?:some |more )?(?:careful |deep )?thought|"
    r"analy[sz]e (?:this |it )?(?:thoroughly|carefully|deeply|in[- ]depth)|"
    r"slow down and think|"
    r"think (?:step[- ]by[- ]step|out loud) (?:carefully|thoroughly|properly)|"
    r"weigh (?:this |it )?carefully|"
    r"don'?t rush(?:,)? think|"
    r"be (?:extra |really )?thorough"
    r")\b",
    re.IGNORECASE,
)


def _is_deep_think_request(text: str) -> bool:
    """True for an explicit ask to reason more carefully than a normal turn."""
    return bool(_DEEP_THINK_RE.search(text or ""))


# Deep-think scaffold + evidence helpers live in cognition.deep_think.
from cognition.deep_think import (
    DEEP_THINK_GUIDE as _DEEP_THINK_GUIDE,
    evidence_preamble as _deep_think_evidence_preamble,
    format_summary as _deep_think_format_summary,
    rerank_for_deep_think as _deep_think_rerank,
    tool_need_hint as _deep_think_tool_hint,
    trace_payload as _deep_think_trace_payload,
)
# Matches our own injected "[Style only — never quote this. ...]" control
# blocks (see cognition.attention.soft_user_prompt). The LLM prompt keeps
# them, but recall queries, history, cognitive-state recording and memory
# writes must use the raw user text — otherwise the instruction words leak
# into memory search, goals/open-loops ("... style only never quote ask"),
# false contradiction hits (the same block repeats every clarify turn) and
# durable semantic facts.
_STYLE_DIRECTIVE_RE = re.compile(
    r"\s*\[Style only\s*[—–-]\s*never quote this\.[^\]]*\]",
    re.IGNORECASE | re.DOTALL,
)


def _strip_style_directives(text: str) -> str:
    """Remove injected style-control blocks, returning the raw user text."""
    if not text:
        return text
    return _STYLE_DIRECTIVE_RE.sub("", text).strip()


def _extract_search_results_block(system_prompt: str) -> str:
    match = re.search(r"<search_results\b[^>]*>.*?</search_results>", system_prompt or "", re.DOTALL)
    return match.group(0) if match else ""


# ── proactive check-in config (config/proactive.yaml, via os.environ) ─────────
# system.config.load_config() has already populated these into the process
# environment by the time this module is imported (see system/wakeup.py /
# memory/learn.py — whichever entrypoint runs first calls it). We just read
# them here the same way every other module's config block does.
#
# Timezone is no longer configured here — every "now"/timezone lookup in
# this module goes through system.bioclock, the app-wide single source of
# truth (config/bioclock.yaml).




# ── think ─────────────────────────────────────────────────────────────────────

def _format_system_notices(system_note: str | None) -> str:
    """Format drained notice-bus lines as an ephemeral system-prompt block."""
    if not system_note or not system_note.strip():
        return ""
    return (
        "<system_notices>\n"
        "Ephemeral subsystem status for this turn (already handled — "
        "acknowledge briefly only if the user notices something wrong, "
        "otherwise ignore):\n"
        f"{system_note.strip()}\n"
        "</system_notices>"
    )


class AikoThink:
    def __init__(self) -> None:
        self._client    = OpenAI(base_url=LLM_BASE_URL, api_key=os.getenv("LLM_API_KEY", "") or "not-needed")
        self._llm_model = LLM_MODEL
        self._router_model = ROUTER_MODEL
        self._memorize  = None    # injected later via set_memorize() — see system/wakeup.py
        self._speak     = None    # injected later via set_speak()    — see system/wakeup.py
        # Guards self._speak against the toggle-vs-background-thread race.
        # set_speak() is called from the main thread (main.py's /voice
        # toggle). Readers snapshot self._speak under the lock so a toggle
        # landing mid-read can't produce a stale ref or a None mismatch.
        self._speak_lock = threading.Lock()
        self._memorize_lock = threading.Lock()

        self._persona   = _load_static_persona()
        self._history:  list[dict] = []
        self._history_lock = threading.Lock()
        # Cache of (labels, embedding_matrix) per (example-corpus-id, instruct)
        # pair — built via reason.embed_example_matrix, which always
        # re-embeds; caching the result here avoids paying that cost on
        # every routing call for a static example corpus.
        self._semantic_example_cache: dict = {}
        self._semantic_example_cache_lock = threading.RLock()
        self._active_user_ids: set[str] = set()
        self._active_users_lock = threading.Lock()
        self._reasoning = False
        # Deep-think mode flag — set alongside self._reasoning by
        # chat(deep_think=True), and read only by _stream_response to pick
        # the token-budget scale (DEEP_THINK_TOKEN_SCALE vs _REASONING_SCALE).
        # See module docstring "Deep-think mode".
        self._deep_think = False
        self.last_deep_think_summary: str | None = None  # structured UI summary for /think
        self.last_usage: dict = {}
        self.last_prompt_debug: dict = {}
        self._last_chat_time = time.time()

        self._idle_learner_thread: threading.Thread | None = None
        self._warmup_thread: threading.Thread | None = None

        # ── rest-signal state for learn.idle_learner_loop ───────────────────
        # The proactive idle check-in state machine lives in main.py's
        # ProactiveIdleRunner. That runner sets this flag via
        # set_proactive_resting() so learn.idle_learner_loop can see when
        # Aiko is "resting" and pause autonomous study. The flag is cleared
        # by _note_user_activity() on every normal turn.
        self._proactive_lock = threading.Lock()
        self._proactive_resting = False

    def _warmup_llm(self) -> None:
        try:
            self._client.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False, max_tokens=1,
            )
        except Exception as e:
            log.warning("LLM warmup failed: %s", e)

    def join_warmup(self) -> None:
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    def start_warmup(self) -> None:
        """Kick off the LLM warmup call. Call once, right after construction."""
        if self._warmup_thread is not None:
            return
        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def start_idle_learner(self) -> None:
        """Start the background idle-learning loop. Call only after
        set_memorize() has been called — the loop reads self._memorize
        on its first iteration."""
        if self._idle_learner_thread is not None:
            return
        if self._get_memorize() is None:
            log.warning("[think] Memory unavailable — idle learner not started.")
            return
        self._idle_learner_thread = threading.Thread(
            target=learn.idle_learner_loop, args=(self,), daemon=True
        )
        self._idle_learner_thread.start()

    def _persona_core(self) -> str:
        """Stable identity core: persona text + owner block.

        Everything that rarely changes within a session lives here so the
        LLM server's prompt cache (cache_prompt) reuses it across turns —
        see _current_system_prompt_parts() for the stable/volatile split.
        """
        display_name, user_block = _load_user_context()
        return self._persona.replace("USER_ID_HERE", display_name) + user_block

    def _current_system_prompt_parts(self, user_input: str = "") -> tuple[str, str]:
        """Assemble this turn's system prompt as (stable_core, volatile_tail).

        Same content as _current_system_prompt(), split so chat() can send
        them as TWO system messages with conversation history in between:

            [system: core] [history...] [system: volatile] [user]

        The volatile tail (time, state ticks, priming, memories) sits right
        before the newest user turn, leaving the byte-stable core + history
        prefix intact for llama-server's KV reuse. Call only from within a
        turn where current_user_id()/current_display_name() already resolve
        to the real caller — never at construction time.
        """
        core = self._persona_core()
        volatile_parts: list[str] = []
        try:
            from cognition.attention import for_identity
            state_obj = for_identity(current_user_id())
            state_obj.record_activity(os.getenv("AIKO_ACTIVITY", ""))
            state_obj.continuous_tick()
            state_obj.persist()
            state = state_obj.context(user_input)
            if state:
                volatile_parts.append(state)
            try:
                from system.schedule import list_schedule_records
                scheduled_jobs = list_schedule_records(user_id=current_user_id())
            except Exception:
                scheduled_jobs = []
            project_signals = []
            if _CODE_TRIGGER_RE.search(user_input) or _LOCAL_KNOWLEDGE_RE.search(user_input):
                try:
                    result = subprocess.run(
                        ["git", "status", "--short", "--untracked-files=no"],
                        cwd=Path.cwd(), capture_output=True, text=True, timeout=1, check=False,
                    )
                    project_signals = [line.strip() for line in result.stdout.splitlines() if line.strip()][:5]
                except Exception:
                    project_signals = []
            grounded = for_identity(current_user_id()).grounded_context(
                now=bioclock.local_now(),
                idle_seconds=max(0.0, time.time() - self._last_chat_time),
                resting=self.is_proactive_resting(),
                scheduled_jobs=scheduled_jobs,
                project_signals=project_signals,
            )
            volatile_parts.append(grounded)
            volatile_parts.append(state_obj.adaptive_response_guidance())
            volatile_parts.append(state_obj.reflection_summary())
            volatile_parts.append(state_obj.preference_guidance())
            volatile_parts.append(state_obj.lesson_guidance())
            volatile_parts.append(state_obj.identity_guidance())
            volatile_parts.append(state_obj.self_model_context())
            volatile_parts.append(state_obj.subconscious_guidance())
            # Structured reasoning instruction (Anthropic-style CoT with explicit tags)
            reasoning_guide = (
                "When facing complex questions, use explicit structured reasoning:\n"
                "<thinking> identify what is being asked, recall relevant memory, evaluate evidence </thinking>\n"
                "<plan> outline steps: check memory, verify facts, consider context, decide response </plan>\n"
                "<check> verify no contradictions with user preferences, no invented facts, sufficient evidence </check>\n"
                "Keep these tags brief (1-2 sentences each) and visible in your internal reasoning. "
                "Do NOT include these tags in your final spoken response to the user — they are for your own cognition."
            )
            volatile_parts.append(reasoning_guide)
            priming = state_obj.priming_context(user_input)
            if priming:
                volatile_parts.append(priming)
        except Exception:
            pass
        volatile_parts.append(_conditional_persona_blocks(user_input))
        volatile = "\n\n".join(p for p in volatile_parts if p)
        return core, volatile

    def _current_system_prompt(self, user_input: str = "") -> str:
        """Assemble this turn's system prompt: static persona core + fresh
        per-user context + any conditional overrides this input triggers.

        Single-string convenience wrapper over _current_system_prompt_parts()
        — byte-identical output to the historical sequential construction.
        Call only from within a turn where current_user_id()/current_display_name()
        already resolve to the real caller — never at construction time.
        """
        core, volatile = self._current_system_prompt_parts(user_input)
        return core + ("\n\n" + volatile if volatile else "")

    # ── public api ────────────────────────────────────────────────────────────

    def route(self, user_input: str, token_callback=None, system_note: str | None = None) -> str:
        """Main entry point. Quaternary routing.

        Intent is resolved before memory/KB recall. Greeting-only turns are
        intentionally cheap: they go straight to the LLM with persona + recent
        chat history only and skip memory recall, KB recall, and memory
        extraction/writeback. Non-greeting turns then start the shared
        memory+KB future and pass it to the selected handler.

        Deep-think fast path: an explicit ask to think more carefully
        (_is_deep_think_request) is checked FIRST, before quaternary intent
        classification — see module docstring "Deep-think mode". It routes
        straight to chat(deep_think=True) rather than competing with
        agentic/webchat/localchat as a fifth semantic label.

        Per-user-active tracking: multiple users' turns can run concurrently
        (e.g. agentic loop for one user, quick chat for another). Shared
        state (_history, _speak) has its own per-resource lock.
        """
        user_id = current_user_id()
        with self._active_users_lock:
            self._active_user_ids.add(user_id)

        def _finish_route_tracking() -> None:
            with self._active_users_lock:
                self._active_user_ids.discard(user_id)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()

        self._note_user_activity()
        _route_t0 = time.monotonic()
        # Approval commands ("approve run-<id>", "yes") must be handled
        # BEFORE intent classification — the quaternary intent LLM labels
        # terse commands like "Approve run-…" as chat, which would skip
        # run_agentic_chat and leave the pending tool permanently stuck.
        try:
            from agentic.agentic import _maybe_resume_approval
            resumed = _maybe_resume_approval(self, user_input, token_callback=token_callback)
            if resumed is not None:
                try:
                    from cognition.attention import for_identity
                    for_identity(user_id).record_turn_latency(time.monotonic() - _route_t0)
                except Exception:
                    pass
                _finish_route_tracking()
                return resumed
        except Exception as exc:
            log.debug("[route] approval resume pre-check skipped: %s", exc)

        # Conscience HITL: "approve ccc-<id>" / "deny ccc-<id>"
        try:
            from cognition.conscience.hooks import resolve_ccc_approval
            ccc_reply = resolve_ccc_approval(user_input, user_id=user_id, owner=self)
            if ccc_reply is not None:
                self._emit(ccc_reply, token_callback=token_callback)
                try:
                    from cognition.attention import for_identity
                    for_identity(user_id).record_turn_latency(time.monotonic() - _route_t0)
                except Exception:
                    pass
                _finish_route_tracking()
                return ccc_reply
        except Exception as exc:
            log.warning("[route] conscience approval pre-check failed: %s", exc)
            if re.search(r"\b(?:approve|deny|reject)\s+ccc-[0-9a-f]{6,16}\b", user_input or "", re.I):
                ccc_reply = "I couldn't safely resolve that approval just now, so nothing was executed."
                self._emit(ccc_reply, token_callback=token_callback)
                _finish_route_tracking()
                return ccc_reply

        # Self-assessment *before* quaternary routing so localchat/webchat
        # also get executable soft outcomes (defer / clarify / degrade_chat).
        try:
            from cognition.attention import for_identity
            state = for_identity(user_id)
            ok, reason, action = state.should_attempt(user_input, mode="route")
            gate_result = (ok, reason, action)
            snap = state.snapshot()
            _brain_trace.record_step(
                "attention.should_attempt",
                layer="gate",
                inputs={"user_input": user_input, "mode": "route", "user_id": user_id},
                outputs={"ok": ok, "reason": reason, "action": action},
                factors=[
                    f"energy={snap.get('energy')}",
                    f"uncertainty={snap.get('uncertainty')}",
                    f"recent_tool_failures={sum(1 for o in snap.get('tool_outcomes', []) if not o.get('ok'))}",
                    f"contradictions={len(snap.get('contradictions', []))}",
                    f"response_review_flags={snap.get('response_reviews', [{}])[0].get('flags', []) if snap.get('response_reviews') else []}",
                    "time_sensitivity=checked in attention",
                    "answer_completeness=checked from latest response review",
                    "self_consistency=checked from latest response review",
                ],
            )
            if not ok:
                log.info("[route] should_attempt action=%s reason=%s", action, reason)
                reply = self._soft_gate_reply(
                    user_input, action, reason, token_callback=token_callback,
                )
                _finish_route_tracking()
                return reply
        except Exception as exc:
            log.debug("[route] should_attempt skipped: %s", exc)
            gate_result = (True, "gate unavailable", "proceed")

        # ── Conscience Circuit Core (inbound gate) ─────────────────────────
        try:
            from cognition.conscience.hooks import gate_respond
            memorize = self._get_memorize()
            mem_inner = getattr(memorize, "_mem", None) if memorize is not None else None
            embedder = getattr(mem_inner, "_embedder", None)
            decision, reply, note = gate_respond(
                user_input=user_input,
                user_id=user_id,
                llm_client=getattr(self, "_client", None),
                embedder=embedder,
                surface="chat",
            )
            if decision in ("refuse", "escalate") and reply:
                self._emit(reply, token_callback=token_callback)
                with self._history_lock:
                    self._history.append({"role": "user", "content": user_input})
                    self._history.append({"role": "assistant", "content": reply})
                try:
                    from cognition.attention import for_identity
                    for_identity(user_id).record_turn_latency(time.monotonic() - _route_t0)
                except Exception:
                    pass
                _finish_route_tracking()
                return reply
            if decision == "caution" and note:
                system_note = (f"{system_note}" + "\n" + note).strip() if system_note else note
        except Exception as exc:
            log.warning("[route] conscience gate failed; continuing with caution: %s", exc)
            fallback_note = (
                "<conscience_constraint>Conscience evaluation is temporarily unavailable. "
                "Proceed cautiously and take no irreversible action.</conscience_constraint>"
            )
            system_note = (f"{system_note}\n{fallback_note}").strip() if system_note else fallback_note

        try:
            # ── deep-think fast path ────────────────────────────────────────
            # Checked before quaternary intent classification: an explicit
            # "think this through carefully" style ask is a request about HOW
            # to answer, not WHAT kind of task this is, so it bypasses
            # agentic/webchat/localchat classification entirely rather than
            # competing with them. See module docstring "Deep-think mode".
            if _is_deep_think_request(user_input):
                _brain_trace.record_step(
                    "think.route",
                    layer="route",
                    inputs={"user_input": user_input},
                    outputs={"intent": "think_chat", "handler": "chat(deep_think=True)"},
                    factors=[
                        "explicit deep-think phrase matched _DEEP_THINK_RE",
                        "bypasses quaternary intent routing — same mechanism as /think",
                    ],
                )
                return self.chat(
                    user_input,
                    token_callback=token_callback,
                    _skip_search=True,
                    deep_think=True,
                    system_note=system_note,
                )

            intent, route_vec = self._route_intent(user_input)
            log.info("[route] intent=%s", intent)

            if intent == "greeting":
                _brain_trace.record_step(
                    "think.route",
                    layer="route",
                    inputs={"user_input": user_input},
                    outputs={"intent": "greeting", "handler": "chat(skip_memory=True)"},
                    factors=["greeting-only regex match OR threshold + gap on greeting label"],
                )
                return self.chat(
                    user_input,
                    token_callback=token_callback,
                    _skip_search=True,
                    skip_memory=True,
                    store_turn=False,
                    system_note=system_note,
                )

            # Reuse the embedding computed during intent routing as the memory
            # query vector instead of embedding the same text a second time
            # with _QUERY_INSTRUCT. This drops one HTTP embed call per turn.
            query_vec = route_vec
            if query_vec is None:
                try:
                    mem = self._get_memorize()
                    embedder = getattr(getattr(mem, "_mem", None), "_embedder", None) if mem else None
                    if embedder is not None and hasattr(embedder, "embed_query"):
                        query_vec = embedder.embed_query(user_input)
                except Exception:
                    query_vec = None
            mem_kb_future = CONTEXT_POOL.submit(
                self._fetch_memory_and_knowledge, user_input, query_vec
            )

            if intent == "agentic":
                _brain_trace.record_step(
                    "think.route",
                    layer="route",
                    inputs={"user_input": user_input, "intent": "agentic"},
                    outputs={"handler": "agentic_chat", "vector_reused": route_vec is not None},
                    factors=["agentic_score >= 0.78 and gap >= min_gap"],
                )
                return self.agentic_chat(user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec, _from_route=True, system_note=system_note, gate_result=gate_result)
            if intent == "webchat":
                _brain_trace.record_step(
                    "think.route",
                    layer="route",
                    inputs={"user_input": user_input, "intent": "webchat"},
                    outputs={"handler": "webchat", "vector_reused": route_vec is not None},
                    factors=["webchat_score >= 0.72 and gap >= min_gap"],
                )
                return self.webchat(user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec, system_note=system_note)
            _brain_trace.record_step(
                "think.route",
                layer="route",
                inputs={"user_input": user_input, "intent": "localchat"},
                outputs={"handler": "chat", "vector_reused": route_vec is not None, "mem_kb_future_started": mem_kb_future is not None},
                factors=["no label cleared greeting/agentic/webchat thresholds"],
            )
            return self.chat(user_input, token_callback=token_callback, _skip_search=True, mem_kb_future=mem_kb_future, query_vec=query_vec, system_note=system_note)
        finally:
            try:
                from cognition.attention import for_identity
                for_identity(user_id).record_turn_latency(time.monotonic() - _route_t0)
            except Exception:
                pass
            _finish_route_tracking()

    def _recall_query(self, user_input: str) -> str:
        """Recall query enriched with the tail of recent conversation.

        Pronouns ("we went there yesterday") carry no lexical or embedding
        overlap with the memories that contain the antecedent, so searching
        the raw input alone misses them. Folding the last exchange into the
        query lets KNN/FTS find the row "there" points at.
        """
        try:
            with self._history_lock:
                recent = [
                    str(m.get("content") or "")[:200]
                    for m in self._history[-2:]
                    if m.get("content")
                ]
        except Exception:
            recent = []
        tail = " ".join(reversed(recent)).strip()
        enriched = user_input
        if tail:
            enriched = f"{user_input}\n{tail}"[:600]
        if _brain_trace and _brain_trace.TRACE_ENABLED:
            _brain_trace.record_step(
                "think._recall_query",
                layer="context",
                inputs={"user_input": user_input, "enriched": enriched != user_input},
                outputs={"query": enriched, "tail_chars": len(tail), "enriched_query_preview": enriched[:600]},
                factors=["pronoun resolution: recent chat tail folded into query for memory recall"],
            )
        if not tail:
            return user_input
        return f"{user_input}\n{tail}"[:600]

    def _fetch_memory_and_knowledge(
        self, user_input: str, query_vector: np.ndarray | None = None,
        mem_limit: int = MEMORY_RECALL_LIMIT, know_limit: int = KNOWLEDGE_RECALL_LIMIT,
    ) -> tuple[list[dict], str]:
        """Fetch long-term memory + learned-knowledge (KB) concurrently.

        Both are independent reads against separate stores (memory.db /
        knowledge.db). route() now starts this only after quaternary intent
        routing, so greeting-only turns can skip recall entirely while
        agentic/webchat/localchat still receive the same shared future.
        Callers that run standalone (e.g. a scheduled agentic job with no
        prior route() call, or chat(deep_think=True) with no future) can call
        this directly instead.

        mem_limit/know_limit — overridable recall depth. chat(deep_think=True)
        passes DEEP_THINK_MEMORY_LIMIT/DEEP_THINK_KNOWLEDGE_LIMIT here so a
        deep-think turn pulls in noticeably more context than a normal one.

        query_vector — pre-computed _QUERY_INSTRUCT embedding of user_input,
        avoids a redundant HTTP call inside _MemoryBackend.search().

        Returns (memories, knowledge_block).
        """
        memorize = self._get_memorize()
        if memorize is None:
            log.warning("[think] Memory unavailable — skipping memory/KB recall.")
            return [], ""

        recall_query = self._recall_query(user_input)
        query_for_call = query_vector if recall_query == user_input else None

        with _brain_trace.step(
            "think._fetch_memory_and_knowledge",
            layer="recall",
            inputs={
                "user_input": user_input,
                "recall_query": recall_query,
                "enriched_query": recall_query != user_input,
                "mem_limit": mem_limit,
                "know_limit": know_limit,
                "vector_reused": query_for_call is not None,
            },
            factors=[
                "memory + KB fetched concurrently via CONTEXT_POOL",
                f"query_vector reused from route() = {query_for_call is not None}",
            ],
        ) as ctx:
            embedder = getattr(getattr(memorize, "_mem", None), "_embedder", None)
            mem_future = CONTEXT_POOL.submit(memorize.search, recall_query, limit=mem_limit, query_vector=query_for_call)
            know_future = CONTEXT_POOL.submit(
                knowledge_context_for, recall_query, limit=know_limit, max_chars=2000, embedder=embedder
            )
            try:
                memories = mem_future.result(timeout=MEMORY_RECALL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                log.warning("Memory recall timed out after %.1fs; skipping", MEMORY_RECALL_TIMEOUT)
                know_future.cancel()
                memories = []
            except Exception as e:
                log.error("Memory search failed: %s", e)
                know_future.cancel()
                memories = []

            if MEMORY_MIN_SCORE > 0:
                before = len(memories)
                memories = [m for m in memories if m.get("_recall_score", 0.0) >= MEMORY_MIN_SCORE]
                if len(memories) < before:
                    log.debug(
                        "[memory] filtered %d/%d below MEMORY_MIN_SCORE=%.4f",
                        before - len(memories), before, MEMORY_MIN_SCORE,
                    )

            try:
                knowledge_block = know_future.result()
            except Exception as e:
                log.error("Knowledge lookup failed: %s", e)
                knowledge_block = "<knowledge_context>\nLookup failed.\n</knowledge_context>"

            # Per-hit preview so the trace file shows what got recalled.
            hit_preview = []
            for i, m in enumerate((memories or [])[:10], 1):
                hit_preview.append({
                    "rank": i,
                    "score": round(float(m.get("_recall_score", 0.0)), 4),
                    "text": (m.get("memory") or m.get("text") or "")[:400],
                    "kind": m.get("kind"),
                    "pinned": bool(m.get("pinned")),
                })

            ctx.set(
                outputs={
                    "memories_returned": len(memories),
                    "knowledge_chars": len(knowledge_block or ""),
                    "top_hits": hit_preview,
                },
                factors=(ctx.__dict__.get("_step", {}).get("factors", []) if False else [
                    f"min_score filter MEMORY_MIN_SCORE={MEMORY_MIN_SCORE} kept {len(memories)}/{sum(1 for _ in hit_preview)}",
                    f"knowledge_block {'empty' if not knowledge_block else str(len(knowledge_block)) + ' chars'}",
                ]),
            )
            return memories, knowledge_block

    def _resolve_mem_kb(self, user_input: str, mem_kb_future) -> tuple[list[dict], str]:
        """Resolve a pending memory+KB future, or fetch directly if this
        handler was called standalone (no future supplied by route())."""
        if mem_kb_future is not None:
            try:
                return mem_kb_future.result(timeout=MEMORY_RECALL_TIMEOUT)
            except concurrent.futures.TimeoutError:
                log.warning("Memory/KB recall timed out after %.1fs; skipping", MEMORY_RECALL_TIMEOUT)
                return [], "<knowledge_context>\nLookup timed out.\n</knowledge_context>"
            except Exception as e:
                log.error("Memory/KB fetch failed: %s", e)
                return [], "<knowledge_context>\nLookup failed.\n</knowledge_context>"
        return self._fetch_memory_and_knowledge(user_input, query_vector=None)

    def _note_user_activity(self) -> None:
        """Clear the rest flag on real user activity so learn.idle_learner_loop
        sees Aiko is no longer resting and can resume autonomous study."""
        with self._proactive_lock:
            self._proactive_resting = False

    def is_proactive_resting(self) -> bool:
        """True when Aiko is resting and should not start autonomous study.
        Set by set_proactive_resting() (called from main.py's
        ProactiveIdleRunner) and cleared by _note_user_activity() on
        every normal turn. Polled by learn.idle_learner_loop."""
        return self._proactive_resting

    def set_proactive_resting(self, resting: bool) -> None:
        """Set/clear the rest flag for learn.idle_learner_loop.
        Called from main.py's ProactiveIdleRunner."""
        with self._proactive_lock:
            self._proactive_resting = resting

    def _route_intent(self, user_input: str) -> tuple[str, np.ndarray | None]:
        """Quaternary routing: single embedding, four-way decision with a
        high-confidence margin so a close call doesn't get committed to
        agentic (or webchat) just because it happened to be checked first.

        The close-vector label-scoring math itself (normalize + batched
        matmul + top-k mean per label) lives in cognition.reason; this method
        only owns the routing policy (thresholds, gap, LLM tie-break).
        ROUTE_MODE picks the classification method; AGENTIC_MODE_ON gates
        whether "agentic" can ever be the result, independent of method.

        Returns (intent, query_vec). The embedding computed for routing is
        returned so route() can reuse it for memory recall instead of paying
        for a second embed of the same text (see route()).
        """
        with _brain_trace.step("think._route_intent", layer="route",
                               inputs={"user_input": user_input, "mode": _ROUTE_MODE}) as ctx:
            if not _ROUTE_ENABLED:
                ctx.set(outputs={"intent": "localchat", "vector": None},
                        factors=["ROUTE_ENABLED=0 → forced localchat"])
                return "localchat", None
            # Tiny inputs ("dl y", "ok?") carry no signal for semantic
            # label scoring — embedding noise once routed "dl y" to
            # greeting with memory skipped. Fall through to localchat
            # (full recall + store) unless it's an explicit greeting.
            if len((user_input or "").split()) <= 2 and not _is_greeting_only(user_input):
                ctx.set(outputs={"intent": "localchat", "vector": None},
                        factors=["≤2 tokens and no greeting-only regex → localchat default (no semantic score)"])
                return "localchat", None
            if _ROUTE_MODE == "llm_only":
                label = self._classify_quaternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
                ctx.set(outputs={"intent": label, "vector": None, "method": "llm_only"},
                        factors=["llm_only mode bypasses semantic scoring"])
                return label, None

            instruct = _ROUTE_INSTRUCT_QUATERNARY
            memorize = self._get_memorize()
            if memorize is None:
                log.warning("[think] Memory unavailable — intent routing via LLM only")
                label = self._classify_quaternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
                ctx.set(outputs={"intent": label, "vector": None, "method": "llm_fallback"},
                        factors=["memory backend unavailable → fell back to LLM classifier"])
                return label, None
            embedder = getattr(getattr(memorize, "_mem", None), "_embedder", None)
            if embedder is None or not hasattr(embedder, "embed_query"):
                log.warning("[think] Embedder unavailable — intent routing via LLM only")
                label = self._classify_quaternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
                ctx.set(outputs={"intent": label, "vector": None, "method": "llm_fallback"},
                        factors=["embedder unavailable → fell back to LLM classifier"])
                return label, None
            query_vec = embedder.embed_query(user_input, instruct=instruct)
            labels, example_vecs = self._semantic_example_vectors(_ROUTE_QUATERNARY_EXAMPLES, instruct)
            scores = reason.label_scores_topk(query_vec, labels, example_vecs, top_k=_SEMANTIC_LABEL_TOP_K)

            if not _AGENTIC_MODE_ON:
                scores.pop("agentic", None)

            greeting_score = scores.get("greeting", 0.0)
            agentic_score = scores.get("agentic", 0.0)
            webchat_score = scores.get("webchat", 0.0)

            ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
            best_label, best_score = ranked[0] if ranked else ("localchat", 0.0)
            gap = best_score - ranked[1][1] if len(ranked) > 1 else 1.0

            agentic_threshold = float(os.getenv("ROUTE_AGENTIC_THRESHOLD", "0.78"))
            webchat_threshold = float(os.getenv("ROUTE_WEBCHAT_THRESHOLD", "0.72"))
            greeting_threshold = float(os.getenv("ROUTE_GREETING_THRESHOLD", "0.60"))

            log.debug(
                "[route] quaternary scores: greeting=%.3f agentic=%.3f webchat=%.3f best=%s gap=%.3f for: %r",
                greeting_score, agentic_score, webchat_score, best_label, gap, user_input
            )

            ctx.add_extra(scores=dict(scores), best=best_label, gap=gap,
                          thresholds={"agentic":agentic_threshold, "webchat":webchat_threshold,
                                     "greeting":greeting_threshold},
                          method="semantic")

            if best_label == "greeting" and greeting_score >= greeting_threshold and (gap >= _SEMANTIC_ROUTE_MIN_GAP or _is_greeting_only(user_input)):
                ctx.set(outputs={"intent": "greeting", "vector_dim": int(query_vec.shape[0])},
                        factors=[f"greeting_score={greeting_score:.3f} ≥ {greeting_threshold}",
                                 f"gap={gap:.3f} ≥ min_gap={_SEMANTIC_ROUTE_MIN_GAP} OR greeting-only regex hit"])
                return "greeting", query_vec

            if best_label == "agentic" and agentic_score >= agentic_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
                ctx.set(outputs={"intent": "agentic", "vector_dim": int(query_vec.shape[0])},
                        factors=[f"agentic_score={agentic_score:.3f} ≥ {agentic_threshold}",
                                 f"gap={gap:.3f} ≥ min_gap={_SEMANTIC_ROUTE_MIN_GAP}"])
                return "agentic", query_vec
            if best_label == "webchat" and webchat_score >= webchat_threshold and gap >= _SEMANTIC_ROUTE_MIN_GAP:
                ctx.set(outputs={"intent": "webchat", "vector_dim": int(query_vec.shape[0])},
                        factors=[f"webchat_score={webchat_score:.3f} ≥ {webchat_threshold}",
                                 f"gap={gap:.3f} ≥ min_gap={_SEMANTIC_ROUTE_MIN_GAP}"])
                return "webchat", query_vec

            ambiguous = (
                (agentic_score >= agentic_threshold or webchat_score >= webchat_threshold or greeting_score >= greeting_threshold)
                and gap < _SEMANTIC_ROUTE_MIN_GAP
            )
            if ambiguous:
                if _is_greeting_only(user_input):
                    ctx.set(outputs={"intent": "greeting", "vector_dim": int(query_vec.shape[0])},
                            factors=["ambiguous but greeting-only regex overrides → greeting"])
                    return "greeting", query_vec
                if _ROUTE_MODE == "semantic_only":
                    log.debug("[route] semantic_only: ambiguous gap, defaulting localchat")
                    ctx.set(outputs={"intent": "localchat", "vector_dim": int(query_vec.shape[0])},
                            factors=[f"ambiguous gap={gap:.3f}, semantic_only mode → localchat default"])
                    return "localchat", query_vec
                if _ROUTE_MODE == "llm":
                    label = self._classify_quaternary_intent_llm(user_input, allow_agentic=_AGENTIC_MODE_ON)
                    ctx.set(outputs={"intent": label, "vector_dim": int(query_vec.shape[0]), "method": "llm_tiebreak"},
                            factors=[f"ambiguous gap={gap:.3f}, llm mode → LLM tiebreak"])
                    return label, query_vec
                llm_label = self._classify_quaternary_intent_llm(
                    user_input, allow_agentic=_AGENTIC_MODE_ON,
                )
                ctx.set(outputs={"intent": llm_label, "vector_dim": int(query_vec.shape[0]), "method": "llm_tiebreak"},
                        factors=[f"ambiguous gap={gap:.3f}, semantic mode → LLM tiebreak"])
                return llm_label, query_vec

            if _is_greeting_only(user_input):
                ctx.set(outputs={"intent": "greeting", "vector_dim": int(query_vec.shape[0])},
                        factors=["no threshold met but greeting-only regex hit"])
                return "greeting", query_vec

            ctx.set(outputs={"intent": "localchat", "vector_dim": int(query_vec.shape[0])},
                    factors=["no threshold met, no greeting regex → default localchat"])
            return "localchat", query_vec

    def _semantic_example_vectors(self, examples_by_label: dict, instruct: str) -> tuple[list[str], object]:
        """Return cached route-example vectors.

        Hot turns use the in-memory cache. Cold boots can reuse a per-user
        on-disk NumPy archive keyed by the route examples, instruct string, and
        embedding backend metadata.
        If the cache is missing/stale/unreadable, Aiko recomputes and overwrites it.
        """
        cache_key = (id(examples_by_label), instruct)
        with self._semantic_example_cache_lock:
            cached = self._semantic_example_cache.get(cache_key)
            if cached is not None:
                return cached

            mem = self._get_memorize()
            embedder = getattr(getattr(mem, "_mem", None), "_embedder", None)
            if embedder is None or not hasattr(embedder, "embed_queries"):
                raise RuntimeError("embedder unavailable for route-vector cache")
            disk_path = self._route_vector_cache_path(examples_by_label, instruct, embedder)
            if disk_path is not None and disk_path.exists():
                try:
                    with disk_path.open("rb") as f:
                        data = np.load(f, allow_pickle=False)
                        cached = (list(data["labels"].astype(str)), data["vectors"])
                    self._semantic_example_cache[cache_key] = cached
                    return cached
                except Exception as exc:
                    log.debug("[route] ignoring stale vector cache %s: %s", disk_path, exc)

            labels, vectors = reason.embed_example_matrix(embedder, examples_by_label, instruct=instruct)
            cached = (labels, vectors)
            self._semantic_example_cache[cache_key] = cached
            if disk_path is not None:
                try:
                    disk_path.parent.mkdir(parents=True, exist_ok=True)
                    with disk_path.open("wb") as f:
                        np.savez(f, labels=np.asarray(cached[0], dtype=str), vectors=cached[1])
                except Exception as exc:
                    log.debug("[route] could not write vector cache %s: %s", disk_path, exc)
            return cached

    def _route_vector_cache_path(self, examples_by_label: dict, instruct: str, embedder) -> Path | None:
        payload = {
            "examples": examples_by_label,
            "instruct": instruct,
            "embedder": {
                "class": type(embedder).__name__,
                "model": getattr(embedder, "model", None) or getattr(embedder, "model_name", None) or getattr(embedder, "name", None),
                "dims": os.getenv("EMBED_DIMS", ""),
            },
        }
        return reason.cache_vector_path(
            payload,
            cache_dir_env="ROUTE_VECTOR_CACHE_DIR",
            default_dir=_ROUTE_VECTOR_CACHE_DIR,
            per_user=True,
        )

    def _intent_llm_prompt_parts(self, *, allow_agentic: bool, include_greeting: bool) -> tuple[str, str, set[str]]:
        labels = []
        guidance_parts: list[str] = []
        examples: list[str] = []
        valid: set[str] = set()

        if include_greeting:
            labels.append("greeting")
            valid.add("greeting")
            guidance_parts.append(
                "greeting = the entire message is only a salutation, thanks, "
                "or small acknowledgement with no substantive request.\n"
            )
            examples.extend([
                "Message: 'hey Aiko'\nLabel: greeting\n",
                "Message: 'thanks'\nLabel: greeting\n",
            ])

        if allow_agentic:
            labels.append("agentic")
            valid.add("agentic")
            guidance_parts.append(
                "agentic = the message asks for an action/task (write, save, "
                "schedule, debug, plan, remind, research-and-report).\n"
            )
            examples.extend([
                "Message: 'set a reminder for 9pm'\nLabel: agentic\n",
                "Message: 'debug why asyncio.run() hangs'\nLabel: agentic\n",
            ])

        labels.extend(["webchat", "chat"])
        valid.update({"webchat", "chat"})
        guidance_parts.extend([
            "webchat = the message needs current/external information "
            "(news, prices, scores, recent releases, real-time facts) but "
            "is not itself a task.\n",
            "chat = casual conversation, opinions, or something answerable "
            "from general/persona knowledge alone.\n",
        ])
        examples.extend([
            "Message: 'what's the weather in Vancouver right now'\nLabel: webchat\n",
            "Message: 'who won the game last night'\nLabel: webchat\n",
            "Message: 'what do you think about minimalism'\nLabel: chat\n",
            "Message: 'explain semaphores from memory'\nLabel: chat\n",
        ])
        return f"Labels: [{', '.join(labels)}]", "\n".join(guidance_parts + examples), valid

    def _classify_intent_llm(self, user_input: str, *, allow_agentic: bool, include_greeting: bool, log_name: str) -> str:
        if allow_agentic and _AGENTIC_ROUTE_RE.search(user_input):
            return "agentic"
        labels_line, guidance, valid = self._intent_llm_prompt_parts(
            allow_agentic=allow_agentic,
            include_greeting=include_greeting,
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._router_model,
                messages=[{"role": "user", "content": (
                    f"Message: {user_input!r}\n\n"
                    "Output only the route label. No explanation.\n"
                    f"{labels_line}\n"
                    f"{guidance}"
                    "Label:"
                )}],
                stream=False, max_tokens=6, temperature=0.0, top_p=1.0, timeout=LLM_TIMEOUT,
                extra_body={"cache_prompt": _LLM_CACHE_PROMPT},
            )
            label = (resp.choices[0].message.content or "chat").strip().lower()
            label = re.sub(r"[^a-z_].*$", "", label)
            if label not in valid:
                return "localchat"
            return "localchat" if label == "chat" else label
        except Exception as e:
            log.warning("%s LLM routing failed: %s", log_name, e)
            return "localchat"

    def _classify_quaternary_intent_llm(self, user_input: str, allow_agentic: bool = True) -> str:
        """LLM classify with greeting/agentic/webchat/chat labels."""
        return self._classify_intent_llm(
            user_input,
            allow_agentic=allow_agentic,
            include_greeting=True,
            log_name="Quaternary",
        )

    def _classify_ternary_intent_llm(self, user_input: str, allow_agentic: bool = True) -> str:
        """Backward-compatible LLM classifier with no greeting label."""
        return self._classify_intent_llm(
            user_input,
            allow_agentic=allow_agentic,
            include_greeting=False,
            log_name="Ternary",
        )

    def _soft_gate_reply(
        self,
        user_input: str,
        action: str,
        reason: str,
        token_callback=None,
        mem_kb_future=None,
        query_vec: np.ndarray | None = None,
    ) -> str:
        """Handle defer / clarify / degrade_chat without starting agentic tools.

        Used by route() (pre-routing) and agentic_chat() (direct agentic entry).
        Prompts come from attention.soft_user_prompt (one short reply only).
        """
        try:
            from cognition.attention import for_identity
            state = for_identity(current_user_id())
            kind = action if action in {"defer", "clarify", "degrade_chat"} else "other"
            state.record_self_decision(kind, reason)
            state.persist()
        except Exception as exc:
            log.debug("[soft_gate] self-decision record skipped: %s", exc)
        from cognition.attention import soft_user_prompt
        prompt = soft_user_prompt(user_input, action, reason)
        return self.chat(
            prompt,
            token_callback=token_callback,
            _skip_search=True,
            mem_kb_future=mem_kb_future,
            query_vec=query_vec,
            store_turn=True,
        )

    def agentic_chat(self, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None, _from_route: bool = False, system_note: str | None = None, gate_result: tuple[bool, str, str] | None = None) -> str:
        """Delegate task-mode execution to agentic.agentic.

        Runs a bounded self-assessment gate first (attention.should_attempt).
        Critical requests always proceed; discretionary work may degrade to
        chat, defer, or ask for clarification instead of starting the tool loop.
        Direct entry (scheduled jobs) still gates here; normal turns are gated
        earlier in route() with mode=route.
        """
        user_id = current_user_id()
        with self._active_users_lock:
            self._active_user_ids.add(user_id)
        _agentic_t0 = time.monotonic()
        try:
            # Reuse the route decision for routed turns; direct agentic calls gate once.
            if gate_result is None:
                try:
                    from cognition.attention import for_identity
                    state = for_identity(user_id)
                    gate_result = state.should_attempt(user_input, mode="agentic")
                except Exception as exc:
                    log.debug("[agentic_chat] should_attempt skipped: %s", exc)
                    gate_result = (True, "gate unavailable", "proceed")
            ok, reason, action = gate_result
            if not ok:
                log.info("[agentic_chat] should_attempt action=%s reason=%s", action, reason)
                return self._soft_gate_reply(
                    user_input, action, reason, token_callback=token_callback,
                    mem_kb_future=mem_kb_future, query_vec=query_vec,
                )

            memorize = self._get_memorize()
            mem_inner = getattr(memorize, "_mem", None) if memorize is not None else None
            embedder = getattr(mem_inner, "_embedder", None)
            cap_vec = embedder.embed_query(
                user_input,
                instruct="Which capability/tool domain applies to this task?",
            ) if embedder is not None else None
            if system_note and system_note.strip():
                # Agentic prompt plumbing lives in run_agentic_chat; carry the
                # notices as marked situational context so drained notes are
                # never silently dropped on tool turns.
                user_input = f"{user_input}\n\n[{_format_system_notices(system_note)}]"
            response = run_agentic_chat(self, user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec, cap_vec=cap_vec)
            return response
        finally:
            # Only record latency if called directly (not from route, which already records)
            if not _from_route:
                try:
                    from cognition.attention import for_identity
                    for_identity(user_id).record_turn_latency(time.monotonic() - _agentic_t0)
                except Exception:
                    pass
            with self._active_users_lock:
                self._active_user_ids.discard(user_id)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()

    def webchat(self, user_input: str, token_callback=None, mem_kb_future=None, query_vec: np.ndarray | None = None, system_note: str | None = None) -> str:
        """Web-aware chat: web_search + optional webfetch fallback."""
        # Guard: experience-sharing narration must never be answered from
        # search results ("Answer ONLY using these results" would discard
        # what the user just told us). Fall back to plain chat, which keeps
        # memory recall and turn writeback intact.
        if _is_personal_sharing(user_input):
            log.info("[route] webchat override -> chat (personal sharing)")
            return self.chat(user_input, token_callback=token_callback, mem_kb_future=mem_kb_future, query_vec=query_vec, system_note=system_note)
        speak = self._get_speak()
        if speak and speak.is_playing():
            speak.stop()

        # Memory + KB — either resolved from route()'s pre-intent future,
        # or fetched directly if this was called standalone.
        memories, knowledge_block = self._resolve_mem_kb(user_input, mem_kb_future)
        from cognition.attention import for_identity
        memories = for_identity(current_user_id()).prioritize_memories(user_input, memories)
        memory_block = self._get_memorize().format_for_context(
          memories, query=user_input, query_vector=query_vec
        )
        persona_block = self._get_memorize().persona_context()
        situation_block = ""
        metacognitive_block = ""
        try:
            state = for_identity(current_user_id())
            situation_block = state.situation_context(user_input, memories, knowledge_block)
            metacognitive_block = state.metacognitive_context(user_input, memories)
        except Exception:
            pass

        # Build base system (persona + memory + knowledge)
        system = self._current_system_prompt()
        system += "\n\n" + bioclock.current_datetime_block()
        if persona_block:
            system = f"{system}\n\n{persona_block}"
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        if situation_block:
            system = f"{system}\n\n{situation_block}"
        if metacognitive_block:
            system = f"{system}\n\n{metacognitive_block}"
        system = f"{system}\n\n{knowledge_block}"
        notices_block = _format_system_notices(system_note)
        if notices_block:
            system = f"{system}\n\n{notices_block}"

        # Search directly with the raw user input — same approach as /web.
        # No LLM-based query condensation: it adds latency, depends on a
        # small router model that often produces worse queries than the
        # original text, and /web already proves the raw path works.
        display_name = current_display_name()
        if token_callback:
            token_callback("__STATUS__:searching\n")
            token_callback(f"__SEARCHING__:{user_input}\n")

        max_results = int(os.getenv("SEARXNG_MAX_RESULTS", 3))
        from agentic.toolkit.websearch import web_search as _web_search
        from urllib.parse import urlparse as _urlparse

        def _format_hits(query: str, results: list) -> tuple[str, list]:
            if not results:
                return "", []
            lines = [f"[Web search results for: {query}]"]
            sources: list[dict] = []
            for i, result in enumerate(results, 1):
                title = (result.get("title") or "").strip()
                url = (result.get("url") or "").strip()
                content = (result.get("content") or "").strip()
                lines.append(f"{i}. {title}\n   {url}\n   {content}")
                domain = ""
                try:
                    domain = _urlparse(url).netloc.lower().removeprefix("www.")
                except Exception:
                    domain = ""
                if url:
                    sources.append({"title": title or url, "url": url, "domain": domain})
            context = "\n\n".join(lines) + f"\n\nUser asked: {query}"
            return context, sources

        results, search_err = _web_search(user_input, max_results)
        if search_err:
            log.warning("[webchat] search error: %s", search_err)
        context, sources = _format_hits(user_input, results or [])

        if not context:
            log.info("[webchat] First search returned nothing, retrying once...")
            if token_callback:
                token_callback("__STATUS__:retry\n")
                token_callback("__RETRYING__\n")
            try:
                results, search_err = _web_search(user_input, 1)
                if search_err:
                    log.warning("[webchat] retry error: %s", search_err)
                context, sources = _format_hits(user_input, results or [])
            except Exception as e:
                log.warning("[webchat] Retry failed: %s", e)
                context, sources = "", []

        if sources and token_callback:
            token_callback("__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n")

        # Inject web context if available
        if context:
            if token_callback:
                token_callback("__STATUS__:ok\n")
            system = (
                f"{system}\n\n"
                f"<search_results query='{user_input}'>\n"
                f"Answer ONLY using these search results:\n\n"
                f"{context}\n"
                f"</search_results>"
            )
        else:
            if token_callback:
                token_callback("__STATUS__:offline\n")
            system = (
                f"{system}\n\n"
                "<search_failed>\n"
                f"Web search returned no usable results. You are speaking with {display_name}.\n"
                "Respond as Aiko in one or two short natural sentences:\n"
                "- Briefly acknowledge you could not reach live internet information (vary wording; "
                "no fixed script, no system tokens, no phrases like 'using local knowledge').\n"
                "- Do NOT invent time-sensitive facts (weather, scores, headlines, prices).\n"
                "- If memory or knowledge context genuinely helps a non-live question, offer that "
                "briefly; otherwise say you do not have current information.\n"
                "- Stay in character. No meta commentary about tools or pipelines.\n"
                "</search_failed>"
            )

        # Build message history (same as chat())
        llm_prompt = user_input
        if self._reasoning:
            llm_prompt = f"{user_input}\n\nThink through this carefully."

        with self._history_lock:
            self._history.append({"role": "user", "content": user_input})
            if len(self._history) > CONTEXT_WINDOW_TURNS * 10:
                self._history = self._history[-(CONTEXT_WINDOW_TURNS * 10):]
            trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]

        trimmed = self._sanitize_history(trimmed)
        if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != user_input:
            trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]

        # Log debug info
        self.last_prompt_debug = {
            "mode": "webchat",
            "system_prompt": system,
            "memory_prompt": memory_block or "<memory_context>\nNo memories.\n</memory_context>",
            "knowledge_prompt": knowledge_block,
            "web_prompt": _extract_search_results_block(system),
            "previous_chat_messages": [dict(m) for m in trimmed],
        }

        # Live working-memory (<grasp>) block — same explicit injection as
        # chat(); replaces the old grasp_hub _stream_response wrapper.
        try:
            _wm_mem = self._get_memorize()
            if _wm_mem is not None:
                _wm_block = _wm_mem.wm_context_block()
                if _wm_block:
                    system = f"{system}\n\n{_wm_block}"
        except Exception:
            pass

        # Meta-cognitive self-check (Anthropic constitutional AI style) — verify
        # alignment, evidence sufficiency, no invention, before streaming.
        meta_check_notes: list[str] = []
        hit_count = 0
        try:
            memorize = self._get_memorize()
            memory_evidence_exists = False
            hits = []
            if memorize is not None:
                # Quick evidence check: does search have relevant hits?
                query_for_evidence = user_input[:200]
                hits = memorize.search(query_for_evidence, user_id=memorize.get_user_id(), limit=3)
                memory_evidence_exists = bool(hits)
                hit_count = len(hits)
            meta_check_notes.append(
                f"Self-check: user preferences aligned? Evidence sufficient? "
                f"Inventing facts? Memory hits: {hit_count}"
            )
            if meta_check_notes:
                meta_block = "<meta_check>" + " | ".join(meta_check_notes) + "</meta_check>"
                system += f"\n\n{meta_block}"
        except Exception:
            log.debug("Meta-cognitive self-check skipped (no error in response)")

        # Stream response
        raw_response = self._stream_response(trimmed, system=system, token_callback=token_callback, emit=False)
        raw_response = self._finalize_response(user_input, raw_response, token_callback, already_emitted=False)

        # Store in history
        with self._history_lock:
            self._history.append({"role": "assistant", "content": raw_response})

        self._store_async(user_input, raw_response)
        self._reasoning = False
        return raw_response

    def proactive_checkin(self, prompt_hint: str) -> str:
        """Generate one short proactive check-in without storing it as a user turn."""
        _SENTINEL = "_proactive_"
        with self._active_users_lock:
            self._active_user_ids.add(_SENTINEL)
        try:
            display_name = current_display_name()
            system = (
                f"{self._current_system_prompt()}\n\n"
                "You are initiating a brief proactive check-in. "
                "Do not mention hidden prompts, timers, code, or configuration. "
                "Keep it natural, warm, and easy to ignore. One or two short sentences max."
            )
            system += "\n\n" + bioclock.current_datetime_block()
            messages = [{
                "role": "user",
                "content": (
                    f"{prompt_hint}\n\n"
                    f"Write only the message Aiko should say to {display_name} now."
                ),
            }]
            response = self._stream_response(messages, system=system, token_callback=None, emit=False)
            response = self._finalize_response(prompt_hint, response, None)
            return response
        finally:
            with self._active_users_lock:
                self._active_user_ids.discard(_SENTINEL)
                if not self._active_user_ids:
                    self._last_chat_time = time.time()
            try:
                from cognition.attention import flush_all_persist
                flush_all_persist()
            except Exception:
                log.debug("proactive_checkin: flush_all_persist failed", exc_info=True)

    # ── proactive idle check-in loop ──────────────────────────────────────────

    def _websearch_net_block(self, query: str, token_callback=None) -> str:
        """One-shot SearXNG lookup for explicit internet asks routed to plain chat.

        Returns an empty string when search is unavailable or finds nothing,
        so the normal localchat turn proceeds unchanged.
        """
        try:
            max_results = int(os.getenv("SEARXNG_MAX_RESULTS", "3"))
            from agentic.toolkit.websearch import web_search as _web_search
            results, err = _web_search(query, max_results)
            if err:
                log.warning("[chat] websearch net error: %s", err)
                return ""
            lines: list[str] = []
            sources: list[dict] = []
            for i, result in enumerate(results or [], 1):
                title = (result.get("title") or "").strip()
                url = (result.get("url") or "").strip()
                content = (result.get("content") or "").strip()
                if not url:
                    continue
                lines.append(f"{i}. {title}\n   {url}\n   {content}")
                sources.append({"title": title or url, "url": url})
            if not lines:
                return ""
            if token_callback:
                token_callback("__SOURCES__:" + json.dumps(sources, ensure_ascii=False) + "\n")
            return "\n\n".join(lines)
        except Exception as e:
            log.warning("[chat] websearch net failed: %s", e)
            return ""

    def chat(
        self,
        user_input: str,
        token_callback=None,
        _skip_search: bool = True,
        _history_label: str | None = None,
        mem_kb_future=None,
        *,
        skip_memory: bool = False,
        store_turn: bool = True,
        query_vec: np.ndarray | None = None,
        websearch_net: bool = True,
        system_note: str | None = None,
        deep_think: bool = False,
    ) -> str:
        """Standard chat: persona plus optional memory/KB context.

        deep_think — see module docstring "Deep-think mode". Sets
        self._reasoning + self._deep_think for the duration of this call
        (both are cleared again before returning), widens memory/knowledge
        recall to DEEP_THINK_MEMORY_LIMIT/DEEP_THINK_KNOWLEDGE_LIMIT when no
        mem_kb_future was already supplied, and injects _DEEP_THINK_GUIDE —
        a structured multi-angle reasoning scaffold — into the system prompt.
        """
        speak = self._get_speak()
        if speak and speak.is_playing():
            speak.stop()

        if deep_think:
            self._reasoning = True
            self._deep_think = True

        # Fused soft-gate prompts (see _soft_gate_reply) carry an injected
        # "[Style only — ...]" control block for the LLM. Everything below
        # except llm_prompt uses the raw user text so the directive never
        # leaks into recall, history, cognitive state or memory writes.
        raw_input = _strip_style_directives(user_input)

        with _brain_trace.step("think.chat", layer="context",
                               inputs={"user_input": user_input, "raw_input": raw_input, "skip_memory": skip_memory,
                                       "store_turn": store_turn, "websearch_net": websearch_net,
                                       "deep_think": deep_think}) as ctx:
            situation_block = ""
            metacognitive_block = ""
            if skip_memory:
                memories = []
                knowledge_block = ""
                memory_block = ""
                persona_block = ""
            else:
                memorize = self._get_memorize()
                from cognition.attention import for_identity
                if deep_think and mem_kb_future is None:
                    # No pre-started future (deep-think fast path bypasses
                    # route()'s CONTEXT_POOL future) — fetch directly with
                    # the wider deep-think recall limits instead of the
                    # normal MEMORY_RECALL_LIMIT/KNOWLEDGE_RECALL_LIMIT.
                    memories, knowledge_block = self._fetch_memory_and_knowledge(
                        raw_input, query_vector=query_vec,
                        mem_limit=DEEP_THINK_MEMORY_LIMIT,
                        know_limit=DEEP_THINK_KNOWLEDGE_LIMIT,
                    )
                else:
                    memories, knowledge_block = self._resolve_mem_kb(raw_input, mem_kb_future)
                memories = for_identity(current_user_id()).prioritize_memories(raw_input, memories)
                deep_think_meta = {}
                if deep_think:
                    memories, deep_think_meta = _deep_think_rerank(memories, raw_input)
                    self.last_deep_think_summary = _deep_think_format_summary(
                        query=raw_input,
                        memories=memories,
                        knowledge_block=knowledge_block or "",
                        meta=deep_think_meta,
                        web_present=False,
                    )
                    try:
                        _brain_trace.record_step(
                            "think.deep_think.evidence",
                            layer="deep_think",
                            inputs={"query": raw_input},
                            outputs=_deep_think_trace_payload(
                                query=raw_input,
                                memories=memories,
                                knowledge_block=knowledge_block or "",
                                meta=deep_think_meta,
                                web_present=False,
                            ),
                            factors=[
                                f"rerank kept {deep_think_meta.get('kept_count')}/{deep_think_meta.get('input_count')}",
                                f"contradictions={deep_think_meta.get('contradictions', 0)}",
                            ],
                        )
                    except Exception:
                        pass
                else:
                    self.last_deep_think_summary = None
                memory_block = memorize.format_for_context(
                  memories, query=raw_input, query_vector=query_vec
                ) if memorize is not None else ""
                persona_block = memorize.persona_context() if memorize is not None else ""
                try:
                    from cognition.attention import for_identity
                    situation_block = for_identity(current_user_id()).situation_context(raw_input, memories, knowledge_block)
                    metacognitive_block = for_identity(current_user_id()).metacognitive_context(raw_input, memories)
                except Exception:
                    pass

            core_system, volatile_system = self._current_system_prompt_parts(raw_input)
            if not skip_memory:
                if persona_block:
                    volatile_system = f"{volatile_system}\n\n{persona_block}"
                if memory_block:
                    volatile_system = f"{volatile_system}\n\n{memory_block}"
                if situation_block:
                    volatile_system = f"{volatile_system}\n\n{situation_block}"
                if metacognitive_block:
                    volatile_system = f"{volatile_system}\n\n{metacognitive_block}"
                if not memory_block:
                    volatile_system += "\n\n<memory_context>\nNo relevant memories found.\n</memory_context>"
                if knowledge_block:
                    volatile_system = f"{volatile_system}\n\n{knowledge_block}"
                # Codebase RAG — when user explicitly asks from your codebase/code
                if not skip_memory and any(k in (raw_input or "").lower() for k in ("codebase", "from your code", "from your codebase", "attention gate", "how does your code", "where is", "repo", "source file")):
                    try:
                        from cognition.knowledge.codebase import codebase_context_for
                        memorize = self._get_memorize()
                        embedder = memorize.embedder() if memorize is not None else None
                        cb_block = codebase_context_for(raw_input, limit=4, max_chars=3500, embedder=embedder)
                        if cb_block and "No matching codebase" not in cb_block:
                            volatile_system = f"{volatile_system}\n\n{cb_block}"
                    except Exception as e:
                        log.debug("codebase_context inject failed: %s", e)

            if not skip_memory and _should_use_local_knowledge(raw_input):
                try:
                    memorize = self._get_memorize()
                    embedder = memorize.embedder() if memorize is not None else None
                    wiki_context = wiki_knowledge_context_for(
                        raw_input, limit=3, max_chars=3000,
                        embedder=embedder,
                    )
                    if wiki_context:
                        volatile_system = f"{volatile_system}\n\n{wiki_context}"
                except Exception as e:
                    log.error("Local wiki-knowledge lookup failed: %s", e)

            if (
                not skip_memory
                and websearch_net
                and _CHAT_WEBSEARCH_NET_ENABLED
                and _WEBSEARCH_HINT_RE.search(raw_input)
            ):
                net_context = self._websearch_net_block(raw_input, token_callback)
                if net_context:
                    volatile_system = (
                        f"{volatile_system}\n\n"
                        f"<search_results query='{raw_input}'>\n"
                        f"Live web results — use them when they are relevant; do not invent time-sensitive facts:\n\n"
                        f"{net_context}\n"
                        f"</search_results>"
                    )

            _wm_mem = self._get_memorize()
            if _wm_mem is not None:
                try:
                    _wm_block = _wm_mem.wm_context_block()
                    if _wm_block:
                        volatile_system = f"{volatile_system}\n\n{_wm_block}"
                except Exception:
                    pass

            if deep_think:
                volatile_system = f"{volatile_system}\n\n{_DEEP_THINK_GUIDE}"
                volatile_system = (
                    f"{volatile_system}\n\n"
                    + _deep_think_evidence_preamble(
                        memories if not skip_memory else [],
                        knowledge_block if not skip_memory else "",
                        web_present=bool(
                            not skip_memory and websearch_net
                            and _CHAT_WEBSEARCH_NET_ENABLED
                            and _WEBSEARCH_HINT_RE.search(raw_input)
                        ),
                    )
                )
                _tool_hint = _deep_think_tool_hint(
                    raw_input,
                    memories if not skip_memory else [],
                    knowledge_block if not skip_memory else "",
                )
                if _tool_hint:
                    volatile_system = f"{volatile_system}\n\n{_tool_hint}"

            volatile_system = f"{volatile_system}\n\n{bioclock.current_datetime_block()}".strip()
            notices_block = _format_system_notices(system_note)
            if notices_block:
                volatile_system = f"{volatile_system}\n\n{notices_block}"

            llm_prompt = user_input
            if deep_think:
                llm_prompt = (
                    f"{user_input}\n\n"
                    "Take real time to think this through thoroughly before answering "
                    "— see the deep-thinking guidance above."
                )
            elif self._reasoning:
                llm_prompt = f"{user_input}\n\nThink through this carefully."

            with self._history_lock:
                self._history.append({"role": "user", "content": raw_input})
                if len(self._history) > CONTEXT_WINDOW_TURNS * 10:
                    self._history = self._history[-(CONTEXT_WINDOW_TURNS * 10):]
                trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]

            trimmed = self._sanitize_history(trimmed)
            if trimmed and trimmed[-1]["role"] == "user" and llm_prompt != user_input:
                trimmed = trimmed[:-1] + [{"role": "user", "content": llm_prompt}]

            self.last_prompt_debug = {
                "mode": "greeting" if skip_memory else ("deep_think" if deep_think else "localchat"),
                "system_prompt": core_system + ("\n\n" + volatile_system if volatile_system else ""),
                "memory_prompt": memory_block or "<memory_context>\nNo memories.\n</memory_context>",
                "knowledge_prompt": knowledge_block,
                "web_prompt": "",
                "previous_chat_messages": [dict(m) for m in trimmed],
            }
            _dump_full_prompt(self.last_prompt_debug)

            # The two halves of the system prompt get sent as two separate
            # system messages with conversation history sandwiched between —
            # see _stream_response for the cache_prompt reasoning.
            _brain_trace.record_step(
                "think.chat.prompt_assembled",
                layer="context",
                outputs={
                    "core_chars": len(core_system),
                    "volatile_chars": len(volatile_system),
                    "memory_block_chars": len(memory_block or ""),
                    "memory_block_preview": (memory_block or "")[:600],
                    "history_turns": len(trimmed),
                },
                factors=[
                    f"memories reranked: {len(memories)}",
                    f"situation/metacognitive added: {bool(situation_block)}/{bool(metacognitive_block)}",
                    f"wiki trigger: {_should_use_local_knowledge(raw_input)}",
                    f"websearch_net trigger: {_WEBSEARCH_HINT_RE.search(raw_input) is not None}",
                    f"deep_think: {deep_think} (token_scale={DEEP_THINK_TOKEN_SCALE if deep_think else (_REASONING_SCALE if self._reasoning else 1)})",
                ],
            )

            raw_response = self._stream_response(
                trimmed,
                system=core_system,
                system_tail=volatile_system,
                token_callback=token_callback,
                emit=False,
            )
            raw_response = self._finalize_response(raw_input, raw_response, token_callback, already_emitted=False)

            with self._history_lock:
                self._history.append({"role": "assistant", "content": raw_response})

            if store_turn:
                self._store_async(raw_input, raw_response)
            self._reasoning = False
            self._deep_think = False
            ctx.set(outputs={"reply_chars": len(raw_response or "")},
                    factors=[f"LLM stream done; reply {len(raw_response or '')} chars"])
            return raw_response

    def web_search(self, query: str, token_callback=None) -> str:
        """Explicit /web command path."""
        context = web_search_context(query)
        if not context or "no results" in context or "failed" in context:
            msg = f"[no results for: {query}]"
            if token_callback:
                token_callback(msg)
            return msg
        # websearch_net=False — this turn already carries fetched results;
        # re-triggering the net on the context blob would double-search.
        return self.chat(context, token_callback=token_callback, _skip_search=True, _history_label=query, websearch_net=False)

    def reset_context(self) -> None:
        with self._history_lock:
            self._history.clear()
        try:
            _wm_mem = self._get_memorize()
            if _wm_mem is not None:
                _wm_mem.wm_reset()
        except Exception:
            pass

    def last_turn(self) -> tuple[str, str] | None:
        with self._history_lock:
            history_snapshot = list(self._history)
        users = [m["content"].strip() for m in history_snapshot if m.get("role") == "user" and (m.get("content") or "").strip()]
        assistants = [m["content"].strip() for m in history_snapshot if m.get("role") == "assistant" and (m.get("content") or "").strip()]
        if not users or not assistants:
            return None
        return users[-1], assistants[-1]

    def set_reasoning(self, enabled: bool) -> None: self._reasoning = enabled

    def set_speak(self, speak) -> None:
        with self._speak_lock:
            old = self._speak
            self._speak = speak
        if old is not None and old is not speak:
            old.stop()   # AikoSpeak.stop() is idempotent/safe to call even if already stopped

    def compare_and_set_speak(self, expected, new_value) -> bool:
        """Atomically set _speak to new_value only if it's still `expected`.
        Used by the proactive-checkin save/mute/restore sequence in
        orchestrate.py: if a concurrent explicit toggle (e.g. /voice) changed
        _speak while we were muted for the check-in, that change is more
        recent and should win — we skip the restore instead of stomping it.
        Returns True if the swap happened, False if _speak had already moved."""
        with self._speak_lock:
            if self._speak is expected:
                old = self._speak
                self._speak = new_value
                swapped = True
            else:
                old = None
                swapped = False
        if swapped and old is not None and old is not new_value:
            old.stop()
        return swapped

    def _get_speak(self):
        with self._speak_lock:
            return self._speak

    def set_memorize(self, memorize) -> None:
        """Inject the memory backend after boot. Thread-safe against concurrent reads."""
        with self._memorize_lock:
            self._memorize = memorize

    def _get_memorize(self):
        with self._memorize_lock:
            return self._memorize

    def wait_for_memory(self, timeout: float | None = None) -> bool:
        """Block until AikoMemorize's async write queue drains, or timeout
        elapses. The queue itself now lives in cognition.memory.memorize; this is a
        thin passthrough kept for call sites that only know about the
        AikoThink instance. No longer called from agentic.agentic's turn
        start (see run_agentic_chat) — draining there was removed since
        the write's own idle-grace window plus real turn latency meant it
        rarely caught anything. Still available for any caller that
        genuinely needs to block until writes are flushed (e.g. shutdown).
        """
        return self._get_memorize().wait_for_writes(timeout=timeout)

    def prewarm_caches(self) -> None:
        """Warm both semantic caches used by first-turn routing/capability
        matching, so the first real user turn never pays an embedding cost.

        Route exemplars (self._semantic_example_vectors): in-memory cache,
        then per-user on-disk npz cache (cognition.reason.cache_vector_path),
        then compute+persist if both miss.

        Capability trigger embeddings (agentic.capability._get_trigger_embedding):
        same three-tier pattern, sharing the same cache_vector_path helper —
        in-memory dict first, then on-disk npz, then compute+persist.

        On a warm disk cache, this whole call is disk loads only, no
        embedding calls. On a cold cache (first boot, or after a trigger/
        exemplar edit), it pays the full embed cost once and persists it.

        No-ops (with a log line) if the memory backend isn't wired up yet —
        callers don't need to check that themselves. Never raises; boot
        callers can fire this and move on regardless of outcome.
        """
        if self._get_memorize() is None:
            log.info("[think] Skipping semantic cache prewarm — no memory backend.")
            return
        try:
            # Prewarm intent routing cache
            self._semantic_example_vectors(_ROUTE_TERNARY_EXAMPLES, _ROUTE_INSTRUCT_TERNARY)    # prewarm routing cache in on-disk npz

            # Prewarm capability trigger embeddings (used by agentic_chat -> match_capabilities)
            from agentic.capability import CAPABILITIES, _get_trigger_embedding            # for loading intents and tools from Aiko's capabilities
            embedder = self._get_memorize().embedder()                                    # shared embedder — no DB needed, works pre-login
            for cap in CAPABILITIES.values():                                              # loop through all Aiko's capabilities
                _get_trigger_embedding(cap, embedder)                                      # load all the semantic vectors into cache

            log.info("[think] Semantic exemplar cache warmed (intent + capabilities)")    # log success
        except Exception:                                                                 # if error,
            log.exception("[think] Semantic exemplar prewarm failed")                     # log failure — single point, full traceback

    def handle_scheduled_job(self, job: DueJob) -> None:
        """Announce or execute a due scheduled job without blocking the scheduler."""
        text = f"{job.title}. {job.task}"
        log.info("[schedule] due %s action=%s: %s", job.id, job.action, text)
        if job.action == "announce":
            _play_beep()
            speak = self._get_speak()
            if speak:
                speak.speak(text)
            else:
                log.info(f"Aiko scheduled job: {text}")
            return
        # A plain threading.Thread starts with a fresh contextvars context,
        # not the caller's — so without copying it here, current_user_id()
        # (read throughout this class) would fall back to its default on
        # this worker thread even though ScheduleRunner._run() correctly
        # set it for whichever user's job this is.
        ctx = contextvars.copy_context()
        if job.action == "tool":
            threading.Thread(target=ctx.run, args=(self._run_scheduled_tool_job, job), daemon=True).start()
            return
        threading.Thread(target=ctx.run, args=(self._run_scheduled_agentic_job, job), daemon=True).start()

    def _run_scheduled_tool_job(self, job: DueJob) -> None:
        """Invoke one registered agentic tool from a schedule.json record."""
        tool_call = job.tool_call or {}
        try:
            from agentic.agentic import invoke_registered_tool
            result = invoke_registered_tool(tool_call.get("name", ""), tool_call.get("arguments", {}))
            log.info("Scheduled tool job %s completed: %s", job.id, result)
        except Exception as e:
            log.error("Scheduled tool job %s failed: %s", job.id, e)

    def _run_scheduled_agentic_job(self, job: DueJob) -> None:
        """Run a scheduled autonomous task through Aiko's agent loop."""
        prompt = (
            "Scheduled job due. Use only local available tools. If external action "
            "is unavailable, draft/save the best local artifact and state next step.\n\n"
            f"Title: {job.title}\nTask: {job.task}"
            + (f"\n\nScheduled skill instructions:\n{job.skill}" if job.skill else "")
        )
        try:
            self.agentic_chat(prompt)
        except Exception as e:
            log.error("Scheduled agentic job failed: %s", e)

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, text: str, token_callback=None) -> None:
        if not text:
            return

        # Always drive the TUI callback directly, regardless of TTS
        if token_callback:
            words = text.split(" ")
            for i, word in enumerate(words):
                token_callback(word if i == 0 else " " + word)
                time.sleep(float(os.getenv("EMIT_DELAY", 0.005)))

        # TTS runs independently
        speak = self._get_speak()
        if speak:
            speak.feed(text)
            speak.play_async()

    def _emit_finalized_response(self, text: str, token_callback=None) -> None:
        """Emit only text that has completed review and conscience gating."""
        if not text:
            return
        speak = self._get_speak()
        karaoke_text = bool(
            speak
            and token_callback
            and getattr(speak, "karaoke_text", False)
            and not self._reasoning
        )
        if not karaoke_text:
            self._emit(text, token_callback=token_callback)
            return

        speak.start_speech_stream(token_callback)
        sentences, remainder = split_stream_sentences(text)
        for sentence in sentences:
            speak.feed_speech_stream(sentence)
        if remainder.strip():
            speak.feed_speech_stream(remainder)
        speak.stop_speech_stream()

    @staticmethod
    def _messages_without_system(all_messages: list[dict]) -> list[dict]:
        """Fallback for chat templates that reject `system` role (Jinja: Only user, assistant and tool roles are supported).

        Merges every system message into the next user turn (or appends as user
        if no user follows) so the prompt content is preserved without the
        system role. Used as automatic retry when llama.cpp returns 500 with
        'got system'.
        """
        out: list[dict] = []
        pending: list[str] = []
        for m in all_messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "system":
                if content.strip():
                    pending.append(content)
                continue
            if pending:
                prefix = "\n\n".join(pending)
                pending = []
                if role == "user":
                    m = {"role": "user", "content": prefix + "\n\n" + content}
                else:
                    # Flush system block as a user turn before an assistant/tool turn
                    out.append({"role": "user", "content": prefix})
            out.append(m)
        if pending:
            out.append({"role": "user", "content": "\n\n".join(pending)})
        # Ensure at least one user turn exists
        if not out:
            out = [{"role": "user", "content": "\n\n".join(pending)}]
        return out

    @staticmethod
    def _is_system_role_error(exc: Exception) -> bool:
        msg = str(exc)
        if "got system" not in msg and "system role" not in msg.lower():
            return False
        # The Jinja template error from llama-server truncates the inner
        # raise_exception(...) call to "...Only user, assistant and tool roles ar..."
        # which then fails the literal substring check below. Match on either
        # the full phrase OR the prefix that always survives truncation, plus
        # the templated "role" indicator.
        return (
            "Only user, assistant and tool roles are supported" in msg
            or "Only user, assistant and tool roles ar" in msg
            or "raise_exception" in msg and "system" in msg
        )

    def _stream_response(self, messages: list[dict], system: str = "", token_callback=None, emit: bool = False, system_tail: str = "") -> str:
        """Collect a complete model response without exposing it to UI or TTS.

        ``emit`` remains as a compatibility argument for callers and tests, but
        final output is deliberately deferred to _finalize_response so the
        complete draft passes gate_speak before any token or audio is emitted.
        """
        full_response = []
        if self._reasoning:
            # Deep-think mode gets its own, much larger scale than ordinary
            # self._reasoning — see module docstring "Deep-think mode".
            scale = DEEP_THINK_TOKEN_SCALE if self._deep_think else _REASONING_SCALE
            base_tokens = _BASE_TOKENS * scale
        else:
            base_tokens = _BASE_TOKENS
        max_tokens = _effective_max_tokens(base_tokens)

        # Message layout for llama-server cache_prompt reuse:
        #   [system core] [history ...] [system volatile tail] [user]
        # The tail sits right before the newest user turn so the byte-stable
        # core + history prefix survives across turns; without a tail this
        # degrades to the classic [system] + messages layout.
        all_messages = [{"role": "system", "content": system}] + messages if system else list(messages)
        if system_tail and all_messages and all_messages[-1].get("role") == "user":
            all_messages = all_messages[:-1] + [{"role": "system", "content": system_tail}, all_messages[-1]]
        elif system_tail:
            all_messages = all_messages + [{"role": "system", "content": system_tail}]
        # No-think suffix applies to the request payload only (see helper).
        all_messages = _apply_no_think(all_messages)

        self.last_usage = {
            "prompt_messages": all_messages,
            "completion_text": "",
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }

        stream_success = False
        reasoning_buffer: list[str] = []

        try:
            stream = self._client.chat.completions.create(
                model=self._llm_model, messages=all_messages, stream=True,
                max_tokens=max_tokens,
                temperature=float(os.getenv("TEMPERATURE", 0.72)),
                top_p=float(os.getenv("TOP_P", 0.90)),
                stop=LLM_STOP_SEQUENCES,
                timeout=LLM_TIMEOUT,
                extra_body=_llm_extra_body(),
            )
            _brain_trace.record_step(
                "think._stream_response.llm_open",
                layer="stream",
                outputs={"model": self._llm_model, "max_tokens": max_tokens,
                         "cache_prompt": _LLM_CACHE_PROMPT,
                         "no_think": _LLM_NO_THINK,
                         "n_messages": len(all_messages)},
                factors=["cache_prompt=True reuses KV across turns for stable core prefix"],
            )
            for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                token = (delta.content or "") if delta else ""
                # Thinking-channel harvest: reasoning-capable templates
                # (granite-4, MiniCPM-think, …) may emit the whole turn as
                # thinking tokens. Some servers surface those as
                # `reasoning_content` deltas with EMPTY content — without
                # this we burn the full token budget and see "empty".
                # Collected for diagnostics; never served raw (see below).
                reason_part = ""
                if delta is not None:
                    for attr in ("reasoning_content", "reasoning"):
                        try:
                            val = getattr(delta, attr, None)
                        except Exception:
                            val = None
                        if val:
                            reason_part = str(val)
                            break
                    if not reason_part and isinstance(delta, dict):
                        for key in ("reasoning_content", "reasoning"):
                            if delta.get(key):
                                reason_part = str(delta[key])
                                break
                if reason_part:
                    reasoning_buffer.append(reason_part)

                full_response.append(token)

            text = "".join(full_response).strip()
            reasoning_text = "".join(reasoning_buffer).strip()
            _note_stream_reasoned(bool(reasoning_text) and not bool(text))
            if reasoning_text:
                _note_thinker_detected()
                # Recompute from the pre-budget base: the flag may have
                # flipped mid-stream, in which case the in-turn fallback
                # below must already carry headroom (same-turn recovery).
                max_tokens = _effective_max_tokens(base_tokens)
                # Never served raw, but decisive for diagnosis: content-empty
                # + reasoning-full means a thinking model spent the budget in
                # the think channel (raise max_tokens or disable thinking
                # server-side); content-empty + reasoning-empty means the
                # template emitted EOS immediately (template/EOS mismatch).
                log.warning(
                    "LLM stream thinking-channel: %d reasoning chars, %d content chars (model=%s)",
                    len(reasoning_text), len(text), self._llm_model,
                )
                _brain_trace.record_step(
                    "think._stream_response.thinking",
                    layer="stream",
                    outputs={"model": self._llm_model,
                             "reasoning_chars": len(reasoning_text),
                             "content_chars": len(text),
                             "reasoning_preview": reasoning_text[:300]},
                    factors=["thinking-channel tokens are never served to the user"],
                )
            if text:
                self.last_usage["completion_text"] = text
                stream_success = True
        except Exception as e:
            if self._is_system_role_error(e):
                log.warning(f"LLM stream system-role rejected (will retry without system): {e}")
            else:
                log.error(f"LLM stream failed: {e}")
            # Auto-retry without system role if template rejects it (ministral-type Jinja)
            if self._is_system_role_error(e):
                try:
                    alt_messages = self._messages_without_system(all_messages)
                    log.warning("LLM stream system-role rejected; retrying with %d user-merged messages", len(alt_messages))
                    self.last_usage["prompt_messages"] = alt_messages
                    # Retry stream with converted messages
                    stream2 = self._client.chat.completions.create(
                        model=self._llm_model, messages=alt_messages, stream=True,
                        max_tokens=max_tokens,
                        temperature=float(os.getenv("TEMPERATURE", 0.72)),
                        top_p=float(os.getenv("TOP_P", 0.90)),
                        stop=LLM_STOP_SEQUENCES,
                        timeout=LLM_TIMEOUT,
                        extra_body=_llm_extra_body(),
                    )
                    full_response = []
                    for chunk in stream2:
                        delta = chunk.choices[0].delta if chunk.choices else None
                        token = (delta.content or "") if delta else ""
                        full_response.append(token)
                    text2 = "".join(full_response).strip()
                    if text2:
                        self.last_usage["completion_text"] = text2
                        stream_success = True
                        text = text2
                        # Mark success so we skip fallback
                        log.info("LLM stream retry without system role succeeded")
                except Exception as e2:
                    log.error(f"LLM stream retry without system also failed: {e2}")
        if stream_success:
            return text

        # Stream failed: no partial output exists.  The fallback is also held
        # for final review and conscience gating by the caller.
        fallback_text = self._fallback_completion(
            all_messages,
            max_tokens,
            "LLM stream failed or completed without content",
        )
        return fallback_text

    def _fallback_completion(self, messages: list[dict], max_tokens: int, reason: str) -> str:
        """Try one non-streaming completion before surfacing the LLM error in chat."""
        def _try_once(msgs: list[dict]) -> str | None:
            resp = self._client.chat.completions.create(
                model=self._llm_model,
                messages=_apply_no_think(msgs),
                stream=False,
                max_tokens=max_tokens,
                temperature=float(os.getenv("TEMPERATURE", 0.72)),
                top_p=float(os.getenv("TOP_P", 0.90)),
                stop=LLM_STOP_SEQUENCES,
                timeout=LLM_TIMEOUT,
                extra_body=_llm_extra_body(),
            )
            choice = resp.choices[0] if resp.choices else None
            msg = getattr(choice, "message", None)
            txt = (getattr(msg, "content", None) or "").strip()
            # Diagnostic surface for empty completions: stop reason +
            # token counts + thinking channel. llama.cpp reports these on
            # the non-streaming response; without them "empty" is a guess.
            finish = getattr(choice, "finish_reason", None)
            usage = getattr(resp, "usage", None)
            think_txt = ""
            if msg is not None:
                for attr in ("reasoning_content", "reasoning"):
                    try:
                        val = getattr(msg, attr, None)
                    except Exception:
                        val = None
                    if val:
                        think_txt = str(val)
                        break
            if not txt:
                log.warning(
                    "LLM non-streaming empty: model=%s finish=%s prompt_tok=%s "
                    "completion_tok=%s thinking_chars=%d n_msgs=%d",
                    self._llm_model, finish,
                    getattr(usage, "prompt_tokens", None),
                    getattr(usage, "completion_tokens", None),
                    len(think_txt), len(msgs),
                )
                if think_txt:
                    log.warning("LLM thinking-channel preview: %.300s", think_txt)
            else:
                self.last_usage.update({
                    "completion_text": txt,
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                })
            return txt

        try:
            text = _try_once(messages)
            if text:
                log.warning("%s; recovered with non-streaming completion", reason)
                return text
            # Empty but system role present — retry as user-merged (some templates return empty instead of 500).
            # Skipped when the stream already proved the cause is a burned
            # think budget (reasoning without content): re-sending would
            # burn another full budget thinking, not fix anything.
            if any(m.get("role") == "system" for m in messages) and not _take_stream_reasoned():
                try:
                    alt_empty = self._messages_without_system(messages)
                    log.warning("Non-streaming empty with system role; retrying without system (%d msgs)", len(alt_empty))
                    self.last_usage["prompt_messages"] = alt_empty
                    text_retry = _try_once(alt_empty)
                    if text_retry:
                        log.warning("%s; recovered with non-streaming (no-system) after empty", reason)
                        return text_retry
                except Exception as e_empty:
                    log.debug("No-system retry after empty failed: %s", e_empty)
            reason = f"{reason}; non-streaming completion was also empty"
        except Exception as e:
            # System-role template error — retry with merged user messages
            if self._is_system_role_error(e):
                try:
                    alt = self._messages_without_system(messages)
                    log.warning("Fallback system-role rejected; retrying non-streaming with %d user-merged messages", len(alt))
                    self.last_usage["prompt_messages"] = alt
                    text2 = _try_once(alt)
                    if text2:
                        log.warning("%s; recovered with non-streaming (no-system) completion", reason)
                        return text2
                    reason = f"{reason}; non-streaming (no-system) also empty"
                except Exception as e2:
                    reason = f"{reason}; non-streaming fallback failed: {e} | retry failed: {e2}"
            else:
                reason = f"{reason}; non-streaming fallback failed: {e}"

        log.error(reason)
        return f"[LLM error] {reason}"

    def _sanitize_history(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return []
        sanitized = [messages[0]]
        for msg in messages[1:]:
            if msg["role"] == sanitized[-1]["role"]:
                sanitized[-1] = msg
            else:
                sanitized.append(msg)
        while sanitized and sanitized[0]["role"] != "user":
            sanitized.pop(0)
        return sanitized

    def _finalize_response(self, user_input: str, draft: str, token_callback=None, *, already_emitted: bool = False) -> str:
        review = self._review_response(user_input, draft)
        response = self._correct_response(user_input, draft, review)
        if response != draft:
            self._review_response(user_input, response)
        try:
            from cognition.conscience.hooks import gate_speak
            memorize = self._get_memorize()
            mem_inner = getattr(memorize, "_mem", None) if memorize is not None else None
            embedder = getattr(mem_inner, "_embedder", None)
            gated_draft = response or draft
            replaced = gate_speak(
                draft=gated_draft,
                user_input=user_input,
                llm_client=getattr(self, "_client", None),
                embedder=embedder,
                already_emitted=False,
            )
            if replaced is not None:
                response = replaced
                if response != gated_draft:
                    log.info("[finalize] conscience replaced outbound draft")
        except Exception as exc:
            log.warning("[finalize] conscience speak gate failed; preserving draft under caution: %s", exc)
        try:
            from cognition.attention import for_identity
            speak = self._get_speak()
            if speak is not None and hasattr(speak, "set_expression"):
                snap = for_identity(current_user_id()).snapshot()
                affect = float(snap.get("affect", 0.0))
                volume = 1.05 if affect > 0.25 else 0.9 if affect < -0.25 else 1.0
                pitch = 0.05 if affect > 0.25 else -0.05 if affect < -0.25 else 0.0
                speak.set_expression(for_identity(current_user_id()).adaptive_tts_rate(), volume, pitch)
            elif speak is not None and hasattr(speak, "set_speech_rate"):
                speak.set_speech_rate(for_identity(current_user_id()).adaptive_tts_rate())
        except Exception:
            pass
        # Compatibility for any out-of-tree caller that already emitted an
        # accepted draft.  A refusal/escalation replacement must still be sent;
        # current in-tree streaming callers always arrive with no prior output.
        if already_emitted and response == draft:
            return response
        if already_emitted and token_callback and hasattr(token_callback, "reset"):
            token_callback.reset()
        self._emit_finalized_response(response, token_callback=token_callback)
        return response

    def _correct_response(self, user_input: str, draft: str, review: dict | None) -> str:
        """Repair only high-risk drafts, keeping the correction bounded."""
        if not review or len(review.get("flags", [])) < 2 or not draft.strip():
            return draft
        system = (
            f"{self._current_system_prompt()}\n\n"
            "You are Aiko's final response editor. Rewrite the draft only to fix the listed reliability issues. "
            "Preserve the user's intent, do not invent facts, and return only the corrected answer. "
            "If information is uncertain, say so plainly. Keep the original tone and length when possible."
        )
        prompt = (
            f"User request:\n{user_input[:1200]}\n\n"
            f"Draft:\n{draft[:3000]}\n\n"
            f"Warnings:\n- " + "\n- ".join(review.get("flags", [])[:4])
        )
        corrected = self._fallback_completion(
            [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            _effective_max_tokens(min(600, _BASE_TOKENS)),
            "metacognitive response correction",
        )
        return draft if not corrected.strip() or corrected.startswith("[LLM error]") else corrected.strip()

    def _review_response(self, user_input: str, response_text: str) -> dict:
        try:
            from cognition.attention import for_identity
            state = for_identity(current_user_id())
            review = state.review_response(user_input, response_text)
            state.persist()
            return review
        except Exception:
            return {}
    def _store_async(self, user_input: str, response_text: str) -> None:
        """Queue a fire-and-forget memory write. The actual queue/worker
        thread now lives on AikoMemorize (cognition.memory.memorize); this just wires
        up this instance's idle-tracking callables (is_active_turn /
        idle_since) so the write waits for a genuinely idle window before
        using the shared LLM for fact extraction. Kept as a method (rather
        than inlining self._get_memorize().queue_write(...) at every call site)
        because agentic.agentic's run_agentic_chat also calls
        owner._store_async(...) directly at the end of the agent loop.

        EMC-2: also stage the turn into episodic memory (best-effort, never
        blocks the turn and never invents metadata).
        """
        cognitive_state = None
        with _brain_trace.step("think._store_async", layer="write",
                               inputs={"user_input": user_input, "response_chars": len(response_text or "")}) as ctx:
            try:
                from cognition.attention import for_identity
                state = for_identity(current_user_id())
                confirmed = state.confirm_memory_update(user_input)
                if confirmed:
                    memorize = self._get_memorize()
                    for conflict in confirmed:
                        if conflict.get("memory_id") and memorize is not None:
                            memorize.supersede_exact(conflict["memory_id"], conflict.get("current", user_input), current_user_id())
            except Exception:
                # Conflict confirmation must never skip recording below.
                pass
            try:
                from cognition.attention import for_identity as _for_identity
                _state = _for_identity(current_user_id())
                _state.record(user_input, response_text)
                _sync_goal_review_schedule(_state)
                _state.persist()
                cognitive_state = _for_identity(current_user_id()).snapshot()
            except Exception:
                pass

            def _is_any_active():
                with self._active_users_lock:
                    return bool(self._active_user_ids)
            mem = self._get_memorize()
            mem.queue_write(
                user_input,
                response_text,
                user_id=current_user_id(),
                display_name=current_display_name(),
                is_active_turn=_is_any_active,
                idle_since=lambda: self._last_chat_time,
            )
            try:
                mem.queue_episode(
                    user_input,
                    response_text,
                    cognitive_state=cognitive_state,
                    user_id=current_user_id(),
                )
            except Exception:
                pass
            try:
                mem.wm_record_turn(user_input, response_text)
            except Exception:
                pass
            ctx.set(
                outputs={
                    "write_queued": True,
                    "episodic_queued": True,
                    "wm_recorded": True,
                    "memory_conflicts_consumed": cognitive_state is not None,
                },
                factors=[
                    "attention.record() updates affect/energy/uncertainty/goals/loops",
                    "queue_write → background thread, waits for idle window before LLM fact extraction",
                    "queue_episode → EpisodicStore staging",
                ],
            )


_STREAM_SENTENCE_END = set(".?!。？！")
_STREAM_CLOSERS = set("\"')]}」』”’")


def _is_stream_noise(char: str) -> bool:
    codepoint = ord(char)
    if char in {"\u200d", "\ufe0e", "\ufe0f", "\u20e3"}:
        return True
    if 0x1F000 <= codepoint <= 0x1FFFF:
        return True
    if 0x2600 <= codepoint <= 0x27BF:
        return True
    if 0x2300 <= codepoint <= 0x23FF:
        return True
    if 0x2B00 <= codepoint <= 0x2BFF:
        return True
    return unicodedata.category(char)[0] == "S"


def split_stream_sentences(buffer: str) -> tuple[list[str], str]:
    """
    Parse the streaming buffer, extract completed sentences, and return
    a list of completed sentences and the remaining partial sentence text.
    """
    sentences = []
    start = 0
    i = 0
    while i < len(buffer):
        char = buffer[i]
        if char in "\n\r":
            end = i + 1
        elif char in _STREAM_SENTENCE_END:
            end = i + 1
            while end < len(buffer) and buffer[end] in _STREAM_CLOSERS:
                end += 1
            if end == len(buffer):
                break
            if not (buffer[end].isspace() or _is_stream_noise(buffer[end])):
                i += 1
                continue
            while end < len(buffer) and (buffer[end].isspace() or _is_stream_noise(buffer[end])):
                end += 1
        else:
            i += 1
            continue

        sentence = buffer[start:end].strip()
        if sentence:
            sentences.append(sentence)
        start = end
        i = end

    remaining = buffer[start:]
    if not sentences and len(remaining) > 150:
        split_pts = [m.start() for m in re.finditer(r'[\s,、;:]', remaining)]
        if split_pts:
            split_pt = max([p for p in split_pts if p <= 150] or [split_pts[-1]])
            sentence = remaining[:split_pt + 1].strip()
            tail = remaining[split_pt + 1:]
            return ([sentence] if sentence else []), tail
    return sentences, remaining
