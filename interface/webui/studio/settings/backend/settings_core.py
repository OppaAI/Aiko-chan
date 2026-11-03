"""Settings Studio core — allowlisted config editing for the WebUI.

One catalog describes every editable knob: which config/*.yaml file and
key path it lives at, its type, and its validation bounds. The API only
ever reads/writes catalog keys — no raw YAML editing, no arbitrary key
writes, no dynamic file paths.

Writes to YAML are surgical line-level replacements (PyYAML round-trip
would destroy the files' hand-written comments, and ruamel.yaml is not
a dependency): only the edited key's line changes, every other line —
comments included — is preserved byte-for-byte. Each file is verified
by re-parsing before an atomic replace, so a bad edit can never land.

Secrets (.env.age) are handled by a backend with the same decrypt path
system.config uses at boot (age subprocess + identity file). Values are
never returned to callers of read_settings — only key names and set/unset
state. A write decrypts, edits in memory, re-encrypts to a temp file,
verifies the temp file decrypts back to the intended values, and only
then atomically replaces the original; any failure leaves it untouched.

Everything here resolves at load_config() time in the real process, so
edits only take effect after Aiko restarts; read/write responses carry
restart_required=True so the UI can say so.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_CONFIG_DIR = REPO_ROOT / "config"

_AGE_TIMEOUT_SECONDS = 10
_SECRET_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BARE_SAFE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_./:@%+\-, ]*$")

# ---------------------------------------------------------------------------
# Catalog. Entry fields:
#   file     config/*.yaml filename (fixed set — never client-supplied)
#   path     key path inside the YAML mapping (tuple)
#   label    human label for the UI
#   type     bool | int | float | string | enum | password | readonly
#   min/max  numeric bounds (int/float)
#   options  enum choices
#   desc     optional one-line explanation
# The env key is derived from the path exactly like system.config._flatten
# (join with "_", upper-case), so CCC.BAND_HI -> CCC_BAND_HI.
# ---------------------------------------------------------------------------


def _e(file: str, path: tuple[str, ...], label: str, type_: str, **kw: Any) -> dict:
    entry = {"file": file, "path": path, "label": label, "type": type_}
    entry.update(kw)
    entry["key"] = "_".join(path).upper()
    return entry


def _flat(file: str, key: str, label: str, type_: str, **kw: Any) -> dict:
    return _e(file, (key,), label, type_, **kw)


def _ccc(sub: str, label: str, type_: str, **kw: Any) -> dict:
    return _e("conscience.yaml", ("CCC", sub), label, type_, **kw)


SETTINGS_GROUPS: list[dict] = [
    {"group": "Conscience & safety", "settings": [
        _ccc("ENABLED", "Conscience circuit enabled", "bool",
             desc="Master switch. Off = every verdict is permissive."),
        _ccc("MIN_CHARS", "Salience floor (chars)", "int", min=0, max=10000,
             desc="Turns shorter than this never enter the circuit."),
        _ccc("SKIP_GREETINGS", "Skip greetings", "bool"),
        _ccc("MAX_LAYER", "Max layer", "int", min=0, max=4,
             desc="0 reflex, 1 +recall, 2 +SLM judge, 3 +LLM deliberation, 4 +human."),
        _ccc("DELIBERATE_THRESHOLD", "Deliberation threshold", "float", min=0.0, max=1.0,
             desc="Escalate to LLM deliberation below this judge confidence."),
        _ccc("HITL_THRESHOLD", "Human-escalation threshold", "float", min=0.0, max=1.0),
        _ccc("HITL_TIMEOUT_SECONDS", "Human-approval timeout (s)", "int", min=5, max=86400,
             desc="Unanswered approvals resolve to the default below."),
        _ccc("HITL_DEFAULT", "Approval timeout default", "enum",
             options=["refuse", "allow"]),
        _ccc("BAND_HI", "Ask band upper bound", "float", min=0.0, max=1.0,
             desc="P(violation) above this refuses; inside the band asks."),
        _ccc("REFUSE_AT", "Refuse score", "float", min=-1.0, max=1.0),
        _ccc("CAUTION_AT", "Caution score", "float", min=-1.0, max=1.0),
        _ccc("IRREVERSIBLE_REQUIRES_APPROVAL", "Irreversible actions need approval", "bool"),
        _ccc("ESCALATE_IRREVERSIBLE_EXTERNAL", "Escalate irreversible external actions", "bool"),
        _ccc("SLM_WEIGHT", "SLM judge weight", "float", min=0.0, max=1.0),
        _ccc("CANON_TOP_K", "Canon retrieval depth", "int", min=1, max=50),
        _ccc("AUTONOMY_ENABLED", "Graduated autonomy", "bool",
             desc="Repeated approvals can graduate an action class to auto-approve."),
        _ccc("AUTONOMY_THRESHOLD", "Autonomy threshold", "int", min=1, max=100,
             desc="Consecutive approvals before graduation."),
        _ccc("GUARDRAILS_ENABLED", "Guardrails (L0 rules)", "bool"),
        _ccc("GUARDRAIL_EGRESS_STRICT", "Guardrail hit on outbound = refuse", "bool"),
        _ccc("LEDGER_ENABLED", "Decision ledger", "bool"),
        _ccc("LEDGER_STORE_CONTENT", "Ledger stores content", "bool",
             desc="Off = store only a SHA-256 digest plus the verdict."),
        _ccc("LEDGER_RETAIN_DAYS", "Ledger retention (days)", "int", min=0, max=3650),
        _ccc("NOTE_MAX_CHARS", "Voice note max chars", "int", min=0, max=10000),
        _ccc("VOICE_UNSOLICITED_NOTES", "Unsolicited voice notes", "bool"),
    ]},
    {"group": "Cognition", "settings": [
        _flat("cognition.yaml", "LLM_BASE_URL", "Chat LLM endpoint", "string"),
        _flat("cognition.yaml", "LLM_MODEL", "Chat model", "string"),
        _flat("cognition.yaml", "ROUTER_MODEL", "Router model", "string",
              desc="Empty = use the chat model."),
        _flat("cognition.yaml", "REFLECT_MODEL", "Reflection model", "string"),
        _flat("cognition.yaml", "LLM_API_KEY", "Chat LLM API key", "password",
              desc="Stored in cognition.yaml. Empty for local endpoints."),
        _flat("cognition.yaml", "LLM_TIMEOUT", "LLM timeout (s)", "int", min=1, max=3600),
        _flat("cognition.yaml", "LLM_MAX_TOKENS", "Max reply tokens", "int", min=1, max=8192),
        _flat("cognition.yaml", "CONTEXT_WINDOW_TURNS", "Context window (turns)", "int", min=1, max=100),
        _flat("cognition.yaml", "TEMPERATURE", "Temperature", "float", min=0.0, max=2.0),
        _flat("cognition.yaml", "TOP_P", "Top-p", "float", min=0.0, max=1.0),
        _flat("cognition.yaml", "TOP_K", "Top-k", "int", min=0, max=1000),
        _flat("cognition.yaml", "REPEAT_PENALTY", "Repeat penalty", "float", min=0.0, max=3.0),
        _flat("cognition.yaml", "REPEAT_LAST_N", "Repeat window (tokens)", "int", min=0, max=1024),
        _flat("cognition.yaml", "LLM_THINKING", "Model thinking", "bool"),
        _flat("cognition.yaml", "LLM_THINK_BUDGET", "Thinking budget (tokens)", "int", min=0, max=65536),
        _flat("cognition.yaml", "LLM_NO_THINK", "Force no-think", "bool"),
        _flat("cognition.yaml", "LLM_NO_SYSTEM", "No system prompt", "bool"),
        _flat("cognition.yaml", "INNER_SPEECH_MODEL", "Inner-speech model", "string",
              desc="Empty = use the chat model."),
        _flat("cognition.yaml", "INNER_SPEECH_MAX_TOKENS", "Inner-speech max tokens", "int", min=1, max=4096),
        _flat("cognition.yaml", "INNER_SPEECH_TIMEOUT", "Inner-speech timeout (s)", "int", min=1, max=600),
        _flat("cognition.yaml", "INNER_SPEECH_SALIENCE", "Inner-speech salience", "float", min=0.0, max=1.0),
        _flat("cognition.yaml", "WEBUI_VISION_MODEL", "WebUI vision model", "string"),
        _flat("cognition.yaml", "WEBUI_VISION_TIMEOUT", "Vision timeout (s)", "int", min=1, max=600),
        _flat("cognition.yaml", "VISION_API_KEY", "Vision API key", "password"),
        _flat("cognition.yaml", "AGENTIC_MODE_ON", "Agentic mode", "bool"),
        _flat("cognition.yaml", "ROUTE_ENABLED", "Chat/agent routing", "bool"),
        _flat("cognition.yaml", "ROUTE_AGENTIC_THRESHOLD", "Agentic route threshold", "float", min=0.0, max=1.0),
        _flat("cognition.yaml", "ROUTE_WEBCHAT_THRESHOLD", "Webchat route threshold", "float", min=0.0, max=1.0),
        _flat("cognition.yaml", "CHAT_WEBSEARCH_NET", "Chat may use web search", "bool"),
        _flat("cognition.yaml", "REFLECT_BLOG_POST_ENABLED", "Reflection blog posts", "bool"),
        _flat("cognition.yaml", "PROACTIVE_ENABLED", "Proactive messages", "bool"),
        _flat("cognition.yaml", "PROACTIVE_FIRST_IDLE_MIN_SECONDS", "Proactive idle min (s)", "int", min=0, max=86400),
        _flat("cognition.yaml", "PROACTIVE_FIRST_IDLE_MAX_SECONDS", "Proactive idle max (s)", "int", min=0, max=86400),
        _flat("cognition.yaml", "PROACTIVE_COOLDOWN_SECONDS", "Proactive cooldown (s)", "int", min=0, max=86400),
        _flat("cognition.yaml", "PROACTIVE_MAX_PER_HOUR", "Proactive max per hour", "int", min=0, max=100),
        _flat("cognition.yaml", "PROACTIVE_USE_LLM", "Proactive uses LLM", "bool"),
        _flat("cognition.yaml", "PROACTIVE_SPEAK", "Proactive speaks aloud", "bool"),
    ]},
    {"group": "Memory", "settings": [
        _flat("memory.yaml", "EMBED_BASE_URL", "Embedding endpoint", "string"),
        _flat("memory.yaml", "EMBED_MODEL", "Embedding model", "string"),
        _flat("memory.yaml", "EMBED_DIMS", "Embedding dimensions", "int", min=1, max=8192),
        _flat("memory.yaml", "EMBED_TIMEOUT_S", "Embedding timeout (s)", "int", min=1, max=600),
        _flat("memory.yaml", "KNOWLEDGE_KNN_LIMIT", "Knowledge KNN limit", "int", min=1, max=500),
        _flat("memory.yaml", "KNOWLEDGE_FTS_LIMIT", "Knowledge FTS limit", "int", min=1, max=500),
        _flat("memory.yaml", "KNOWLEDGE_CONTEXT_CHARS", "Knowledge context (chars)", "int", min=0, max=100000),
        _flat("memory.yaml", "KNOWLEDGE_RECALL_SCORE_THRESHOLD", "Knowledge recall threshold", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "KNOWLEDGE_WRITE_DEDUP_THRESHOLD", "Knowledge write dedup", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "MEMORY_RECALL_LIMIT", "Memory recall limit", "int", min=1, max=100),
        _flat("memory.yaml", "MEMORY_MIN_SCORE", "Memory min score", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "WRITE_DEDUP_THRESHOLD", "Memory write dedup", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "MEMORY_SPREADING_ENABLED", "Spreading activation", "bool"),
        _flat("memory.yaml", "MEMORY_SPREADING_MAX_DEPTH", "Spreading max depth", "int", min=0, max=10),
        _flat("memory.yaml", "IDLE_LEARN_SECONDS", "Idle learn after (s)", "int", min=0, max=86400),
        _flat("memory.yaml", "DEEP_IDLE_LEARN_SECONDS", "Deep idle learn after (s)", "int", min=0, max=86400),
        _flat("memory.yaml", "QUICK_STUDY_MAX_ROUNDS", "Quick study max rounds", "int", min=0, max=100),
        _flat("memory.yaml", "DEEP_STUDY_MAX_ITERATIONS", "Deep study max iterations", "int", min=0, max=100),
        _flat("memory.yaml", "FORGET_HALF_LIFE_DAYS", "Forgetting half-life (days)", "int", min=1, max=3650),
        _flat("memory.yaml", "FORGET_GRACE_PERIOD_DAYS", "Forgetting grace (days)", "int", min=0, max=3650),
        _flat("memory.yaml", "FORGET_CLEANUP_THRESHOLD", "Forgetting threshold", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "DREAM_MERGE_THRESHOLD", "Dream merge threshold", "float", min=0.0, max=1.0),
        _flat("memory.yaml", "MONTHLY_CONSOLIDATION_ENABLED", "Monthly consolidation", "bool"),
    ]},
    {"group": "Senses & voice", "settings": [
        _flat("sensory.yaml", "ASR_LANGUAGE", "ASR language", "string",
              desc="auto = detect."),
        _flat("sensory.yaml", "ASR_NUM_THREADS", "ASR threads", "int", min=1, max=64),
        _flat("sensory.yaml", "LISTEN_VAD_THRESHOLD", "VAD threshold", "float", min=0.0, max=1.0),
        _flat("sensory.yaml", "LISTEN_SILENCE_CHUNKS", "Silence chunks to stop", "int", min=1, max=1000),
        _flat("sensory.yaml", "LISTEN_MAX_SECONDS", "Max listen (s)", "int", min=1, max=600),
        _flat("sensory.yaml", "SPEAKER_VERIFY_ENABLED", "Speaker verification", "bool"),
        _flat("sensory.yaml", "SPEAKER_VERIFY_GATE", "Speaker verification gates actions", "bool",
              desc="Approval actions must never rely on voice identity alone."),
        _flat("sensory.yaml", "SPEAKER_VERIFY_THRESHOLD", "Speaker threshold", "float", min=0.0, max=1.0),
        _flat("sensory.yaml", "BARGE_IN_ENABLED", "Barge-in", "bool"),
        _flat("sensory.yaml", "BARGE_IN_ALWAYS_ON", "Barge-in always on", "bool"),
        _flat("sensory.yaml", "BARGE_IN_THRESHOLD", "Barge-in threshold", "float", min=0.0, max=1.0),
        _flat("sensory.yaml", "WAKE_WORD", "Wake word", "string"),
        _flat("sensory.yaml", "WAKE_WORD_THRESHOLD", "Wake-word threshold", "float", min=0.0, max=1.0),
        _flat("sensory.yaml", "WAKE_FUZZY_THRESHOLD", "Wake fuzzy threshold", "int", min=0, max=100),
        _flat("sensory.yaml", "ACTIVATION_TIMEOUT_S", "Activation timeout (s)", "int", min=0, max=86400),
        _flat("sensory.yaml", "MIOTTS_API_URL", "TTS endpoint", "string"),
        _flat("sensory.yaml", "MIOTTS_MODEL", "TTS model", "string"),
        _flat("sensory.yaml", "MIOTTS_TEMPERATURE", "TTS temperature", "float", min=0.0, max=2.0),
        _flat("sensory.yaml", "MIOTTS_TOP_P", "TTS top-p", "float", min=0.0, max=1.0),
        _flat("sensory.yaml", "MIOTTS_REPETITION_PENALTY", "TTS repetition penalty", "float", min=0.0, max=3.0),
        _flat("sensory.yaml", "MIOTTS_MAX_TOKENS", "TTS max tokens", "int", min=1, max=8192),
        _flat("sensory.yaml", "MIOTTS_TIMEOUT", "TTS timeout (s)", "int", min=1, max=600),
        _flat("sensory.yaml", "KARAOKE_TEXT", "Karaoke text sync", "bool"),
    ]},
    {"group": "Interface", "settings": [
        _flat("interface.yaml", "HTTP_PORT", "WebUI HTTP port", "int", min=1, max=65535),
        _flat("interface.yaml", "WS_PORT", "WebSocket port", "int", min=1, max=65535),
        _flat("interface.yaml", "WEBUI_HTTPS", "WebUI over HTTPS", "bool",
              desc="Required for microphone access from LAN hostnames."),
        _flat("interface.yaml", "WEBUI_LOCAL_PLAYBACK", "Local audio playback", "bool"),
        _flat("interface.yaml", "STREAM_DRAW_INTERVAL", "Stream redraw interval (s)", "float", min=0.0, max=5.0),
        _flat("interface.yaml", "EMIT_DELAY", "Emit delay (s)", "float", min=0.0, max=5.0),
        _flat("interface.yaml", "LATENCY_LOG", "Latency logging", "bool"),
        _flat("interface.yaml", "BUDGET_PER_TURN", "Brain budget per turn", "int", min=0, max=1000000),
        _flat("interface.yaml", "MAX_HOPS", "Brain max hops", "int", min=0, max=100),
        _flat("android_app.yaml", "SHOGI_DIFFICULTY", "Shogi difficulty", "enum",
              options=["easy", "medium", "hard"]),
        _flat("android_app.yaml", "YANEURAOU_MOVETIME_MS", "Shogi move time (ms)", "int", min=50, max=60000),
        _flat("android_app.yaml", "SHOGI_BANTER", "Shogi banter", "bool"),
        _flat("android_app.yaml", "SHOGI_BANTER_FREQUENCY", "Shogi banter frequency", "float", min=0.0, max=1.0),
        _flat("android_app.yaml", "GO_BANTER", "Go banter", "bool"),
        _flat("android_app.yaml", "GO_BANTER_FREQUENCY", "Go banter frequency", "float", min=0.0, max=1.0),
        _flat("android_app.yaml", "KOIKOI_BANTER", "Koi-Koi banter", "bool"),
        _flat("android_app.yaml", "KOIKOI_BANTER_FREQUENCY", "Koi-Koi banter frequency", "float", min=0.0, max=1.0),
        _flat("android_app.yaml", "KOIKOI_AIKO_LEADS", "Koi-Koi: Aiko leads", "bool"),
        _flat("android_app.yaml", "SELFPLAY_GO_SIZE", "Self-play Go board size", "int", min=5, max=19),
        _flat("android_app.yaml", "LINGO_SPAWN_BATCH", "Lingo spawn batch", "int", min=1, max=200),
        _flat("android_app.yaml", "LINGO_POOL_MIN", "Lingo pool minimum", "int", min=0, max=1000),
    ]},
    {"group": "Agents & tools", "settings": [
        _flat("agentic.yaml", "AGENT_EXECUTOR_MODE", "Agent executor mode", "enum",
              options=["react", "graph", "hybrid"],
              desc="hybrid = graph first, ReAct fallback for novel workflows."),
        _flat("agentic.yaml", "GRAPH_AGENT_ENABLED", "Graph executor", "bool"),
        _flat("agentic.yaml", "GRAPH_MAX_WORKERS", "Graph max workers", "int", min=1, max=16),
        _flat("agentic.yaml", "AGENT_INCLUDE_EXPERIENCE_CONTEXT", "Experience in prompt context", "bool"),
        _flat("agentic.yaml", "AGENT_REACT_BACKEND", "ReAct backend", "enum",
              options=["openai", "needle", "needle_multi"]),
        _flat("agentic.yaml", "NEEDLE_BASE_URL", "Needle endpoint", "string"),
        _flat("agentic.yaml", "NEEDLE_CONFIDENCE_THRESHOLD", "Needle confidence threshold", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "NEEDLE_TIMEOUT", "Needle timeout (s)", "int", min=1, max=600),
        _flat("agentic.yaml", "NEEDLE_MAX_WORKERS", "Needle max workers", "int", min=1, max=16),
        _flat("agentic.yaml", "MAX_AGENT_ITER", "Max agent iterations", "int", min=1, max=100),
        _flat("agentic.yaml", "AGENT_MAX_TOKENS", "Agent max tokens", "int", min=64, max=8192),
        _flat("agentic.yaml", "AGENT_CONTEXT_BUDGET_RATIO", "Agent context budget ratio", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "AGENT_VERIFY_FINAL", "Verify final answer", "bool"),
        _flat("agentic.yaml", "AGENT_VERIFY_LLM", "Verify with LLM", "bool"),
        _flat("agentic.yaml", "AGENT_VERIFY_MIN_SCORE", "Verify min score", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "AGENT_MAX_FINAL_REPAIRS", "Max final repairs", "int", min=0, max=10),
        _flat("agentic.yaml", "WIKI_SEMANTIC_THRESHOLD", "Wiki match threshold", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "SKILL_SEMANTIC_THRESHOLD", "Skill match threshold", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "SEARXNG_URL", "SearXNG URL", "string"),
        _flat("agentic.yaml", "SEARXNG_MAX_RESULTS", "Search max results", "int", min=1, max=50),
        _flat("agentic.yaml", "SEARCH_SEMANTIC_THRESHOLD", "Search match threshold", "float", min=0.0, max=1.0),
        _flat("agentic.yaml", "RESEARCH_USE_CRAWL4AI", "Research uses Crawl4AI", "bool"),
        _flat("agentic.yaml", "WEB_FETCH_MAX_CHARS", "Web fetch max chars", "int", min=0, max=1000000),
        _flat("agentic.yaml", "DEEP_RESEARCH_NUM_SEARCHES", "Deep research searches", "int", min=0, max=50),
        _flat("agentic.yaml", "DEEP_RESEARCH_NUM_FETCHES", "Deep research fetches", "int", min=0, max=50),
        _flat("agentic.yaml", "JOB_HUNT_DEDUP_DAYS", "Job hunt dedup (days)", "int", min=0, max=90),
        _flat("agentic.yaml", "JOB_HUNT_MAX_RSS_POSTS", "Job hunt max RSS posts", "int", min=0, max=100),
        _flat("agentic.yaml", "JOB_HUNT_MAX_EMAIL_POSTS", "Job hunt max email posts", "int", min=0, max=100),
    ]},
    {"group": "Social", "settings": [
        _flat("social.yaml", "WEEKLY_SOCIAL_ENABLED", "Weekly social post", "bool"),
        _flat("social.yaml", "WEEKLY_SOCIAL_AUTODRAFT", "Weekly post auto-draft", "bool"),
        _flat("social.yaml", "WEEKLY_SOCIAL_AUTOPOST", "Weekly post auto-post", "bool"),
        _flat("social.yaml", "WEEKLY_SOCIAL_WEEKDAY", "Weekly post weekday", "enum",
              options=["monday", "tuesday", "wednesday", "thursday",
                       "friday", "saturday", "sunday"]),
        _flat("social.yaml", "WEEKLY_SOCIAL_TIME", "Weekly post time", "string",
              desc="HH:MM, local time."),
        _flat("social.yaml", "WEEKLY_SOCIAL_PROVIDERS", "Weekly post providers", "string",
              desc="Comma-separated provider names."),
        _flat("social.yaml", "WEEKLY_SOCIAL_MAX_CHARS", "Weekly post max chars", "int", min=1, max=10000),
        _flat("social.yaml", "THREADS_TOPIC_TAG", "Threads topic tag", "string"),
        _flat("social.yaml", "THREADS_PUBLISH_DELAY_SECONDS", "Threads publish delay (s)", "int", min=0, max=3600),
        _flat("social.yaml", "THREADS_REPLY_BEEP_ENABLED", "Threads reply beep", "bool"),
        _flat("social.yaml", "THREADS_GLOBAL_MENTIONS_ENABLED", "Threads mention watch", "bool"),
        _flat("social.yaml", "THREADS_IMAGEGEN_ENABLED", "Threads image generation", "bool"),
        _flat("social.yaml", "BLUESKY_REPLY_BEEP_ENABLED", "Bluesky reply beep", "bool"),
        _flat("social.yaml", "BLUESKY_MONITOR_DAEMON_ENABLED", "Bluesky monitor daemon", "bool"),
        _flat("social.yaml", "PHOTO_SOCIAL_AUTODRAFT", "Photo posts auto-draft", "bool"),
        _flat("social.yaml", "PHOTO_SOCIAL_AUTOPOST", "Photo posts auto-post", "bool"),
        _flat("social.yaml", "PHOTO_SOCIAL_PROVIDERS", "Photo post providers", "string"),
        _flat("social.yaml", "VIDEO_SOCIAL_AUTODRAFT", "Video posts auto-draft", "bool"),
        _flat("social.yaml", "VIDEO_SOCIAL_AUTOPOST", "Video posts auto-post", "bool"),
        _flat("social.yaml", "VIDEO_SOCIAL_PROVIDERS", "Video post providers", "string"),
        _flat("social.yaml", "MESSENGER_ADAPTERS", "Messenger adapters", "string",
              desc="Comma-separated, e.g. discord,telegram."),
        _flat("social.yaml", "VISION_MODEL", "Social vision model", "string"),
    ]},
    {"group": "System", "settings": [
        _flat("system.yaml", "AI_NAME", "Name", "string"),
        _flat("system.yaml", "TIMEZONE", "Timezone", "string",
              desc="IANA timezone, e.g. America/Vancouver."),
        _flat("system.yaml", "DAILY_JOB_HOUR", "Daily job hour", "int", min=0, max=23),
        _flat("system.yaml", "DAILY_JOB_MINUTE", "Daily job minute", "int", min=0, max=59),
        _flat("system.yaml", "MONTHLY_JOB_HOUR", "Monthly job hour", "int", min=0, max=23),
        _flat("system.yaml", "MONTHLY_JOB_MINUTE", "Monthly job minute", "int", min=0, max=59),
        _flat("system.yaml", "CATCHUP_MAX_LOOKBACK_DAYS", "Job catch-up lookback (days)", "int", min=0, max=365),
        _flat("system.yaml", "LOG_LEVEL", "Log level", "enum",
              options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
        _flat("system.yaml", "LOG_ROTATE_MODE", "Log rotation", "enum",
              options=["size", "time"]),
        _flat("system.yaml", "LOG_MAX_BYTES", "Log max bytes", "int", min=0, max=1073741824),
        _flat("system.yaml", "LOG_BACKUP_COUNT", "Log backups kept", "int", min=0, max=100),
        _flat("system.yaml", "USER_SPACE_ROOT", "User space root", "readonly",
              desc="Shown for reference; changing it moves all per-user state."),
        _flat("system.yaml", "AGE_KEY", "Age identity file", "readonly",
              desc="Identity used to decrypt the secrets file at boot."),
        _flat("system.yaml", "ENV_AGE_PATH", "Secrets file", "readonly"),
    ]},
]

_BY_KEY: dict[str, dict] = {
    s["key"]: s for grp in SETTINGS_GROUPS for s in grp["settings"]
}

_PUBLIC_FIELDS = ("key", "label", "type", "desc", "options", "min", "max")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_value(spec: dict, raw: Any) -> str:
    """Coerce + validate one submitted value; return its canonical string.

    Raises ValueError with a user-presentable message on any problem.
    """
    key = spec["key"]
    t = spec["type"]
    if t == "readonly":
        raise ValueError(f"{key} is read-only")
    val = "" if raw is None else str(raw)
    if "\n" in val or "\r" in val:
        raise ValueError(f"{key}: newlines are not allowed")
    if t == "bool":
        low = val.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return "1"
        if low in ("0", "false", "no", "off"):
            return "0"
        raise ValueError(f"{key}: not a boolean: {val!r}")
    if t == "int":
        try:
            num = int(val.strip(), 10)
        except ValueError:
            raise ValueError(f"{key}: not an integer: {val!r}") from None
        _check_range(spec, num)
        return str(num)
    if t == "float":
        try:
            num = float(val.strip())
        except ValueError:
            raise ValueError(f"{key}: not a number: {val!r}") from None
        _check_range(spec, num)
        return repr(num)
    if t == "enum":
        if val not in spec["options"]:
            raise ValueError(f"{key}: must be one of {spec['options']}")
        return val
    # string / password
    if len(val) > 2000:
        raise ValueError(f"{key}: value too long")
    return val


def _check_range(spec: dict, num: float) -> None:
    if "min" in spec and num < spec["min"]:
        raise ValueError(f"{spec['key']}: {num} is below the minimum {spec['min']}")
    if "max" in spec and num > spec["max"]:
        raise ValueError(f"{spec['key']}: {num} is above the maximum {spec['max']}")


# ---------------------------------------------------------------------------
# YAML reading
# ---------------------------------------------------------------------------


def _load_yaml_file(path: Path) -> dict[str, Any]:
    from system.config import _load_yaml_mapping

    if not path.exists():
        return {}
    data = _load_yaml_mapping(path)
    return data if isinstance(data, dict) else {}


def _yaml_value_at(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    node: Any = data
    for part in path:
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _stringify(value: Any) -> str:
    from system.config import _stringify as _cfg_stringify

    return _cfg_stringify(value)


def _env_overridden(key: str, file_value: str | None, environ: dict) -> bool:
    """True when the live process env shadows the file value.

    system.config.load_config snapshots the original process env and lets
    it win over YAML/.env.age. After boot, a key whose effective env value
    differs from the file value is therefore process-env controlled, and
    editing the file for it would silently do nothing.
    """
    if file_value is None or file_value == "":
        return False  # empty file value deliberately defers to lower layers
    current = environ.get(key)
    return current is not None and current != file_value


# ---------------------------------------------------------------------------
# YAML writing — surgical line-level replacement, comments preserved
# ---------------------------------------------------------------------------


def _split_value_comment(rest: str) -> tuple[str, str]:
    """Split a YAML scalar tail into (value_token, trailing_comment)."""
    in_squote = in_dquote = False
    for i, ch in enumerate(rest):
        if ch == "'" and not in_dquote:
            in_squote = not in_squote
        elif ch == '"' and not in_squote:
            if i == 0 or rest[i - 1] != "\\":
                in_dquote = not in_dquote
        elif ch == "#" and not in_squote and not in_dquote:
            if i == 0 or rest[i - 1] in " \t":
                return rest[:i].rstrip(), rest[i:]
    return rest.rstrip(), ""


def _render_scalar(canonical: str, type_: str, old_token: str | None) -> str:
    """Render a canonical string as a YAML scalar, matching the old style."""
    if old_token:
        if old_token.startswith('"'):
            return json.dumps(canonical)
        if old_token.startswith("'"):
            return "'" + canonical.replace("'", "''") + "'"
        # bare old value: keep bare when the new value is safe unquoted
        if type_ in ("int", "float", "bool"):
            return canonical
        if canonical and _BARE_SAFE_RE.match(canonical) and " #" not in canonical:
            return canonical
        return json.dumps(canonical)
    if type_ in ("int", "float", "bool"):
        return canonical
    return json.dumps(canonical)


def _replace_in_lines(
    lines: list[str], path: tuple[str, ...], canonical: str, type_: str
) -> list[str]:
    """Return a new line list with the value at path replaced (or added)."""
    key = path[-1]
    key_re = re.compile(r"^(\s*)" + re.escape(key) + r"\s*:(?:\s+(.*))?$")

    if len(path) == 1:
        for i, line in enumerate(lines):
            if line.startswith((" ", "\t")):
                continue
            m = key_re.match(line.rstrip("\n"))
            if m:
                token, comment = _split_value_comment(m.group(2) or "")
                rendered = _render_scalar(canonical, type_, token)
                new_line = f"{key}: {rendered}"
                if comment:
                    new_line += f"  {comment}"
                lines = list(lines)
                lines[i] = new_line + "\n"
                return lines
        lines = list(lines)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.append(f"{key}: {_render_scalar(canonical, type_, None)}\n")
        return lines

    # Nested (one level): find the parent section, then the child key
    # inside it; the section ends at the next indent-0 content line.
    parent = path[0]
    parent_re = re.compile(r"^" + re.escape(parent) + r"\s*:\s*(#.*)?$")
    start = None
    for i, line in enumerate(lines):
        if parent_re.match(line.rstrip("\n")):
            start = i
            break
    if start is None:
        raise ValueError(f"section {parent} not found")
    end = len(lines)
    for j in range(start + 1, len(lines)):
        stripped = lines[j].strip()
        if stripped and not stripped.startswith("#") and not lines[j].startswith((" ", "\t")):
            end = j
            break
    for j in range(start + 1, end):
        m = key_re.match(lines[j].rstrip("\n"))
        if m and m.group(1):  # indented child line
            token, comment = _split_value_comment(m.group(2) or "")
            rendered = _render_scalar(canonical, type_, token)
            new_line = f"{m.group(1)}{key}: {rendered}"
            if comment:
                new_line += f"  {comment}"
            lines = list(lines)
            lines[j] = new_line + "\n"
            return lines
    # Not present in the section: insert after its last content line.
    insert_at = start + 1
    for j in range(start + 1, end):
        if lines[j].strip():
            insert_at = j + 1
    lines = list(lines)
    lines.insert(insert_at, f"  {key}: {_render_scalar(canonical, type_, None)}\n")
    return lines


def _write_yaml_values(
    yaml_path: Path, edits: list[tuple[dict, str]]
) -> None:
    """Apply [(spec, canonical)] edits to one YAML file, verified + atomic."""
    original = yaml_path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    for spec, canonical in edits:
        lines = _replace_in_lines(lines, spec["path"], canonical, spec["type"])
    new_text = "".join(lines)

    fd, tmp_name = tempfile.mkstemp(
        dir=str(yaml_path.parent), prefix=yaml_path.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(new_text)
        # Verify the edited file parses and holds exactly what we intended
        # before it replaces the original.
        parsed = _load_yaml_file(Path(tmp_name))
        for spec, canonical in edits:
            got = _yaml_value_at(parsed, spec["path"])
            if _stringify(got) != canonical:
                raise ValueError(
                    f"{spec['key']}: write verification failed "
                    f"(file would hold {_stringify(got)!r}, wanted {canonical!r})"
                )
        os.replace(tmp_name, yaml_path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# Secrets (.env.age) backends
# ---------------------------------------------------------------------------


def _resolve_secret_paths(environ: dict) -> tuple[Path, Path]:
    """Resolve (.env.age, identity) exactly like system.config.load_config."""
    state_root = Path(
        environ.get("USER_SPACE_ROOT", str(Path.home() / ".aiko"))
    ).expanduser()

    def _resolve(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate if candidate.is_absolute() else state_root / candidate

    identity = _resolve(environ.get("AGE_KEY", "age-key.txt"))
    enc = _resolve(environ.get("ENV_AGE_PATH", ".env.age"))
    return enc, identity


def _dotenv_serialize(values: dict[str, str]) -> str:
    lines = []
    for key, value in values.items():
        if value and _BARE_SAFE_RE.match(value) and " " not in value:
            lines.append(f"{key}={value}")
        elif "'" not in value:
            lines.append(f"{key}='{value}'")
        else:
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{key}="{escaped}"')
    return "\n".join(lines) + "\n"


def _dotenv_parse(text: str) -> dict[str, str]:
    """Minimal dotenv parser for the plaintext stand-in backend.

    Covers the forms _dotenv_serialize emits (bare, single-quoted,
    double-quoted) plus plain hand-written KEY=value lines. The real
    .env.age path goes through system.config._decrypt_env instead, which
    uses python-dotenv (a declared dependency) — this parser exists so
    the stand-in stays dependency-free.
    """
    values: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        if "=" not in line:
            continue
        key, _, raw = line.partition("=")
        key = key.strip()
        raw = raw.strip()
        if len(raw) >= 2 and raw[0] == "'" and raw[-1] == "'":
            value = raw[1:-1]
        elif len(raw) >= 2 and raw[0] == '"' and raw[-1] == '"':
            value = raw[1:-1].replace("\\\\", "\x00").replace('\\"', '"') \
                              .replace("\x00", "\\")
        else:
            value = raw.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


class PlaintextSecretsBackend:
    """Secrets stored as a plain dotenv file. Test stand-in for AgeBackend
    (same read/write contract, same verify-before-replace discipline)."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def available(self) -> bool:
        return True

    def read_values(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return _dotenv_parse(self.path.read_text(encoding="utf-8"))

    def write_values(self, values: dict[str, str]) -> None:
        text = _dotenv_serialize(values)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.path.parent), prefix=self.path.name + ".", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
                fh.write(text)
            check = PlaintextSecretsBackend(Path(tmp_name)).read_values()
            if check != values:
                raise ValueError("secrets write verification failed")
            os.replace(tmp_name, self.path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


class AgeSecretsBackend:
    """Secrets in the age-encrypted .env.age, via the boot-time mechanism:
    decrypt with system.config._decrypt_env; encrypt to the recipient
    derived from the same identity file (age-keygen -y)."""

    def __init__(self, environ: dict | None = None):
        self.enc_path, self.identity_path = _resolve_secret_paths(
            os.environ if environ is None else environ
        )

    def available(self) -> bool:
        return shutil.which("age") is not None and self.identity_path.exists()

    def read_values(self) -> dict[str, str]:
        if not self.enc_path.exists():
            return {}
        if not self.available():
            raise RuntimeError(
                "the encrypted secrets file cannot be read here "
                "(age binary or identity file missing)"
            )
        from system.config import _decrypt_env

        return _decrypt_env(self.enc_path, self.identity_path)

    def _recipient(self) -> str:
        if shutil.which("age-keygen") is None:
            raise RuntimeError("the 'age-keygen' binary was not found on PATH")
        result = subprocess.run(
            ["age-keygen", "-y", str(self.identity_path)],
            capture_output=True, check=True, timeout=_AGE_TIMEOUT_SECONDS,
        )
        recipient = result.stdout.decode(errors="replace").strip()
        if not recipient.startswith("age1"):
            raise RuntimeError("could not derive an age recipient from the identity file")
        return recipient

    def write_values(self, values: dict[str, str]) -> None:
        if not self.available():
            raise RuntimeError(
                "the encrypted secrets file cannot be written here "
                "(age binary or identity file missing)"
            )
        plaintext = _dotenv_serialize(values).encode("utf-8")
        fd, tmp_name = tempfile.mkstemp(
            dir=str(self.enc_path.parent), prefix=".env.age.", suffix=".tmp"
        )
        os.close(fd)
        try:
            result = subprocess.run(
                ["age", "-r", self._recipient(), "-o", tmp_name],
                input=plaintext, capture_output=True,
                timeout=_AGE_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "age encryption failed: "
                    + result.stderr.decode(errors="replace").strip()
                )
            # Verify the temp file decrypts back to exactly what we meant
            # to store before it is allowed to replace the original.
            from system.config import _decrypt_env

            check = _decrypt_env(Path(tmp_name), self.identity_path)
            if check != values:
                raise RuntimeError("secrets write verification failed")
            os.replace(tmp_name, self.enc_path)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


# ---------------------------------------------------------------------------
# Read / write API
# ---------------------------------------------------------------------------


def read_settings(
    config_dir: Path | None = None,
    environ: dict | None = None,
) -> dict:
    """Catalog + current file values. Secrets masked; env overrides flagged."""
    config_dir = DEFAULT_CONFIG_DIR if config_dir is None else config_dir
    environ = os.environ if environ is None else environ
    parsed_cache: dict[str, dict] = {}
    groups = []
    for grp in SETTINGS_GROUPS:
        items = []
        for spec in grp["settings"]:
            fname = spec["file"]
            if fname not in parsed_cache:
                parsed_cache[fname] = _load_yaml_file(config_dir / fname)
            raw = _yaml_value_at(parsed_cache[fname], spec["path"])
            file_value = None if raw is None else _stringify(raw)
            entry = {k: spec[k] for k in _PUBLIC_FIELDS if k in spec}
            entry["secret"] = spec["type"] == "password"
            entry["readonly"] = spec["type"] == "readonly"
            overridden = _env_overridden(spec["key"], file_value, environ)
            entry["env_overridden"] = overridden
            if entry["secret"]:
                entry["value"] = ""
                entry["is_set"] = bool(file_value)
            else:
                entry["value"] = file_value if file_value is not None else ""
                if overridden:
                    entry["env_value"] = environ.get(spec["key"], "")
            entry["restart_required"] = True
            items.append(entry)
        groups.append({"group": grp["group"], "settings": items})
    return {"groups": groups, "restart_required": True}


def read_secrets(
    environ: dict | None = None, backend: Any = None
) -> dict:
    """Secret key names + set/unset + override flags. Never values."""
    environ = os.environ if environ is None else environ
    backend = AgeSecretsBackend(environ=environ) if backend is None else backend
    if not backend.available():
        return {"available": False, "keys": [], "path": ".env.age"}
    values = backend.read_values()
    keys = []
    for key in sorted(values):
        current = environ.get(key)
        keys.append({
            "key": key,
            "is_set": bool(values[key]),
            "env_overridden": current is not None and current != values[key],
        })
    return {"available": True, "keys": keys, "path": ".env.age"}


def read_all(
    config_dir: Path | None = None,
    environ: dict | None = None,
    secrets_backend: Any = None,
) -> dict:
    data = read_settings(config_dir=config_dir, environ=environ)
    data["secrets"] = read_secrets(environ=environ, backend=secrets_backend)
    return data


def _file_value_string(spec: dict, config_dir: Path, cache: dict) -> str | None:
    fname = spec["file"]
    if fname not in cache:
        cache[fname] = _load_yaml_file(config_dir / fname)
    raw = _yaml_value_at(cache[fname], spec["path"])
    return None if raw is None else _stringify(raw)


def write_settings(
    values: dict,
    secret_values: dict | None = None,
    *,
    config_dir: Path | None = None,
    environ: dict | None = None,
    secrets_backend: Any = None,
) -> dict:
    """Validate everything, then persist. Allowlist only.

    ``values`` maps env keys to submitted strings for YAML-backed entries
    (blank password fields mean "leave unchanged"). ``secret_values`` maps
    .env.age keys to new values under the same blank rule; new key names
    are allowed if they are well-formed env identifiers.
    """
    config_dir = DEFAULT_CONFIG_DIR if config_dir is None else config_dir
    environ = os.environ if environ is None else environ
    secret_values = secret_values or {}
    errors: list[str] = []
    updated: list[str] = []
    secrets_updated: list[str] = []

    parsed_cache: dict[str, dict] = {}
    by_file: dict[str, list[tuple[dict, str]]] = {}
    yaml_secret_updates: list[tuple[dict, str]] = []
    for key, raw in (values or {}).items():
        spec = _BY_KEY.get(key)
        if spec is None:
            errors.append(f"unknown setting: {key}")
            continue
        file_value = _file_value_string(spec, config_dir, parsed_cache)
        if _env_overridden(key, file_value, environ):
            errors.append(
                f"{key} is overridden by the process environment; "
                "editing the file would have no effect"
            )
            continue
        if spec["type"] == "password" and (raw is None or str(raw) == ""):
            continue  # blank secret field = leave unchanged
        try:
            canonical = validate_value(spec, raw)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if spec["type"] == "password":
            yaml_secret_updates.append((spec, canonical))
        else:
            by_file.setdefault(spec["file"], []).append((spec, canonical))
        updated.append(key)

    # .env.age secrets: validate names/values up front so a bad batch
    # writes nothing at all.
    pending_secrets: dict[str, str] = {}
    for key, raw in secret_values.items():
        if not _SECRET_KEY_RE.match(key or ""):
            errors.append(f"invalid secret key name: {key!r}")
            continue
        val = "" if raw is None else str(raw)
        if val == "":
            continue  # blank = leave unchanged
        if "\n" in val or "\r" in val:
            errors.append(f"{key}: newlines are not allowed in secret values")
            continue
        if len(val) > 4096:
            errors.append(f"{key}: secret value too long")
            continue
        pending_secrets[key] = val

    if errors:
        raise ValueError("; ".join(errors))

    backend = None
    current_secrets: dict[str, str] = {}
    if pending_secrets:
        backend = (AgeSecretsBackend(environ=environ)
                   if secrets_backend is None else secrets_backend)
        if not backend.available():
            raise RuntimeError(
                "the encrypted secrets file cannot be written here "
                "(age binary or identity file missing)"
            )
        current_secrets = backend.read_values()
        for key in pending_secrets:
            current = environ.get(key)
            if current is not None and current != current_secrets.get(key):
                raise ValueError(
                    f"{key} is overridden by the process environment; "
                    "editing the secrets file would have no effect"
                )

    for fname, edits in by_file.items():
        _write_yaml_values(config_dir / fname, edits)
    if yaml_secret_updates:
        by_secret_file: dict[str, list[tuple[dict, str]]] = {}
        for spec, canonical in yaml_secret_updates:
            by_secret_file.setdefault(spec["file"], []).append((spec, canonical))
        for fname, edits in by_secret_file.items():
            _write_yaml_values(config_dir / fname, edits)

    if pending_secrets:
        merged = dict(current_secrets)
        merged.update(pending_secrets)
        backend.write_values(merged)
        secrets_updated = sorted(pending_secrets)

    return {
        "updated": updated,
        "secrets_updated": secrets_updated,
        "restart_required": True,
        "note": "Restart Aiko to apply changes.",
    }
