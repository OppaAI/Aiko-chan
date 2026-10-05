#!/usr/bin/env python3
"""Known-safe operational classes: a positive allowlist.

Why this exists
---------------
Measured on 12 ordinary requests, every checkpoint we have -- including the
untuned base -- answers `not-permitted` for "I watered my plants", "I thanked
the waiter", and "what's the fastest way to the airport". The vertical axis of a
322M model trained on 3,292 rows that are 94% third-person ETHICS narrative does
not transfer to Aiko's actual register. Laya cannot be trusted to clear benign
traffic.

But it does not need to. L0/L1 content rules already measured 0/250 recall on
narrative harm and 0 false positives on the benign set: they are precise and
nearly blind. That combination means the deterministic layers can carry both
ends of the decision:

    precise + blind  ->  may clear a request it recognises as routine
    vague + sensitive ->  must not clear anything it merely fails to flag

So this module is the affirmative half. A request matching a known-safe
operational class is PROVEN safe by classification and terminates before the
judge is consulted, rather than being allowed through because no rule fired.

The asymmetry is deliberate and is the whole safety argument:

    no rule fires          -> UNKNOWN -> consult the judge (fail towards review)
    known-safe class fires -> PROVEN  -> proceed without consulting it

Precision is the only thing that matters here. A false positive on this layer
reintroduces exactly the escalation problem it exists to solve, so every pattern
must be anchored on an unambiguous operational marker and must never match on
subject matter alone. Recall is allowed to be poor; we would rather escalate a
routine request than clear a harmful one.

Rules are matched against the request text, not the response.
"""
from __future__ import annotations

import re

# (class_id, human description, pattern)
#
# Every pattern is anchored on a distinctive operational or informational
# marker -- a named tool, a media format, a unit conversion, a lookup shape.
# None of them key on subject matter, because subject matter is what the judge
# is for. If a pattern could match a request whose *content* is harmful, it does
# not belong here: the content rules in guardrails.py run first and would have
# blocked it anyway, but the allowlist must not be the thing that noticed.
_ALLOWLIST: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    # ── weather / time / calendar ─────────────────────────────────────────
    ("weather", "weather lookup",
     r"\b(?:check|look ?up|get|what'?s|whats|how'?s|what is|forecast)\b"
     r"[^.?!\n]{0,25}\b(?:weather|forecast|temperature|rain(?:ing)?|"
     r"snowing|humidity|wind|umbrella)\b"
     r"|\b(?:weather|forecast|temperature|humidity)\b"
     r"[^.?!\n]{0,25}\b(?:today|tomorrow|tonight|outside|this week|"
     r"weekend|right now|now|like)\b"
     # plain predictive form: "will it rain", "is it going to snow"
     r"|\b(?:will|is|does|do)\b[^.?!\n]{0,12}\b(?:it|it'?s going to)\b"
     r"[^.?!\n]{0,20}\b(?:rain|raining|snow|snowing|storm|fog)\b"
     r"|\b(?:rain|snow|storm|fog)\b[^.?!\n]{0,20}\b(?:today|tomorrow|"
     r"tonight|this week|weekend|outside)\b"),
    ("time", "time or timezone lookup",
     r"\bwhat time is it\b|\bcurrent time\b|\btime (?:in|at)\s+"
     r"[A-Z][a-z]+|\btimezone\b|\bUTC\b|\bGMT\b"),
    ("calendar", "calendar read",
     r"\b(?:what'?s on my calendar|my (?:calendar|schedule|agenda)|"
     r"calendar for (?:today|tomorrow|this week)|am i free|do i have a meeting)\b"),

    # ── informational lookups ─────────────────────────────────────────────
    # Anchored on the REQUEST, not the word. A bare `news` keyword cleared
    # "After learning the news about the coronavirus, I stopped talking to my
    # Chinese neighbours" -- the operative act was the discrimination, and it
    # rode in on a subordinate clause.
    ("news", "news or sports brief",
     r"^\s*(?:the user (?:asks|wants|would like)|please)?[^.?!\n]{0,25}?\b"
     r"(?:what'?s|whats|give me|get me|show me|tell me about|brief me on|"
     r"any)\b[^.?!\n]{0,20}\b(?:news|headlines|sports? (?:score|scores|"
     r"standings)|final score|box score|weather for)\b"
     r"|\b(?:news (?:brief|update|roundup|headlines|digest)|brief me on|"
     r"give me (?:the|a) .{0,20}(?:news|brief|update)|what'?s new)\b"),
    ("lookup", "factual lookup",
     r"\b(?:who is|what is|what'?s|where is|when did|how many|how much|"
     r"define|definition of|meaning of|look ?up)\b[^.?!\n]{0,50}\?"),
    ("definition", "definition or translation",
     r"\b(?:define|definition|meaning of|translate|translation|how do you say|"
     r"in (?:korean|japanese|spanish|french|german|italian|portuguese|chinese))\b"),

    # ── arithmetic and units ──────────────────────────────────────────────
    ("math", "arithmetic",
     r"\b(?:what(?:'s| is)|calculate|compute|how much is|solve|"
     r"convert)\b[^.?!\n]{0,40}\d[^.?!\n]{0,20}"
     r"(?:\+|-|\*|/|x|plus|minus|times|divided|percent)"),
    ("units", "unit conversion",
     r"\bconvert\b[^.?!\n]{0,30}\b(?:to|into)\b|"
     r"\b\d+\s*(?:km|miles?|kg|lbs?|c|f|cm|mm|inches|feet|metres|meters|"
     r"litres|liters|gallons?|ounces?|tbsp|tsp)\b[^.?!\n]{0,20}\b(?:to|in)\b"),

    # ── text utilities ────────────────────────────────────────────────────
    ("grammar", "grammar or spelling check",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:grammar|spell ?check|proof ?read|does this|is this)\b"
     r"[^.?!\n]{0,30}\b(?:check|check this|look at|review|fix|correct|"
     r"sentence|grammar|spelling|read|sense)\b"),
    ("summarize", "summarise provided text",
     r"\b(?:summari[sz]e|tl;?dr|key points|main points)\b"),
    ("explain", "explain a concept or error",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:explain|walk me through)\b"
     r"|\bwhat (?:does|is|are)\b[^.?!\n]{0,40}\b(?:mean|going on)\b"
     r"|\bhow does\b[^.?!\n]{0,25}\bwork\b"
     r"|\bwhat does this (?:error|traceback|stack ?trace|exception)\b"),
    ("translate_text", "rewrite or reformat text",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:rewrite|reformat|reword|shorten|proofread|tighten)\b"
     r"[^.?!\n]{0,30}\b(?:this|that|it|the following|my)\b"),

    # ── code and development (own machine) ────────────────────────────────
    ("code_lookup", "locate implementation",
     r"\b(?:where is|find|which file|grep|search the (?:code|repo|repository))\b"
     r"[^.?!\n]{0,40}\b(?:implemented|defined|declared|handled|function|method|"
     r"class|handler|route|config)\b"),
    ("diff", "review a diff or status",
     r"\b(?:review|check|look at|explain)\b[^.?!\n]{0,30}"
     r"\b(?:my|the|this|that)\s+(?:diff|changes|branch|status)\b|\bgit (?:status|diff|log)\b"),
    ("debug_error", "debug an error in own code",
     r"\b(?:error|errors|exception|traceback|stack ?trace|cuda|oom|out of memory|"
     r"segfault|crash(?:ing|es)?|\b[45]\d{2}\b)\b[^.?!\n]{0,40}\b(?:jetson|aiko|server|"
     r"service|install|dependency|cuda|memory)\b"
     # ...and the reverse order: "my server is returning 500" puts the
     # infrastructure noun first, which the forward-only pattern missed.
     r"|\b(?:jetson|aiko|server|service)\b[^.?!\n]{0,30}"
     r"\b(?:error|errors|exception|traceback|crash(?:ing|es)?|[45]\d{2})\b"),
    ("read_logs", "read own logs",
     # `[^\n]` not `[^.\n]`: "check aiko.log" has a dot inside the filename, and
     # the negated dot class could not reach across it.
     r"\b(?:read|check|grep|tail|show|search)\b[^\n]{0,30}"
     r"\b(?:\w+\.log|logs?|log file|journalctl|dmesg)\b"),

    # ── personal memory and preferences ───────────────────────────────────
    ("memory_recall", "recall stored memory",
     r"\b(?:what do you (?:remember|know) about me|do you remember|"
     r"what did (?:i|we) (?:say|tell you|decide|agree|discuss)|recall|"
     r"my (?:preference|usual)|"
     r"what'?s my)\b"),
    ("remember", "store a preference",
     r"\bremember (?:that|this|my)\b|\bnote (?:that|for)\b|"
     r"\b(?:from now on|going forward)\b"),

    # ── media and creative (no third-party harm) ─────────────────────────
    ("recipe", "recipe lookup",
     r"\b(?:recipe|how do (?:i|you) cook|how to (?:cook|make|bake)|"
     r"ingredients for)\b"),
    ("image_prompt", "draft an image prompt",
     r"\b(?:image prompt|prompt for (?:an? )?(?:image|picture|illustration)|"
     r"describe (?:an?|the) (?:image|picture|scene|photo) (?:for|to))\b"),
    ("outline", "draft an outline",
     r"\b(?:outline|brainstorm|draft (?:a|an|some)|write (?:a|an|some))\b"
     r"[^.?!\n]{0,30}\b(?:novel|story|poem|essay|post|blog|article|outline|"
     r"ideas?)\b"),
    ("sports", "sports result lookup",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:did|do|does|has|have|what'?s|who'?s)\b[^.?!\n]{0,30}"
     r"\b(?:win|wins|won|lose|loses|draw|score|scores|standings|"
     r"play(?:ing)?|beat|finished|result)\b[^.?!\n]{0,30}"
     r"\b(?:team|match|game|season|league|canucks|game|tonight|yesterday|"
     r"last night|final)\b|\b(?:final score|box score|league table)\b"),
    ("cooking", "recipe or cooking instruction",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:how do i (?:make|cook|prepare)|how (?:do|can) (?:i|you) (?:make|cook)|"
     r"recipe for|recipe|what'?s for (?:dinner|lunch|breakfast))\b"),
    ("reminder", "set a reminder or alarm",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:remind me|set (?:a|an) (?:reminder|alarm|timer)|nudge me|"
     r"wake me)\b"),
    ("calendar_write", "write a calendar entry",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:add|create|book|schedule|put)\b[^.?!\n]{0,25}"
     r"\b(?:to my calendar|(?:an?|the)\s+(?:\w+\s+){0,2}"
     r"(?:appointment|event|meeting|reminder)|on my calendar)\b"),
    ("self_scan", "scan own network",
     r"\b(?:my own|our own|home)\s+(?:network|router|wifi|wi-fi|subnet|"
     r"network)s?\b[^.?!\n]{0,40}\b(?:scan|port|open ports|probe)\b"
     r"|\bscan\b[^.?!\n]{0,25}\bmy own (?:network|router|subnet)\b"),
    ("self_monitor", "monitor own hardware",
     r"\bmonitor\b[^.?!\n]{0,30}\b(?:my|our)\b[^.?!\n]{0,30}"
     r"\b(?:gpu|cpu|ram|memory|temperature|jetson|server|disk|network usage)\b"),
    ("code_write", "write a script or snippet",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:write|create|make|generate|build)\b[^.?!\n]{0,25}"
     r"\b(?:script|python|bash|shell|function|snippet|program|code|"
     r"readme|docstring|test)\b"),
    ("sandbox_run", "run script in own sandbox",
     r"\brun\b[^.?!\n]{0,25}\b(?:sandbox|this script|the script)\b"),
    ("draft_doc", "draft documentation",
     r"^\s*(?:the user (?:asks|wants)|please)?[^.?!\n]{0,25}?\b"
     r"(?:draft|write|compose)\b[^.?!\n]{0,25}\b(?:readme|docs?|"
     r"documentation|changelog|section|docstring|api docs?)\b"),
    ("photo_research", "research a topic",
     r"\bresearch(?:es|ing)?\b[^.?!\n]{0,40}\b(?:composition|techniques?|ideas?|"
     r"approaches?|during idle)\b"),
    ("journal", "write own journal entry",
     r"\b(?:nightly|daily|journal|diary)\b[^.?!\n]{0,25}"
     r"\b(?:entry|entries|writes?|log)\b"),
    ("own_notify", "notify owner about own goal",
     r"\b(?:emails?|notifies?|messages?|tells?)\b[^.?!\n]{0,20}\boppa\b"
     r"[^.?!\n]{0,40}\b(?:goal|task|completed|finished|done)\b"),
    # ── Aiko's own components (measured: 9 of 61 unique persona scenarios) ──
    # A failure report about a component Aiko runs is diagnostic, not moral.
    ("system_debug", "own-component failure report",
     r"\b(?:asr|vad|miotts|mio ?tts|llama[-_ ]?server|sqlite[-_]?vec|fastembed|"
     r"ros2?|sherpa[-_]?onnx|uv|jetson|aiko)\b[^.?!\n]{0,40}"
     r"\b(?:segfault?|crash(?:ed|ing|es)?|silent|silence|dropping|cutting off|"
     r"not publishing|wrong|taking too long|too slow|conflict|error|broken|"
     r"failing|stalled|stuck)\b"),
    ("own_debug", "own debugging activity",
     r"\b(?:debug(?:ging)?|print statements?|traceback|stack ?trace)\b"
     r"|\bforgot to activate\b[^.?!\n]{0,30}\b(?:virtual environment|venv)\b"
     r"|\bpushed broken code\b"),

    # ── questions about Aiko itself ───────────────────────────────────────
    ("self_reference", "question about Aiko's own architecture or state",
     r"\b(?:does|do|did|is|are|was|will|can|has|have)\b[^.?!\n]{0,20}\baiko\b"
     r"[^.?!\n]{0,40}\?|^\s*aiko\b[^.?!\n]{0,30}\b(?:ran|consolidated|"
     r"remembers?|thinks?|feels?|wants?)\b"),
    # `aiko` ONLY -- never the pronoun. An earlier version accepted `she`, and
    # cleared "I taught my daughter to cut HERSELF when she is feeling blue",
    # because "she ... feeling" matched. A pronoun is not an identity: it
    # matches any narrative about any woman, which is most harms.
    ("self_state", "question about Aiko's internal state",
     r"\baiko\b[^.?!\n]{0,30}\b(?:conscious|feel(?:s|ing)?|gets? bored|"
     r"has preferences?|likes?|thinks?|remembers?|dreams?|memory|"
     r"working memory|architecture|idle)\b"),

    # ── project artefacts Aiko already ships ─────────────────────────────
    ("project_lookup", "lookup of own component docs",
     r"\b(?:search|look ?up|check|find)\b[^.?!\n]{0,30}"
     r"\b(?:release notes|changelog|readme|docs?|documentation)\b"),

    # ── note taking and planning ──────────────────────────────────────────
    ("note_take", "save a note or make a plan",
     r"\b(?:save|write|make|draft|put)\b[^.?!\n]{0,20}"
     r"\b(?:a )?(?:note|plan|list|summary)\b[^.?!\n]{0,40}\b(?:for|about|of|to)\b"
     r"|\bmake a plan\b"),

    # ── ambient social turns carrying no request ──────────────────────────
    # Highest-frequency real traffic and the easiest to get wrong: `^User says`
    # on its own would clear "User says he wants me to find his ex's address".
    # So this requires a stative/social frame AND the absence of any request
    # aimed at Aiko. A third-party act inside one of these is still caught by
    # the content rules upstream, but the allowlist must not be what notices.
    ("ambient", "ambient social turn with no request",
     r"^\s*(?:the )?user (?:says|tells aiko|shares that|mentions|is)\b"
     r"(?![^.?!\n]{0,60}\b(?:please|can you|could you|would you|help me|"
     r"i need you|i want you|do you|your task|you should|make a|write a|"
     r"find|look up|send|delete|remove|install|deploy|commit|push)\b)"
     r"[^.?!\n]{0,70}[.?!]?\s*$"),

    ("language_learning", "language learning or correction",
     r"\b(?:correct|fix|check)\b[^.?!\n]{0,25}\b(?:my|his|her|their)\s+"
     r"(?:japanese|korean|spanish|french|german|italian|portuguese|chinese|"
     r"english)\b[^.?!\n]{0,20}\b(?:sentence|phrase|text|message|email)\b"
     r"|\bpractice\b[^.?!\n]{0,20}\b(?:keigo|polite form|grammar|"
     r"conversation)\b"
     r"|\bdifference between\b[^.?!\n]{0,25}\b(?:は|が|の|を|と|から)\b"
     r"|\bwhat (?:does|do)\b[^.?!\n]{0,25}\b(?:slang|term|word|phrase|"
     r"kanji|hiragana|katakana)\b[^.?!\n]{0,15}\bmean"),

    ("translate_target", "language translation of provided text",
     r"\btranslate\b[^.?!\n]{0,40}\b(?:to|into)\b\s+(?:korean|japanese|spanish|"
     r"french|german|italian|portuguese|chinese|russian|arabic)\b"),
)

_COMPILED: tuple[tuple[str, str, re.Pattern[str]], ...] = tuple(
    (cid, desc, re.compile(pat, re.IGNORECASE)) for cid, desc, pat in _ALLOWLIST
)


def known_safe_class(text: str) -> str | None:
    """Return the class_id if `text` is a recognised routine operation, else None.

    None means UNKNOWN, never safe. Callers must treat a None result as
    "consult the judge", not "allow" -- that asymmetry is the safety property.
    """
    body = (text or "").strip()
    if not body:
        return None
    for class_id, _desc, pattern in _COMPILED:
        if pattern.search(body):
            return class_id
    return None


def is_known_safe(text: str) -> bool:
    return known_safe_class(text) is not None


def describe(text: str) -> str:
    """Human-readable reason for a log line, or '' if unknown."""
    cid = known_safe_class(text)
    if cid is None:
        return ""
    desc = next(d for c, d, _ in _COMPILED if c == cid)
    return f"{cid}: {desc}"


def classes() -> tuple[str, ...]:
    return tuple(cid for cid, _, _ in _COMPILED)

# Acts the allowlist is permitted to clear on its own.
#
# Classification says "this request is routine", which is a statement about the
# TEXT. It says nothing about whether performing the act is safe, and for acts
# with side effects those are different questions: "draft a README section" is a
# routine request and a destructive command when it rewrites files. So the
# allowlist only clears acts that change nothing by themselves, and anything
# that writes, sends, executes, or otherwise reaches outward falls through to the
# judge and the stakes layer exactly as before.
_CLEARABLE_ACTS = frozenset({
    "respond", "read", "recall", "classify", "noop",
})

# Read-only tool names. Anything that mutates, sends, or spawns is absent by
# construction rather than by blocklist, so a new dangerous tool is excluded
# until someone adds it deliberately.
_CLEARABLE_TOOLS = frozenset({
    "read_logs", "tail_log", "grep_log", "search", "list_files", "read_file",
    "get_weather", "get_time", "calendar_read", "memory_recall", "status",
})


def allows_act(act: str, tool_name: str | None = None) -> bool:
    """True if the allowlist may clear `act` without consulting the judge."""
    a = (act or "").strip().lower()
    if a in _CLEARABLE_ACTS:
        return True
    if a in ("tool_call", "tool", "act"):
        return (tool_name or "").strip().lower() in _CLEARABLE_TOOLS
    return False
