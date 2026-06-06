"""
core/think.py

Aiko's cognitive loop.
  - Drains memory write queue before each recall (prevents write-lag misses)
  - Retrieves relevant memories before each turn
  - Intercepts [SEARCH: query] triggers for web search (post-stream via regex)
  - Streams Ollama response to console + TTS simultaneously
  - Strips [SEARCH:...] tag from display/TTS output
  - Stores the turn into long-term memory after each response (background thread)
  - Supports single-shot reasoning mode via set_reasoning(True) / /think command
"""

import logging
import os
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from datetime import datetime
from ollama import Client
from pathlib import Path
import queue
import re
import threading

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
from core.log      import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'think_start':  'Loading Ollama client + persona...',
    'think_warmup': 'Warming up language model...',
}

# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",    "ministral-3:3b-instruct-2512-q4_K_M")
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 8))

_BASE_PREDICT    = 280   # normal token budget — soul.md says 2–3 sentences default
_REASONING_SCALE = 3     # multiplier applied to num_predict in reasoning mode

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"

# Regex to detect and strip [SEARCH: ...] tags from response text
_SEARCH_RE = re.compile(r'\[SEARCH:\s*(.+?)\]', re.IGNORECASE)


def _load_persona() -> str:
    """Read and return the persona definition from soul.md."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    persona = _PERSONA_PATH.read_text(encoding="utf-8").strip()
    user_id = os.getenv("USER_ID", "OppaAI")
    today   = datetime.now().strftime("%B %d, %Y")
    return persona.replace("USER_ID_HERE", user_id).replace("TODAY_HERE", today)


# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    speak is injected pre-warmed from wakeup.py.
    LLM warmup starts immediately on init in a background thread.
    wakeup.py calls join_warmup() to block until the model is hot.
    """

    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client    = Client(host=OLLAMA_BASE_URL)
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._reasoning = False
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        try:
            self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                keep_alive=-1,
                options={
                    "num_predict": 1,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 4096)),
                },
            )
        except Exception as e:
            log.warning("LLM warmup failed — Ollama may not be running: %s", e)

    def join_warmup(self) -> None:
        """Block until LLM warmup completes. Called by wakeup.py before boot finishes."""
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str, token_callback=None) -> str:
        """
        Process one conversational turn and return the full assistant response.

        Flow:
          1. Drain pending memory writes so recall sees the latest facts.
          2. Search long-term memory; inject explicit no-memory signal if empty.
          3. Build system prompt (persona + memory block).
          4. Stream LLM response silently (full text collected).
          5. Strip any [SEARCH:...] tag from display text before sending to
             token_callback and TTS — tag is never shown to the user raw.
          6. If a search trigger was found, fetch results, re-stream grounded
             response, clean display again.
          7. Append to history and enqueue async memory write.

        Reasoning mode (set via set_reasoning(True) / /think command) wraps the
        user prompt with a step-by-step instruction and triples num_predict.
        Auto-resets to False after each turn — single-shot by design.
        """
        self._token_callback = token_callback

        # interrupt any ongoing speech before processing new input
        if self._speak and self._speak.is_playing():
            self._speak.stop()

        # 1. flush pending memory writes so this turn's recall is up to date
        self.wait_for_memory()

        # 2. retrieve relevant long-term memories
        memories     = self._memorize.search(user_input, limit=int(os.getenv("MEMORY_RECALL_LIMIT", 3)))
        memory_block = self._memorize.format_for_context(memories)

        # 3. build system prompt — inject explicit no-memory signal on miss
        system = self._persona
        if memory_block:
            system = f"{system}\n\n{memory_block}"
        else:
            system = (
                f"{system}\n\n"
                "<memory_context>\n"
                "No relevant memories found for this query. "
                "If Oppa asks about something personal or specific to him, "
                "tell him you don't have it stored rather than guessing.\n"
                "</memory_context>"
            )

        # 4. wrap user turn with reasoning instruction if active
        if self._reasoning:
            prompt = (
                f"{user_input}\n\n"
                "Think through this carefully before answering. "
                "Show your reasoning inside <think> tags, then give your final answer."
            )
        else:
            prompt = user_input

        # 5. append user turn
        self._history.append({"role": "user", "content": prompt})

        # 6. trim history to context window
        trimmed  = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        trimmed  = self._sanitize_history(trimmed)

        # 7. stream response (collected silently — display handled below)
        raw_response = self._stream_response(trimmed, system=system, silent=True)

        # 8. detect search trigger in full response
        search_match = _SEARCH_RE.search(raw_response)
        if search_match:
            query = search_match.group(1).strip()

            # Intent gate: confirm this is actually a factual/data request
            if not self._is_data_intent(user_input):
                # Social/conversational — strip the tag, respond normally
                display_text = _SEARCH_RE.sub("", raw_response).strip()
                self._emit(display_text)
                self._history.append({"role": "assistant", "content": raw_response})
                self._store_async(user_input, display_text)
                self._reasoning = False
                return display_text

            # notify UI that search is happening
            if self._token_callback:
                self._token_callback(f"__SEARCHING__:{query}")

            from core.tools import web_search_context
            context = web_search_context(query)

            if context:
                # inject grounded search results — LLM must not add training data
                grounded_context = (
                    f"<search_results query='{query}'>\n"
                    f"Answer using ONLY the information in these results. "
                    f"Do not add anything from your training data. "
                    f"If the results don't contain the answer, say so.\n\n"
                    f"{context}\n"
                    f"</search_results>"
                )
                self._history.append({"role": "assistant", "content": raw_response})
                self._history.append({"role": "user",      "content": grounded_context})
                trimmed      = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
                trimmed      = self._sanitize_history(trimmed)
                raw_response = self._stream_response(trimmed, system=system, silent=True)
                # clean up injected search turns from persistent history
                self._history = self._history[:-2]
            else:
                raw_response = "[no results found]"

        # 9. strip [SEARCH:...] tag from display text, then emit to callback + TTS
        display_text = _SEARCH_RE.sub("", raw_response).strip()
        self._emit(display_text)

        # 10. append assistant turn (raw, including any search tag, for history fidelity)
        self._history.append({"role": "assistant", "content": raw_response})

        # 11. persist to memory (background) — store original input, not wrapped prompt
        self._store_async(user_input, display_text)

        # 12. auto-reset reasoning mode
        self._reasoning = False

        return display_text

    def web_search(self, query: str, token_callback=None) -> str:
        """
        Run a web search and feed results into a normal chat turn.
        Called by the /web command — no search logic leaks out.
        """
        from core.tools import web_search_context
        context = web_search_context(query)
        if context is None:
            msg = f"[no results for: {query}]"
            if token_callback:
                token_callback(msg)
            return msg
        return self.chat(context, token_callback=token_callback)

    def reset_context(self) -> None:
        """Clear the in-memory conversation history for a fresh session."""
        self._history.clear()

    def last_turn(self) -> tuple[str, str] | None:
        """
        Return the latest complete user/assistant exchange, if one exists.
        Walks history in reverse to find the most recent assistant reply
        and its paired user message.
        Returns (user_text, assistant_text) or None.
        """
        assistant_text: str | None = None

        for message in reversed(self._history):
            role    = message.get("role")
            content = (message.get("content") or "").strip()
            if not content:
                continue
            if assistant_text is None:
                if role == "assistant":
                    assistant_text = content
                continue
            if role == "user":
                return content, assistant_text

        return None

    def set_reasoning(self, enabled: bool) -> None:
        """
        Enable or disable reasoning mode for the next turn only.
        Auto-resets to False after each chat() call.
        """
        self._reasoning = enabled

    def set_speak(self, speak) -> None:
        """Hot-swap the TTS backend. Pass None to silence, speak instance to restore."""
        self._speak = speak

    def wait_for_memory(self) -> None:
        """Block until all enqueued memory writes have been persisted."""
        self._mem_queue.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _emit(self, text: str) -> None:
        """
        Send display text to token_callback (TUI) or stdout, and feed TTS.
        Called once with the full cleaned response after search handling.
        """
        if not text:
            return

        if self._token_callback:
            self._token_callback(text)
        else:
            print(f"\nAiko-chan: {text}", flush=True)

        if self._speak:
            self._speak.feed(text)
            self._speak.play_async()

    def _stream_response(
        self,
        messages: list[dict],
        system:   str  = "",
        silent:   bool = False,
    ) -> str:
        """
        Stream LLM response and return the full assembled text.

        When silent=True, no output is sent to token_callback or stdout during
        streaming — the caller handles display after search detection and tag
        stripping. This prevents raw [SEARCH:...] tags from reaching the user.

        When silent=False (legacy path, not currently used by chat()), tokens
        stream live to callback/stdout as before.

        num_predict is scaled by _REASONING_SCALE when reasoning mode is active.
        """
        full_response = []

        num_predict = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT

        all_messages = (
            [{"role": "system", "content": system}] + messages
            if system else messages
        )

        try:
            stream = self._client.chat(
                model=OLLAMA_MODEL,
                messages=all_messages,
                stream=True,
                keep_alive=-1,
                options={
                    "num_ctx":        int(os.getenv("OLLAMA_NUM_CTX", 4096)),
                    "temperature":    float(os.getenv("OLLAMA_TEMPERATURE", 0.72)),
                    "repeat_penalty": float(os.getenv("OLLAMA_REPEAT_PENALTY", 1.15)),
                    "repeat_last_n":  int(os.getenv("OLLAMA_REPEAT_LAST_N", 64)),
                    "num_predict":    num_predict,
                    "top_p":          float(os.getenv("OLLAMA_TOP_P", 0.90)),
                    "top_k":          int(os.getenv("OLLAMA_TOP_K", 40)),
                    "tfs_z":          1.0,
                    "stop":           ["<|im_end|>", "</s>", "[INST]"],
                }
            )

            for chunk in stream:
                token = (
                    chunk.message.content
                    if hasattr(chunk, "message")
                    else chunk.get("message", {}).get("content", "")
                ) or ""

                full_response.append(token)

                # live streaming only when silent=False
                if not silent:
                    if self._token_callback:
                        self._token_callback(token)
                    else:
                        if len(full_response) == 1:
                            print("\nAiko-chan: ", end="", flush=True)
                        print(token, end="", flush=True)

            if not silent and not self._token_callback:
                print(flush=True)

        except Exception as exc:
            msg = f"Stream failed: {exc}"
            log.error(msg)
            if self._token_callback:
                self._token_callback(f"[think] {msg}")
            else:
                print(f"\n[think] {msg}")
            return ""

        return "".join(full_response)

    def _sanitize_history(self, messages: list[dict]) -> list[dict]:
        """
        Enforce strict user/assistant alternation.
        Merges consecutive same-role messages (keeps last).
        Strips leading assistant turns — history must start with user.
        """
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

    def _store_async(self, user_input: str, response_text: str) -> None:
        """
        Enqueue a completed turn for background memory persistence.
        Non-blocking — chat() returns immediately after this call.
        The background worker in _mem_write_loop drains the queue serially.
        """
        self._mem_queue.put((user_input, response_text))

    def _mem_write_loop(self) -> None:
        """
        Serial background worker that drains the memory write queue.
        Runs for the lifetime of the process (daemon thread).
        """
        while True:
            user_input, response_text = self._mem_queue.get()
            try:
                self._memorize.add([
                    {"role": "user",      "content": user_input[:500]},
                    {"role": "assistant", "content": response_text[:800]},
                ])
            except Exception as exc:
                log.error(f"Async memory write failed: {exc}")
            finally:
                self._mem_queue.task_done()