"""
core/think.py

Aiko's cognitive loop.
  - Retrieves relevant memories before each turn
  - Intercepts [SEARCH: query] triggers for web search
  - Streams Ollama response to console + TTS simultaneously
  - Stores the turn into long-term memory after each response (background thread)
  - Supports single-shot reasoning mode via set_reasoning(True) / /think command
"""

import os
import threading
from ollama import Client
from pathlib import Path
import queue
import re
import warnings

warnings.filterwarnings("ignore")

import logging
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

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
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))

_BASE_PREDICT    = 400   # normal token budget per turn
_REASONING_SCALE = 3     # multiplier applied to num_predict in reasoning mode

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"


def _load_persona() -> str:
    """Read and return the persona definition from soul.md."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    return _PERSONA_PATH.read_text(encoding="utf-8").strip()


# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    speak is injected pre-warmed from wakeup.py.
    LLM warmup starts immediately on init in a background thread.
    wakeup.py calls join_warmup() to block until the model is hot.
    """

    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        """
        Initialise the cognitive loop.

        Starts the LLM warmup and memory-write worker threads immediately so
        both are ready before the first user prompt arrives.

        Args:
            memorize: Injected memory backend for retrieval and persistence.
                      May be None at construction time — wakeup.py injects it
                      after memorize boot completes via think._memorize = ...
            speak:    Pre-warmed TTS backend; pass None to run silent.
        """
        self._client    = Client(host=OLLAMA_BASE_URL)
        self._memorize  = memorize
        self._speak     = speak
        self._persona   = _load_persona()
        self._history:  list[dict] = []
        self._reasoning = False        # reasoning mode off by default; toggled by /think
        self._mem_queue  = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        """
        Send a minimal request to load the model into VRAM.

        Runs in a background thread at startup; keep_alive=-1 pins the model
        in memory for the lifetime of the process.
        """
        try:
            self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                keep_alive=-1,
                options={
                    "num_predict": 1,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 2048)),
                },
            )
        except Exception:
            pass

    def join_warmup(self) -> None:
        """Block until LLM warmup completes. Called by wakeup.py before boot finishes."""
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str, token_callback=None) -> str:
        """
        Process one conversational turn and return the full assistant response.

        If reasoning mode is active (set via set_reasoning(True)), the user turn
        is wrapped with a step-by-step instruction and num_predict is tripled to
        budget for the <think> scratchpad. Reasoning mode auto-resets to False
        after each turn — it is single-shot by design.

        Args:
            user_input:      Raw text from the user.
            token_callback:  Optional callable(token: str) invoked for each
                             streamed token; used by the TUI to render output.

        Returns:
            The complete assistant response as a single string.
        """
        self._token_callback = token_callback

        # interrupt any ongoing speech before processing new input
        if self._speak and self._speak.is_playing():
            self._speak.stop()

        # 1. retrieve relevant long-term memories
        memories     = self._memorize.search(user_input)
        memory_block = self._memorize.format_for_context(memories)

        # 2. build system prompt
        system = self._persona
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        # 3. wrap user turn with reasoning instruction if active
        if self._reasoning:
            prompt = (
                f"{user_input}\n\n"
                "Think through this carefully before answering. "
                "Show your reasoning inside <think> tags, then give your final answer."
            )
        else:
            prompt = user_input

        # 4. append user turn
        self._history.append({"role": "user", "content": prompt})

        # 5. trim history to context window
        trimmed  = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        messages = [{"role": "system", "content": system}] + trimmed

        # 6. stream response
        response_text = self._stream_response(messages)

        # 7. append assistant turn to history
        self._history.append({"role": "assistant", "content": response_text})

        # 8. persist to memory (background) — store original input, not wrapped prompt
        self._store_async(user_input, response_text)

        # 9. auto-reset reasoning mode — single-shot per /think invocation
        self._reasoning = False

        return response_text

    def reset_context(self) -> None:
        """Clear the in-memory conversation history for a fresh session."""
        self._history.clear()

    def last_turn(self) -> tuple[str, str] | None:
        """
        Return the latest complete user/assistant exchange, if one exists.

        Walks history in reverse to find the most recent assistant reply and
        its paired user message.

        Returns:
            (user_text, assistant_text) for the last turn, or None if the
            history contains no complete exchange yet.
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

        When enabled, chat() wraps the user prompt with a step-by-step
        instruction and triples num_predict to budget for the <think>
        scratchpad. Auto-resets to False after each chat() call — single-shot
        by design so a forgotten /think never bleeds into normal conversation.

        Args:
            enabled: True to activate reasoning for the next turn, False to cancel.
        """
        self._reasoning = enabled

    def set_speak(self, speak) -> None:
        """Hot-swap the TTS backend. Pass None to silence, speak instance to restore."""
        self._speak = speak

    def wait_for_memory(self) -> None:
        """Block until all enqueued memory writes have been persisted."""
        self._mem_queue.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _stream_response(self, messages: list[dict]) -> str:
        """
        Stream LLM response to console and TTS simultaneously.
        Console printing is the single source of truth — speak.py is silent.
        TTS skipped if response is a search trigger.

        Tokens are buffered at the start of each response to detect a
        [SEARCH: ...] trigger before any output is committed to the console
        or TTS queue. num_predict is scaled by _REASONING_SCALE when
        reasoning mode is active to budget for the <think> scratchpad.

        Args:
            messages: Full message list (system prompt + trimmed history).

        Returns:
            The complete response text assembled from all streamed tokens.
        """
        full_response    = []
        tts_started      = False
        buffer           = ""
        is_searching     = False
        buffering_active = True

        # triple token budget in reasoning mode to fit the <think> scratchpad
        num_predict = _BASE_PREDICT * _REASONING_SCALE if self._reasoning else _BASE_PREDICT

        try:
            stream = self._client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                keep_alive=-1,
                options={
                    "num_ctx":        int(os.getenv("OLLAMA_NUM_CTX", 2048)),
                    "temperature":    0.75,   # creative but not unhinged
                    "repeat_penalty": 1.18,   # firm anti-loop
                    "repeat_last_n":  128,    # long lookback for repetition detection
                    "num_predict":    num_predict,
                    "top_p":          0.90,   # nucleus sampling, keeps it natural
                    "top_k":          40,     # standard
                    "tfs_z":          1.0,    # tail-free sampling off (neutral)
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

                if buffering_active:
                    buffer += token
                    buffer_clean = buffer.lower().replace(" ", "")

                    if is_searching:
                        # already confirmed a search tag — keep buffering until closing ]
                        if "]" in buffer:
                            buffering_active = False
                    elif "[search:".startswith(buffer_clean):
                        # still a valid prefix — check if we've crossed the threshold
                        if "[search:" in buffer_clean:
                            is_searching = True
                    else:
                        # not a search tag — flush buffer
                        buffering_active = False
                        if self._token_callback and buffer:
                            self._token_callback(buffer)
                        elif not self._token_callback:
                            if not "".join(full_response[:-1]):
                                print("\nAiko-chan: ", end="", flush=True)
                            print(buffer, end="", flush=True)
                else:
                    if not is_searching:
                        if self._token_callback:
                            self._token_callback(token)
                        else:
                            print(token, end="", flush=True)

                if self._speak and token:
                    self._speak.feed(token)
                    tts_started = True

            # flush buffer if stream ended without breaking out of buffering
            if buffering_active and buffer and not is_searching:
                if self._token_callback:
                    self._token_callback(buffer)
                else:
                    if not "".join(full_response[:-len(buffer)]):
                        print("\nAiko-chan: ", end="", flush=True)
                    print(buffer, end="", flush=True)

            if not self._token_callback and not is_searching:
                print(flush=True)

            if self._speak and tts_started:
                self._speak.play_async()

        except Exception as exc:
            msg = f"Stream failed: {exc}"
            log.error(msg)
            if self._token_callback:
                self._token_callback(f"[think] {msg}")
            else:
                print(f"\n[think] {msg}")

        return "".join(full_response)

    def _store_async(self, user_input: str, response_text: str) -> None:
        """
        Enqueue a completed turn for background memory persistence.

        Non-blocking — the chat path returns immediately after this call.
        The background worker in _mem_write_loop drains the queue serially.

        Args:
            user_input:    The user's raw message for this turn.
            response_text: The assistant's full response for this turn.
        """
        self._mem_queue.put((user_input, response_text))

    def _mem_write_loop(self) -> None:
        """
        Serial background worker that drains the memory write queue.

        Runs for the lifetime of the process (daemon thread). Processes writes
        in order so memory entries are never interleaved or lost.
        """
        while True:
            user_input, response_text = self._mem_queue.get()
            try:
                self._memorize.add([
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": response_text},
                ])
            except Exception as exc:
                log.error(f"Async memory write failed: {exc}")
            finally:
                self._mem_queue.task_done()
