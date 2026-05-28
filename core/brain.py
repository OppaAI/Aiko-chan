"""
core/brain.py

Aiko's cognitive loop.
  - Retrieves relevant memories before each turn
  - Intercepts [SEARCH: query] triggers for web search
  - Streams Ollama response
  - Stores the turn into long-term memory after each response (background thread)
"""

import os
import re
import threading
from pathlib import Path
from ollama import Client
from core.memory import AikoMemory
from core.tools import web_search


# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")

# soul.md lives at project root — resolve relative to this file
_PERSONA_PATH = Path(__file__).resolve().parent.parent / "soul.md"

# how many past turns to keep in the active context window
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))

# regex to detect search trigger in LLM output
_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)


def _load_persona() -> str:
    """Load Aiko's system prompt from soul.md at project root."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    return _PERSONA_PATH.read_text(encoding="utf-8").strip()


def _inject_search_instruction(system: str) -> str:
    """Append search tool instructions to the system prompt."""
    return system + """

## Web Search
You have access to the internet via a search tool.
When you need current information, facts, news, or anything you're unsure about,
output ONLY this on its own line before your response:
[SEARCH: your search query here]

You will then receive results and can use them to answer.
Only search when genuinely needed — don't search for things you already know.
"""


# ── brain ─────────────────────────────────────────────────────────────────────

class AikoBrain:
    """
    Aiko's conversational core.
    Manages the short-term context window and long-term mem0 memory.
    Memory writes run in a background thread — never blocks the chat loop.
    """

    def __init__(self, memory: AikoMemory) -> None:
        self._client     = Client(host=OLLAMA_BASE_URL)
        self._memory     = memory
        self._history:   list[dict] = []
        self._mem_thread: threading.Thread | None = None
        print(f"[brain] Ollama client ready — model: {OLLAMA_MODEL}")

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn, return Aiko's full response string.
        Handles memory retrieval, optional web search, and memory storage.
        """
        # 1. retrieve relevant long-term memories
        memories     = self._memory.search(user_input)
        memory_block = self._memory.format_for_context(memories)

        # 2. build system prompt with memories + search instruction
        system = _load_persona()
        system = _inject_search_instruction(system)
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        # 3. append user turn to rolling history
        self._history.append({"role": "user", "content": user_input})

        # 4. trim history to context window
        trimmed  = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        messages = [{"role": "system", "content": system}] + trimmed

        # 5. first LLM call — may return a search trigger
        raw_response = self._call_llm(messages)

        # 6. intercept search trigger if present
        search_match = _SEARCH_RE.search(raw_response)
        if search_match:
            query = search_match.group(1).strip()
            print(f"\n[search] {query}", flush=True)
            results = web_search(query)

            # inject results and call LLM again for final response
            messages.append({"role": "assistant", "content": raw_response})
            messages.append({"role": "user",      "content": results})
            response_text = self._stream_response(messages)
        else:
            # no search needed — stream the first response
            print("\nAiko-chan: ", end="", flush=True)
            # re-stream cleanly from the already-fetched response
            response_text = self._stream_text(raw_response)

        # 7. append Aiko's response to rolling history
        self._history.append({"role": "assistant", "content": response_text})

        # 8. persist to long-term memory in background
        self._store_async(user_input, response_text)

        return response_text

    def reset_context(self) -> None:
        """Clear the short-term rolling context (long-term memory persists)."""
        self._history.clear()
        print("[brain] Short-term context cleared.")

    def wait_for_memory(self) -> None:
        """Block until any pending memory write completes. Call before exit."""
        if self._mem_thread and self._mem_thread.is_alive():
            print("[memory] Waiting for memory write to finish...")
            self._mem_thread.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Non-streaming LLM call — used for the first pass to detect search triggers.
        Returns the full response string without printing.
        """
        response = self._client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=False,
        )
        if hasattr(response, "message"):
            return response.message.content or ""
        return response.get("message", {}).get("content", "") or ""

    def _stream_response(self, messages: list[dict]) -> str:
        """
        Streaming LLM call — used for the final response after search injection.
        Prints tokens live and returns full assembled string.
        """
        full_response = []

        stream = self._client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
        )

        print("\nAiko-chan: ", end="", flush=True)
        for chunk in stream:
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            else:
                token = chunk.get("message", {}).get("content", "") or ""
            print(token, end="", flush=True)
            full_response.append(token)
        print(flush=True)

        return "".join(full_response)

    def _stream_text(self, text: str) -> str:
        """
        Print already-fetched text token by token (simulated stream).
        Returns the text unchanged.
        """
        print(text, flush=True)
        return text

    def _store_async(self, user_input: str, response_text: str) -> None:
        """Fire memory write in a background thread — non-blocking."""
        def _write():
            try:
                self._memory.add([
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": response_text},
                ])
            except Exception as exc:
                print(f"[memory] async write failed: {exc}")
        # wait for previous write before starting a new one
        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join()

        self._mem_thread = threading.Thread(target=_write, daemon=True)
        self._mem_thread.start()
