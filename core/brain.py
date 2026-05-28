"""
core/brain.py

Aiko's cognitive loop.
  - Retrieves relevant memories before each turn
  - Streams Ollama response
  - Stores the turn into long-term memory after each response (background thread)
"""

import os
import threading
from pathlib import Path
from ollama import Client
from core.memory import AikoMemory


# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")

# soul.md lives at project root — resolve relative to this file
_PERSONA_PATH = Path(__file__).resolve().parent.parent / "soul.md"

# how many past turns to keep in the active context window
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))


def _load_persona() -> str:
    """Load Aiko's system prompt from soul.md at project root."""
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    return _PERSONA_PATH.read_text(encoding="utf-8").strip()


# ── brain ─────────────────────────────────────────────────────────────────────

class AikoBrain:
    """
    Aiko's conversational core.
    Manages the short-term context window and long-term mem0 memory.
    Memory writes run in a background thread — never blocks the chat loop.
    """

    def __init__(self, memory: AikoMemory) -> None:
        self._client  = Client(host=OLLAMA_BASE_URL)
        self._memory  = memory
        self._history: list[dict] = []   # rolling short-term context
        self._mem_thread: threading.Thread | None = None
        print(f"[brain] Ollama client ready — model: {OLLAMA_MODEL}")

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn, return Aiko's full response string.
        Handles memory retrieval, context assembly, LLM call, and storage.
        """
        # 1. retrieve relevant long-term memories
        memories     = self._memory.search(user_input)
        memory_block = self._memory.format_for_context(memories)

        # 2. build the system prompt (inject memories when present)
        system = _load_persona()
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        # 3. append user turn to rolling history
        self._history.append({"role": "user", "content": user_input})

        # 4. trim history to context window
        trimmed = self._history[-(CONTEXT_WINDOW_TURNS * 2):]

        # 5. assemble full message list for Ollama
        messages = [{"role": "system", "content": system}] + trimmed

        # 6. stream response from Ollama
        response_text = self._stream_response(messages)

        # 7. append Aiko's response to rolling history
        self._history.append({"role": "assistant", "content": response_text})

        # 8. persist to long-term memory in background — never blocks chat loop
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

    def _store_async(self, user_input: str, response_text: str) -> None:
        """Fire memory write in a background thread — non-blocking."""
        def _write():
            self._memory.add([
                {"role": "user",      "content": user_input},
                {"role": "assistant", "content": response_text},
            ])

        # wait for previous write to finish before starting a new one
        # prevents concurrent writes to the same Qdrant collection
        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join()

        self._mem_thread = threading.Thread(target=_write, daemon=True)
        self._mem_thread.start()

    def _stream_response(self, messages: list[dict]) -> str:
        """
        Stream tokens from Ollama, printing them live to stdout.
        Returns the full assembled response string.
        """
        full_response = []

        stream = self._client.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            stream=True,
        )

        print("\nAiko-chan: ", end="", flush=True)
        for chunk in stream:
            # handle both dict and object response formats
            if hasattr(chunk, "message"):
                token = chunk.message.content or ""
            else:
                token = chunk.get("message", {}).get("content", "") or ""
            print(token, end="", flush=True)
            full_response.append(token)
        print(flush=True)  # newline after stream ends

        return "".join(full_response)