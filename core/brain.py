"""
core/brain.py

Aiko's cognitive loop.
  - Retrieves relevant memories before each turn
  - Streams Ollama response
  - Stores the turn into long-term memory after each response
"""

import os
from ollama import Client
from core.persona import get_system_prompt
from core.memory import AikoMemory


# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "llama3.2")

# how many past turns to keep in the active context window
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))


# ── brain ─────────────────────────────────────────────────────────────────────

class AikoBrain:
    """
    Aiko's conversational core.
    Manages the short-term context window and long-term mem0 memory.
    """

    def __init__(self, memory: AikoMemory) -> None:
        self._client  = Client(host=OLLAMA_BASE_URL)
        self._memory  = memory
        self._history: list[dict] = []   # rolling short-term context
        print(f"[brain] Ollama client ready — model: {OLLAMA_MODEL}")

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        """
        Process one user turn, return Aiko's full response string.
        Handles memory retrieval, context assembly, LLM call, and storage.
        """
        # 1. retrieve relevant long-term memories
        memories      = self._memory.search(user_input)
        memory_block  = self._memory.format_for_context(memories)

        # 2. build the system prompt (inject memories when present)
        system = get_system_prompt()
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

        # 8. persist this turn to long-term memory (async-ish — runs inline)
        self._memory.add([
            {"role": "user",      "content": user_input},
            {"role": "assistant", "content": response_text},
        ])

        return response_text

    def reset_context(self) -> None:
        """Clear the short-term rolling context (long-term memory persists)."""
        self._history.clear()
        print("[brain] Short-term context cleared.")

    # ── internal ──────────────────────────────────────────────────────────────

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
            token = chunk["message"]["content"]
            print(token, end="", flush=True)
            full_response.append(token)
        print()  # newline after stream ends

        return "".join(full_response)
