"""
core/think.py

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

from core.memorize import AikoMemorize
from core.speak import AikoSpeak
from core.tools import web_search


# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "ministral-3:3b-instruct-2512-q4_K_M")

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "soul.md"

CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))

_SEARCH_RE = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)


def _load_persona() -> str:
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    return _PERSONA_PATH.read_text(encoding="utf-8").strip()


def _inject_search_instruction(system: str) -> str:
    return system + """

## Web Search
You MUST use this exact format when you need current information:
[SEARCH: your query here]

Output ONLY that line, nothing else, before waiting for results.
Do NOT answer questions about current events, news, or real-time data without searching first.
Examples:
- User asks about today's news → output: [SEARCH: top news today]
- User asks who is PM of Canada → output: [SEARCH: current Prime Minister Canada 2026]
"""

# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    Manages the short-term context window and long-term mem0 memory.
    Memory writes run in a background thread — never blocks the chat loop.
    """

    def __init__(self, memorize: AikoMemorize, voice: bool = True) -> None:
        """Initialise Ollama client, memory, persona cache, and optional TTS."""
        self._client     = Client(host=OLLAMA_BASE_URL)
        self._memorize   = memorize
        self._speak      = AikoSpeak() if voice else None
        self._persona    = _load_persona()
        self._history:   list[dict] = []
        self._mem_thread: threading.Thread | None = None
        print(f"[think] Ollama client ready — model: {OLLAMA_MODEL}")
        print(f"[think] Voice output: {'on' if voice else 'off'}")

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        # 1. retrieve relevant long-term memories
        memories     = self._memorize.search(user_input)
        memory_block = self._memorize.format_for_context(memories)

        # 2. build system prompt
        system = _inject_search_instruction(self._persona)
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        # 3. append user turn
        self._history.append({"role": "user", "content": user_input})

        # 4. trim history
        trimmed  = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        messages = [{"role": "system", "content": system}] + trimmed

        # 5. stream first response — TTS feeds live, search detection inline
        response_text = self._stream_response(messages)

        # 6. handle search trigger if present
        search_match = _SEARCH_RE.search(response_text)
        if search_match:
            query = search_match.group(1).strip()
            print(f"\n[search] {query}", flush=True)
            results = web_search(query)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user",      "content": results})
            response_text = self._stream_response(messages)

        # 7. append to history
        self._history.append({"role": "assistant", "content": response_text})

        # 8. persist to memory
        self._store_async(user_input, response_text)

        return response_text

    def reset_context(self) -> None:
        """Clear the short-term rolling context (long-term memory persists)."""
        self._history.clear()
        print("[think] Short-term context cleared.")

    def wait_for_memory(self) -> None:
        if self._mem_thread and self._mem_thread.is_alive():
            print("[memorize] Waiting for memory write to finish...")
            self._mem_thread.join()

    # ── internal ──────────────────────────────────────────────────────────────

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Non-streaming LLM call — used for the first pass to detect search triggers.
        Returns the full response string without printing.
        """
        try:
            response = self._client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=False,
            )
            if hasattr(response, "message"):
                return response.message.content or ""
            return response.get("message", {}).get("content", "") or ""
        except Exception as exc:
            print(f"[think] Ollama call failed: {exc}")
            return ""

    def _stream_response(self, messages: list[dict]) -> str:
        full_response = []
        tts_started   = False
        try:
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
                # don't feed to TTS if response looks like a search trigger
                assembled = "".join(full_response)
                if self._speak and token and not _SEARCH_RE.search(assembled):
                    self._speak.feed(token)
                    tts_started = True
            print(flush=True)
            if self._speak and tts_started:
                self._speak.play_async()
        except Exception as exc:
            print(f"\n[think] stream failed: {exc}")
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
                self._memorize.add([
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": response_text},
                ])
            except Exception as exc:
                print(f"[memorize] async write failed: {exc}")

        if self._mem_thread and self._mem_thread.is_alive():
            self._mem_thread.join()

        self._mem_thread = threading.Thread(target=_write, daemon=True)
        self._mem_thread.start()
