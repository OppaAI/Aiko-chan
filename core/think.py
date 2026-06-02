"""
core/think.py

Aiko's cognitive loop.
  - Retrieves relevant memories before each turn
  - Intercepts [SEARCH: query] triggers for web search
  - Streams Ollama response to console + TTS simultaneously
  - Stores the turn into long-term memory after each response (background thread)
"""

import logging
import os
import threading
from ollama import Client
from pathlib import Path
import queue
import re
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("phonemizer").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

from core.memorize import AikoMemorize
from core.speak    import AikoSpeak
#from core.tools    import web_search


# ── config ────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL      = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL         = os.getenv("OLLAMA_MODEL",    "ministral-3:3b-instruct-2512-q4_K_M")
CONTEXT_WINDOW_TURNS = int(os.getenv("CONTEXT_WINDOW_TURNS", 20))

_PERSONA_PATH = Path(__file__).resolve().parent.parent / "persona" / "soul.md"
#_SEARCH_RE    = re.compile(r"\[SEARCH:\s*(.+?)\]", re.IGNORECASE)


def _load_persona() -> str:
    if not _PERSONA_PATH.exists():
        raise FileNotFoundError(f"soul.md not found at {_PERSONA_PATH}")
    return _PERSONA_PATH.read_text(encoding="utf-8").strip()

#def _inject_search_instruction(system: str) -> str:
#    return system + """

## Web Search
#You have access to a web search tool. Use it ONLY for real-time or time-sensitive facts you cannot know: current news, live prices, today's weather, recent events.

#NEVER search for:
#- Anything about yourself, your identity, your creator, or your relationship with the user
#- Greetings, chit-chat, or conversational questions
#- General knowledge, history, science, or anything you already know
#- Questions the user is asking YOU personally
#
#To search, output ONLY this on its own line, nothing else before or after:
#[SEARCH: your query here]
#
## Grounding Rule
#When search results are provided to you, they represent current ground truth.
#You MUST base your answer on the search results.
#Do NOT contradict or ignore search results with your own prior knowledge.
#If the search results conflict with what you think you know, the search results are correct.
#"""

# ── think ─────────────────────────────────────────────────────────────────────

class AikoThink:
    """
    Aiko's conversational core.
    speak is injected pre-warmed from cli.py.
    LLM warmup starts immediately on init in a background thread.
    cli.py calls join_warmup() to block until both are ready before showing prompt.
    """

    def __init__(self, memorize: AikoMemorize, speak: AikoSpeak | None = None) -> None:
        self._client         = Client(host=OLLAMA_BASE_URL)
        self._memorize       = memorize
        self._speak          = speak
        self._persona        = _load_persona()
        self._history:       list[dict] = []
        self._mem_queue = queue.Queue()
        self._mem_worker = threading.Thread(target=self._mem_write_loop, daemon=True)
        self._mem_worker.start()
        self._warmup_thread: threading.Thread | None = None

        #print(f"[think] Ollama client ready — model: {OLLAMA_MODEL}")
        #print(f"[think] Voice output: {'on' if speak else 'off'}")

        self._warmup_thread = threading.Thread(target=self._warmup_llm, daemon=True)
        self._warmup_thread.start()

    def _warmup_llm(self) -> None:
        try:
            self._client.chat(
                model=OLLAMA_MODEL,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                options={
                    "num_predict": 1,
                    "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", 2048)),
                },
            )
        except Exception:
            pass

    def join_warmup(self) -> None:
        """Block until LLM warmup completes. Called by cli.py before showing prompt."""
        if self._warmup_thread and self._warmup_thread.is_alive():
            self._warmup_thread.join()

    # ── public api ────────────────────────────────────────────────────────────

    def chat(self, user_input: str, token_callback=None) -> str:
        self._token_callback = token_callback   # store for _stream_response
        # 1. retrieve relevant long-term memories
        memories     = self._memorize.search(user_input)
        memory_block = self._memorize.format_for_context(memories)

        # 2. build system prompt
        system = self._persona
        if memory_block:
            system = f"{system}\n\n{memory_block}"

        # 3. append user turn
        self._history.append({"role": "user", "content": user_input})

        # 4. trim history to context window
        trimmed  = self._history[-(CONTEXT_WINDOW_TURNS * 2):]
        messages = [{"role": "system", "content": system}] + trimmed

        # 5. stream first response
        response_text = self._stream_response(messages)

        # 6. handle search trigger if present
        #search_match = _SEARCH_RE.search(response_text)
        #if search_match:
        #    query = search_match.group(1).strip()
        #    if not self._token_callback:
        #        print(f"\n[search] {query}", flush=True)
        #    if self._token_callback:
        #        self._token_callback(f"__SEARCHING__: {query}")
        #    results = web_search(query)
        #    messages.append({"role": "assistant", "content": response_text})
        #    messages.append({"role": "user",      "content": results})
        #    response_text = self._stream_response(messages)

        # 7. append assistant turn to history
        self._history.append({"role": "assistant", "content": response_text})

        # 8. persist to memory (background)
        self._store_async(user_input, response_text)

        return response_text

    def reset_context(self) -> None:
        self._history.clear()

    def set_speak(self, speak):
      """Hot-swap the TTS backend. Pass None to silence, speak instance to restore."""
      self._speak = speak
    
    def wait_for_memory(self) -> None:
        self._mem_queue.join()
      
    # ── internal ──────────────────────────────────────────────────────────────

    def _stream_response(self, messages: list[dict]) -> str:
        """
        Stream LLM response to console and TTS simultaneously.
        Console printing is the single source of truth — speak.py is silent.
        TTS skipped if response is a search trigger.
        """
        full_response = []
        tts_started   = False

        # Buffering mechanism to intercept [SEARCH: query] tags
        buffer = ""
        is_searching = False
        buffering_active = True

        try:
            stream = self._client.chat(
                model=OLLAMA_MODEL,
                messages=messages,
                stream=True,
                options={
                    "num_ctx":        int(os.getenv("OLLAMA_NUM_CTX", 2048)),
                    "temperature":    0.75,   # creative but not unhinged
                    "repeat_penalty": 1.18,   # firm anti-loop
                    "repeat_last_n":  128,    # long lookback for repetition detection
                    "num_predict":    400,    # hard cap — no infinite loops
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

                # Process token streaming and buffering
                if buffering_active:
                    buffer += token
                    buffer_clean = buffer.lower().replace(" ", "")
                
                    if is_searching:
                        # Already confirmed a search tag — keep buffering until closing ]
                        if "]" in buffer:
                            buffering_active = False
                            # Full [search: query] captured, do NOT stream it
                    elif "[search:".startswith(buffer_clean):
                        # Still a valid prefix — check if we've crossed the threshold
                        if "[search:" in buffer_clean:
                            is_searching = True
                    else:
                        # Not a search tag — flush buffer
                        buffering_active = False
                        if self._token_callback and buffer:
                            self._token_callback(buffer)
                        elif not self._token_callback:
                            if not "".join(full_response[:-1]):
                                print("\nAiko-chan: ", end="", flush=True)
                            print(buffer, end="", flush=True)
                else:
                    if not is_searching:
                        # Stream normally
                        if self._token_callback:
                            self._token_callback(token)
                        else:
                            print(token, end="", flush=True)

                assembled = "".join(full_response)
                if self._speak and token: # and not _SEARCH_RE.search(assembled):
                    self._speak.feed(token)
                    tts_started = True

            # If we completed the stream but never broke out of buffering (e.g. short regular text)
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
            msg = f"[think] stream failed: {exc}"
            if self._token_callback:
                self._token_callback(msg)
            else:
                print(f"\n{msg}")

        return "".join(full_response)

    def _store_async(self, user_input: str, response_text: str) -> None:
        """Enqueue memory write — never blocks the chat path."""
        self._mem_queue.put((user_input, response_text))
    
    def _mem_write_loop(self) -> None:
        """Serial background worker — processes writes in order."""
        while True:
            user_input, response_text = self._mem_queue.get()
            try:
                self._memorize.add([
                    {"role": "user",      "content": user_input},
                    {"role": "assistant", "content": response_text},
                ])
            except Exception as exc:
                print(f"[memorize] async write failed: {exc}")
            finally:
                self._mem_queue.task_done()
