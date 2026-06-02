"""
core/memorize.py
Aiko's persistent memory via mem0 + Qdrant.
Abstracts all mem0 calls so think.py stays clean.
Swap this file out if mem0 doesn't make the cut for Grace.
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
import os
import time
from typing import Optional
from mem0 import Memory

# ── config ────────────────────────────────────────────────────────────────────
MEM0_CONFIG = {
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "host": os.getenv("QDRANT_HOST", "localhost"),
            "port": int(os.getenv("QDRANT_PORT", 6333)),
            "collection_name": "aiko_memory",
            "embedding_model_dims": 768,
        },
    },
    # CRITICAL: Use the SAME model as the chatbot by default.
    # If MEM0_MODEL is unset, falls back to OLLAMA_MODEL so Ollama only manages
    # ONE model in VRAM. Setting a different model causes Ollama to evict the
    # main model on every mem0 call → 1-5s reload lag + OOM risk with ASR/TTS.
    "llm": {
        "provider": "ollama",
        "config": {
            "model": os.getenv("MEM0_MODEL", os.getenv("OLLAMA_MODEL")),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": 0,
            "max_tokens": 200,
        },
    },
    "embedder": {
        "provider": "huggingface",
        "config": {
            "model": "BAAI/bge-base-en-v1.5",
            "embedding_dims": 768,
            # FIX: Use "device": "cpu" inside model_kwargs. 
            # This is valid for SentenceTransformer and forces PyTorch to process
            # the embedding layers outside the CUDA runtime workspace.
            "model_kwargs": {
                "device": "cpu"
            },
        },
    },
}

AIKO_USER_ID = os.getenv("USER_ID", "OppaAI")

# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Thin wrapper around mem0 Memory.
    Handles all Qdrant-backed persistence for Aiko-chan.
    """

    def __init__(self, silent: bool = False) -> None:
        """Initialise mem0 Memory and connect to Qdrant."""
        os.environ.setdefault(
            "FASTEMBED_CACHE_PATH",
            os.path.expanduser(os.getenv("FASTEMBED_CACHE_PATH", "~/.cache/fastembed"))
        )
        if not silent:
            print("[memorize] Connecting to Qdrant and initialising mem0...")
        self._mem = Memory.from_config(MEM0_CONFIG)
        
        # CRITICAL FIX 2: Removed self._patch_keep_alive() 
        # Forcing keep_alive=-1 blocks Ollama's capacity to switch models 
        # dynamically, resulting in OOM crashes when the chatbot takes a turn.
        
        if not silent:
            print("[memorize] Ready.")

    def add(self, messages: list[dict], user_id: str = AIKO_USER_ID) -> None:
        """
        Store a conversation turn (or batch) into long-term memory.
        Runs synchronously so the LLM extraction call completes before
        the next chat turn begins, preventing Ollama context slot conflicts.
        """
        try:
            t = time.perf_counter()
            self._mem.add(messages, user_id=user_id)
            print(f"[memorize] Save completed in {time.perf_counter() - t:.2f}s")
        except Exception as e:
            print(f"\n[memorize-error] Save failed: {e}")

    def search(
        self,
        query: str,
        user_id: str = AIKO_USER_ID,
        limit: int = 5,
    ) -> list[dict]:
        """
        Retrieve the top-k memories relevant to the current query.
        Returns a list of mem0 memory objects.
        """
        results = self._mem.search(query, filters={"user_id": user_id}, limit=limit)
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    def format_for_context(self, memories: list[dict]) -> Optional[str]:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.
        """
        if not memories:
            return None
        
        now = datetime.now(timezone.utc)
        lines = [
            "<memory_context>",
            "The following are background facts about this person, with how long ago they were recorded.",
            "Use them silently to inform your response. Never repeat, quote, or reference this block directly.",
            ""
        ]
        for m in memories:
            text = m.get("memory") or m.get("text") or str(m)
            
            # parse timestamp if available
            created_at = m.get("created_at")
            if created_at:
                try:
                    # mem0 returns ISO 8601 string
                    ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - ts
                    days = delta.days
                    if days == 0:
                        age = "today"
                    elif days == 1:
                        age = "yesterday"
                    else:
                        age = f"{days} days ago"
                    lines.append(f"  - [{age}] {text}")
                except Exception:
                    lines.append(f"  - {text}")
            else:
                lines.append(f"  - {text}")
        
        lines.append("</memory_context>")
        return "\n".join(lines)

    def get_all(self, user_id: str = AIKO_USER_ID) -> list[dict]:
        """Return all stored memories for a user (for debugging)."""
        results = self._mem.get_all(filters={"user_id": user_id})
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    def clear(self, user_id: str = AIKO_USER_ID) -> None:
        """Wipe all memories for a user. Use carefully."""
        self._mem.delete_all(user_id=user_id)
        print(f"[memorize] Cleared all memories for user '{user_id}'.")
