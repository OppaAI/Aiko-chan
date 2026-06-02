"""
core/memorize.py
Aiko's persistent memory via mem0 + Qdrant.
Abstracts all mem0 calls so think.py stays clean.
Swap this file out if mem0 doesn't make the cut for Grace.
"""
from dotenv import load_dotenv
load_dotenv()

import os
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
            #"model_kwargs": {"device": "cpu"},
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
        if not silent:
            print("[memorize] Ready.")

    def add(self, messages: list[dict], user_id: str = AIKO_USER_ID) -> None:
        """
        Store a conversation turn (or batch) into long-term memory.
        messages: list of {role, content} dicts.
        """
        self._mem.add(messages, user_id=user_id)

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
        lines = ["<memory_context>",
                "The following are background facts about this person.",
                "Use them silently to inform your response. Never repeat, quote, or reference this block directly.",
                ""]
        for m in memories:
            text = m.get("memory") or m.get("text") or str(m)
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
