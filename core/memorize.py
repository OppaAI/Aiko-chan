"""
core/memorize.py
Aiko's persistent memory via mem0 + Qdrant.
Abstracts all mem0 calls so think.py stays clean.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in Qdrant payload, enabling Ebbinghaus-style exponential decay scoring.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - decay logic lives in core/decay.py (pure math, no I/O).
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
import os
import time
from typing import Optional

from mem0 import Memory
from qdrant_client import QdrantClient

from core.decay import compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD

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
        },
    },
}

USER_ID = os.getenv("USER_ID", "OppaAI")
QDRANT_COLLECTION = "aiko_memory"

# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Thin wrapper around mem0 Memory with Ebbinghaus decay lifecycle.

    Access tracking:
        Every search() call updates Qdrant payload fields (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Cleanup:
        Call cleanup() periodically (e.g. once per session startup) to prune
        memories whose weighted decay score has dropped below CLEANUP_THRESHOLD.
        Memories under 14 days old are grace-protected from deletion.
    """

    def __init__(self, silent: bool = False) -> None:
        """Initialise mem0 Memory, Qdrant client, and connect."""
        os.environ.setdefault(
            "FASTEMBED_CACHE_PATH",
            os.path.expanduser(os.getenv("FASTEMBED_CACHE_PATH", "~/.cache/fastembed"))
        )
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        if not silent:
            print("[memorize] Connecting to Qdrant and initialising mem0...")

        self._mem = Memory.from_config(MEM0_CONFIG)
        self._qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)

        if not silent:
            print("[memorize] Ready.")

    # ── write ──────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store a conversation turn (or batch) into long-term memory.

        Runs synchronously — LLM extraction completes before the next chat
        turn begins, preventing Ollama VRAM context slot conflicts.

        Returns True on success, False on failure so callers can log/alert.
        """
        try:
            t = time.perf_counter()
            self._mem.add(messages, user_id=user_id)
            print(f"[memorize] Save completed in {time.perf_counter() - t:.2f}s")
            return True
        except Exception as e:
            print(f"[memorize-error] Save failed: {e}")
            return False

    # ── read ───────────────────────────────────────────────────────────────────

    def search(self, query: str,  user_id: str = USER_ID,  limit: int = 5) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.

        Side-effect: increments access_count and updates last_accessed_at
        in Qdrant payload for each returned memory, feeding decay scoring.
        """
        results = self._mem.search(query, filters={"user_id": user_id}, limit=limit)
        if isinstance(results, dict):
            results = results.get("results", [])
        results = results or []

        # Track access for each returned memory
        if results:
            now = datetime.now(timezone.utc).isoformat()
            for r in results:
                mem_id = str(r.get("id", ""))
                if not mem_id:
                    continue
                try:
                    pts = self._qdrant.retrieve(
                        collection_name=QDRANT_COLLECTION,
                        ids=[mem_id],
                        with_payload=True,
                    )
                    current_count = 0
                    if pts:
                        current_count = pts[0].payload.get("access_count", 0) or 0

                    self._qdrant.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={
                            "access_count": min(current_count + 1, 255),  # cap at 255
                            "last_accessed_at": now,
                        },
                        points=[mem_id],
                    )
                except Exception as e:
                    print(f"[memorize-warn] Access tracking failed for {mem_id}: {e}")

        return results

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
            "",
        ]
        for m in memories:
            text = m.get("memory") or m.get("text") or str(m)
            created_at = m.get("created_at")
            if created_at:
                try:
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

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id: str = USER_ID,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run: bool = False,
    ) -> dict:
        """
        Prune decayed memories below threshold score.

        Fetches all memories, batch-retrieves Qdrant payloads (single round-trip),
        evaluates decay score via should_cleanup(), and deletes candidates via
        mem0 SDK to keep metadata + vector store in sync.

        Grace period (14 days) protects newly created memories from deletion
        even if they score below threshold.

        Args:
            threshold: Override decay threshold (default: CLEANUP_THRESHOLD = 0.05).
            dry_run:   If True, report candidates without deleting.

        Returns dict with counts: deleted, kept, failed, candidates (dry_run only).
        """
        all_mems = self._mem.get_all(filters={"user_id": user_id})
        if isinstance(all_mems, dict):
            all_mems = all_mems.get("results", [])
        all_mems = all_mems or []

        if not all_mems:
            return {"deleted": 0, "kept": 0, "failed": 0}

        # Batch retrieve all Qdrant payloads — single round-trip
        mem_ids = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)

        candidates = []
        kept = 0

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            ac, la = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            if should_cleanup(ac, la, created_at):
                w = compute_weighted_score(ac, la)
                candidates.append({
                    "id": mem_id,
                    "memory": m.get("memory", "")[:120],
                    "access_count": ac,
                    "weighted_score": round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        candidates.sort(key=lambda x: x["weighted_score"])

        if dry_run:
            print(f"[memorize] Dry run: {len(candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": candidates}

        deleted = []
        failed = []
        for c in candidates:
            try:
                self._mem.delete(memory_id=c["id"])
                deleted.append(c["id"])
            except Exception as e:
                failed.append({"id": c["id"], "error": str(e)})

        print(f"[memorize] Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    # ── debug ──────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str = USER_ID) -> list[dict]:
        """Return all stored memories for a user (for debugging)."""
        results = self._mem.get_all(filters={"user_id": user_id})
        if isinstance(results, dict):
            return results.get("results", [])
        return results or []

    def clear(self, user_id: str = USER_ID) -> None:
        """Wipe all memories for a user. Use carefully."""
        self._mem.delete_all(user_id=user_id)
        print(f"[memorize] Cleared all memories for user '{user_id}'.")

    # ── internal ───────────────────────────────────────────────────────────────

    def _batch_get_payloads(self, mem_ids: list[str]) -> dict:
        """
        Batch retrieve access_count + last_accessed_at from Qdrant.
        Single round-trip — eliminates N+1 query problem for cleanup/stats.
        Returns dict: {mem_id: (access_count, last_accessed_at)}
        """
        if not mem_ids:
            return {}
        try:
            pts = self._qdrant.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=mem_ids,
                with_payload=True,
            )
            return {
                str(p.id): (
                    p.payload.get("access_count", 0) or 0,
                    p.payload.get("last_accessed_at", "never"),
                )
                for p in pts
            }
        except Exception as e:
            print(f"[memorize-warn] Batch payload fetch failed: {e}")
            return {}
