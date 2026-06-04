"""
core/memorize.py
Aiko's persistent memory — custom backend via Qdrant + fastembed + Ollama.
Abstracts all memory calls so think.py stays clean.

Memory lifecycle:
  - Every search() call increments access_count and updates last_accessed_at
    in Qdrant payload, enabling Ebbinghaus-style exponential decay scoring.
  - dream() runs nightly (00:00) as a consolidation pass — no new vectors
    are written. It boosts salient memories, merges near-duplicates, then
    prunes decayed entries. Order matters: boost before prune so boosted
    memories aren't immediately swept.
  - cleanup() deletes memories below decay threshold, with grace period
    protection for newly created entries.
  - Decay logic lives in core/forget.py (pure math, no I/O).
  - Pinned memories (created via pin()) are permanently immune to decay
    cleanup and dream pruning. The pinned flag lives in Qdrant payload.

Dream pass overview:
  1. Boost  — increment access_count on memories matching salience heuristics
              (keyword signals, high prior access, recency) so they survive decay.
  2. Merge  — cosine-similarity search per memory; near-duplicates above
              threshold are collapsed: keep the higher access_count copy,
              delete the redundant one to stay in sync.
              Pinned memories are never chosen as the loser in a merge.
  3. Prune  — standard cleanup() pass; runs after boost so newly protected
              memories aren't caught in the sweep.
              Pinned memories are skipped entirely.

Custom backend (replaces mem0):
  - _MemoryBackend handles LLM-based fact extraction, fastembed embeddings,
    and direct Qdrant upsert/search/delete/scroll.
  - Extraction prompt is tuned for small models: asks for a JSON array of
    atomic facts, strips <think> blocks for CoT models, skips trivial turns.
  - All Qdrant schema fields (memory, user_id, created_at, access_count,
    last_accessed_at, pinned) are owned by this module — no hidden mem0 schema.
"""
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime, timezone
import json
import os
import re
import time
import uuid
from typing import Optional

import httpx
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Filter,
    FieldCondition,
    MatchValue,
    PointStruct,
    VectorParams,
)

from core.forget import compute_weighted_score, should_cleanup, CLEANUP_THRESHOLD
from core.log import get_logger

log = get_logger(__name__)

# ── boot labels ───────────────────────────────────────────────────────────────

BOOT_LABELS = {
    'mem_qdrant':   'Connecting to Qdrant...',
    'mem_cleanup':  'Running memory cleanup...',
    'mem_ready':    'Memory backend ready',
}

# ── constants ─────────────────────────────────────────────────────────────────

QDRANT_COLLECTION  = "aiko_memory"
EMBED_MODEL        = "BAAI/bge-base-en-v1.5"
EMBED_DIMS         = 768
USER_ID            = os.getenv("USER_ID", "OppaAI")

# Cosine similarity threshold for near-duplicate detection during dream pass.
# 0.92 is conservative — only collapses near-identical phrasings.
# Lower (e.g. 0.85) catches more semantic duplicates but risks false merges.
DREAM_MERGE_THRESHOLD = float(os.getenv("DREAM_MERGE_THRESHOLD", 0.92))

# access_count boost applied to salient memories during dream pass.
DREAM_BOOST_AMOUNT = int(os.getenv("DREAM_BOOST_AMOUNT", 2))

# Salience keywords — memories containing these are boosted during dream pass.
_SALIENCE_KEYWORDS = frozenset([
    "name", "called", "likes", "loves", "hates", "dislikes", "always", "never",
    "important", "remember", "favourite", "favorite", "birthday", "works",
    "lives", "studying", "job", "afraid", "dream", "goal",
])

# Minimum conversation size (chars) worth sending to LLM for extraction.
# Skips trivial turns (greetings, one-word replies) to save inference time.
_EXTRACT_MIN_CHARS = int(os.getenv("MEMORY_EXTRACT_MIN_CHARS", 80))

# Extraction prompt — tuned for small models.
# Asks for a flat JSON array of atomic fact strings. Nothing else.
_EXTRACT_PROMPT = """\
Extract memorable facts about the user from this conversation.
Return ONLY a JSON array of short strings. Each string is one atomic fact.
Facts should be about the user's preferences, identity, life, or goals.
If nothing is worth remembering, return an empty array: []
Do NOT include assistant statements. Do NOT explain. No markdown.

Conversation:
{conversation}"""

# ── custom memory backend ─────────────────────────────────────────────────────

class _MemoryBackend:
    """
    Lightweight replacement for mem0.Memory.

    Responsibilities:
      - LLM-based fact extraction via Ollama /api/chat
      - Embedding via fastembed (ONNX, CPU-friendly)
      - Qdrant upsert / vector search / payload scroll / delete

    Public API is intentionally minimal — only what AikoMemorize needs:
      add(), search(), get_all(), delete(), delete_all()

    Collection is created automatically on first use if it doesn't exist.
    """

    def __init__(
        self,
        qdrant_host:     str,
        qdrant_port:     int,
        ollama_base_url: str,
        model:           str,
        fastembed_cache: Optional[str] = None,
    ) -> None:
        self._qdrant   = QdrantClient(host=qdrant_host, port=qdrant_port)
        self._ollama   = ollama_base_url.rstrip("/")
        self._model    = model
        self._embedder = TextEmbedding(
            model_name=EMBED_MODEL,
            cache_dir=fastembed_cache,
        )
        self._ensure_collection()

    # ── collection setup ──────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create Qdrant collection if it doesn't already exist."""
        existing = {c.name for c in self._qdrant.get_collections().collections}
        if QDRANT_COLLECTION not in existing:
            self._qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBED_DIMS, distance=Distance.COSINE),
            )
            log.info(f"Created Qdrant collection '{QDRANT_COLLECTION}'.")

    # ── embedding ─────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Embed a single string with fastembed. Returns a plain float list."""
        return list(self._embedder.embed([text]))[0].tolist()

    # ── extraction ────────────────────────────────────────────────────────────

    def _should_extract(self, messages: list[dict]) -> bool:
        """
        Return False for trivial turns not worth extracting.
        Skips LLM call entirely for greetings, one-word replies, etc.
        """
        total = sum(len(m.get("content") or "") for m in messages)
        return total >= _EXTRACT_MIN_CHARS

    def _extract_facts(self, messages: list[dict]) -> list[str]:
        """
        Send conversation to Ollama LLM and parse the returned JSON fact array.

        Strips <think>…</think> blocks before parsing (CoT models like
        Ministral/Qwen emit these and break JSON parsing otherwise).

        Returns a list of fact strings, or [] on any failure.
        """
        if not self._should_extract(messages):
            return []

        convo = "\n".join(
            f"{m['role'].upper()}: {m['content']}"
            for m in messages
            if (m.get("content") or "").strip()
        )
        prompt = _EXTRACT_PROMPT.format(conversation=convo)

        try:
            resp = httpx.post(
                f"{self._ollama}/api/chat",
                json={
                    "model":  self._model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 512},
                },
                timeout=45,
            )
            resp.raise_for_status()
            raw = resp.json()["message"]["content"].strip()
        except Exception as e:
            log.warning(f"Extraction LLM call failed: {e}")
            return []

        # Strip CoT think blocks before JSON parsing
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

        # Strip accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()

        try:
            facts = json.loads(raw)
            if isinstance(facts, list):
                return [f.strip() for f in facts if isinstance(f, str) and f.strip()]
        except json.JSONDecodeError:
            log.warning(f"Failed to parse extraction JSON: {raw[:200]!r}")

        return []

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str) -> list[str]:
        """
        Extract facts from messages and upsert each as a Qdrant point.

        Returns list of new memory IDs (UUIDs). Empty list if nothing extracted
        or extraction fails — callers treat this as a no-op, not an error.
        """
        facts = self._extract_facts(messages)
        if not facts:
            return []

        now  = datetime.now(timezone.utc).isoformat()
        ids  = []

        for fact in facts:
            mem_id = str(uuid.uuid4())
            try:
                vector = self._embed(fact)
                self._qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=[PointStruct(
                        id=mem_id,
                        vector=vector,
                        payload={
                            "memory":           fact,
                            "user_id":          user_id,
                            "created_at":       now,
                            "access_count":     0,
                            "last_accessed_at": "never",
                            "pinned":           False,
                        },
                    )],
                )
                ids.append(mem_id)
            except Exception as e:
                log.warning(f"Failed to upsert fact {mem_id!r}: {e}")

        return ids

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str, limit: int = 5) -> list[dict]:
        """
        Embed query and return top-k memories from Qdrant filtered by user_id.
        Does NOT update access_count — AikoMemorize.search() handles that.
        """
        try:
            vector  = self._embed(query)
            results = self._qdrant.search(
                collection_name=QDRANT_COLLECTION,
                query_vector=vector,
                limit=limit,
                query_filter=Filter(must=[
                    FieldCondition(key="user_id", match=MatchValue(value=user_id))
                ]),
                with_payload=True,
            )
            return [
                {"id": str(r.id), **r.payload}
                for r in results
            ]
        except Exception as e:
            log.warning(f"Search failed: {e}")
            return []

    def get_all(self, user_id: str) -> list[dict]:
        """Scroll all memories for a user. Returns full payload list."""
        results, offset = [], None
        try:
            while True:
                batch, offset = self._qdrant.scroll(
                    collection_name=QDRANT_COLLECTION,
                    scroll_filter=Filter(must=[
                        FieldCondition(key="user_id", match=MatchValue(value=user_id))
                    ]),
                    limit=100,
                    offset=offset,
                    with_payload=True,
                )
                results.extend({"id": str(p.id), **p.payload} for p in batch)
                if offset is None:
                    break
        except Exception as e:
            log.warning(f"get_all scroll failed: {e}")
        return results

    # ── delete ────────────────────────────────────────────────────────────────

    def delete(self, memory_id: str) -> None:
        """Delete a single memory point from Qdrant by ID."""
        self._qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=[memory_id],
        )

    def delete_all(self, user_id: str) -> None:
        """Delete all memories for a user."""
        self._qdrant.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]),
        )


# ── memorize ──────────────────────────────────────────────────────────────────

class AikoMemorize:
    """
    Persistent memory with Ebbinghaus decay lifecycle and nightly dream() pass.

    Uses a custom _MemoryBackend (Ollama extraction + fastembed + Qdrant)
    instead of mem0. Public API and all lifecycle behaviour are unchanged.

    Boot sequence (called by wakeup.py in order):
        memorize = AikoMemorize()   # connects Qdrant + loads fastembed
        memorize.cleanup()          # prune decayed memories on startup

    Access tracking:
        Every search() call updates Qdrant payload fields (access_count,
        last_accessed_at) so the decay formula has fresh data.

    Pinned memories:
        Created via pin() — the pinned=True Qdrant payload flag makes them
        immune to cleanup(), dream prune, and dream merge (as the loser).

    Dream pass (call nightly at 00:00):
        1. Boost salient memories' access_count so they survive decay.
        2. Merge near-duplicate vectors — keeps higher-access copy.
           Pinned memories are never deleted as a merge loser.
        3. Prune decayed memories via cleanup().
           Pinned memories are skipped entirely.

    Cleanup:
        Also available standalone — deletes memories below decay threshold,
        with grace period protection for newly created entries.
        Pinned memories are always kept regardless of score.
    """

    def __init__(self, silent: bool = False) -> None:
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))

        if not silent:
            log.info("Connecting to Qdrant and loading memory backend...")

        self._mem = _MemoryBackend(
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("MEM0_MODEL") or os.getenv("OLLAMA_MODEL"),
            fastembed_cache=os.getenv("FASTEMBED_CACHE_PATH"),
        )
        # Keep a direct Qdrant handle for payload operations (access tracking,
        # pinning, vector retrieval) that bypass the backend abstraction.
        self._qdrant = self._mem._qdrant

        if not silent:
            log.info("Ready.")

    # ── write ─────────────────────────────────────────────────────────────────

    def add(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store a conversation turn (or batch) into long-term memory.

        Runs synchronously — LLM extraction completes before returning.
        Callers that need non-blocking writes should enqueue via their own
        worker (e.g. think.py's _mem_write_loop).

        Returns True on success, False on failure so callers can log/alert.
        """
        try:
            t    = time.perf_counter()
            ids  = self._mem.add(messages, user_id=user_id)
            elapsed = time.perf_counter() - t
            if ids:
                log.info(f"Saved {len(ids)} memories in {elapsed:.2f}s")
            else:
                log.debug(f"No facts extracted ({elapsed:.2f}s) — nothing saved.")
            return True
        except Exception as e:
            log.error(f"Save failed: {e}")
            return False

    def pin(self, messages: list[dict], user_id: str = USER_ID) -> bool:
        """
        Store messages and immediately mark all resulting memories as pinned.

        Pinned memories are permanently immune to:
          - decay cleanup (cleanup() skips them regardless of score)
          - dream pruning (dream() prune stage skips them)
          - dream merging (never chosen as the loser in a duplicate collapse)

        Uses before/after snapshot of get_all() to identify new memory IDs,
        then sets pinned=True in their Qdrant payload.

        Returns True on success, False on any failure (check logs).
        """
        try:
            before  = {str(m["id"]) for m in self.get_all(user_id=user_id)}
            ok      = self.add(messages, user_id=user_id)
            if not ok:
                return False
            after   = {str(m["id"]) for m in self.get_all(user_id=user_id)}
            pin_ids = list(after - before)

            if not pin_ids:
                query = "\n".join(
                    (m.get("content") or "").strip()
                    for m in messages
                    if (m.get("content") or "").strip()
                )
                pin_ids = [
                    str(m.get("id"))
                    for m in self.search(query, user_id=user_id, limit=3)
                    if m.get("id")
                ]

            if not pin_ids:
                log.warning("pin(): add succeeded but no memory IDs were found to pin.")
                return False

            self._qdrant.set_payload(
                collection_name=QDRANT_COLLECTION,
                payload={"pinned": True},
                points=pin_ids,
            )
            log.info(f"Pinned {len(pin_ids)} memories: {pin_ids}")
            return True
        except Exception as e:
            log.error(f"Pin failed: {e}")
            return False

    # ── read ──────────────────────────────────────────────────────────────────

    def search(self, query: str, user_id: str = USER_ID, limit: int = 5) -> list[dict]:
        """
        Retrieve top-k memories relevant to the current query.

        Side-effect: increments access_count and updates last_accessed_at
        in Qdrant payload for each returned memory, feeding decay scoring.
        """
        results = self._mem.search(query, user_id=user_id, limit=limit)

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
                            "access_count":     min(current_count + 1, 255),
                            "last_accessed_at": now,
                        },
                        points=[mem_id],
                    )
                except Exception as e:
                    log.warning(f"Access tracking failed for {mem_id}: {e}")

        return results

    def format_for_context(self, memories: list[dict]) -> Optional[str]:
        """
        Format retrieved memories into a compact string for injection
        into the conversation context. Returns None if nothing to inject.
        """
        if not memories:
            return None

        now   = datetime.now(timezone.utc)
        lines = [
            "<memory_context>",
            "The following are background facts about this person, with how long ago they were recorded.",
            "Use them silently to inform your response. Never repeat, quote, or reference this block directly.",
            "",
        ]
        for m in memories:
            text       = m.get("memory") or m.get("text") or str(m)
            created_at = m.get("created_at")
            if created_at:
                try:
                    ts    = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    delta = now - ts
                    days  = delta.days
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

    # ── dream pass ────────────────────────────────────────────────────────────

    def dream(
        self,
        user_id:   str   = USER_ID,
        dry_run:   bool  = False,
        threshold: float = DREAM_MERGE_THRESHOLD,
    ) -> dict:
        """
        Nightly memory consolidation pass. No new vectors are written.

        Stages (in order):
          1. Boost  — salient memories get +DREAM_BOOST_AMOUNT access_count.
          2. Merge  — near-duplicate pairs (cosine >= threshold) are collapsed;
                      higher access_count copy survives, other is deleted.
                      Pinned memories are never chosen as the loser.
          3. Prune  — standard decay cleanup runs last, after boosts are applied,
                      so newly protected memories aren't swept.
                      Pinned memories are always kept.

        Args:
            dry_run:   Report actions without writing or deleting anything.
            threshold: Cosine similarity cutoff for duplicate detection.

        Returns dict: {boosted, merged, pruned, duration_s}
        """
        t_start = time.perf_counter()
        log.info(f"{'(dry-run) ' if dry_run else ''}Starting consolidation pass...")

        all_mems = self.get_all(user_id=user_id)
        if not all_mems:
            log.info("No memories found — nothing to do.")
            return {"boosted": 0, "merged": 0, "pruned": 0, "duration_s": 0.0}

        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)

        boosted      = self._dream_boost(all_mems, payload_map, dry_run=dry_run)
        merged       = self._dream_merge(mem_ids, threshold=threshold, dry_run=dry_run)
        prune_result = self.cleanup(user_id=user_id, dry_run=dry_run)
        pruned       = prune_result.get("deleted", 0)

        duration = round(time.perf_counter() - t_start, 2)
        log.info(
            f"{'(dry-run) ' if dry_run else ''}"
            f"Done — boosted={boosted}, merged={merged}, pruned={pruned}, "
            f"duration={duration}s"
        )
        return {"boosted": boosted, "merged": merged, "pruned": pruned, "duration_s": duration}

    def _dream_boost(
        self,
        all_mems:    list[dict],
        payload_map: dict,
        dry_run:     bool = False,
    ) -> int:
        """
        Increment access_count on memories that match salience heuristics.

        Salience criteria (any one triggers boost):
          - Text contains a keyword from _SALIENCE_KEYWORDS
          - access_count >= 3 (user has repeatedly surfaced this memory)
          - created_at within the last 7 days (recency grace boost)

        Pinned memories pass through the boost unchanged — they don't need it.

        Returns count of memories boosted.
        """
        now     = datetime.now(timezone.utc)
        boosted = 0

        for m in all_mems:
            mem_id = str(m.get("id", ""))
            if not mem_id:
                continue

            if self._is_pinned(mem_id):
                continue

            text = (m.get("memory") or "").lower()
            ac, _la = payload_map.get(mem_id, (0, "never"))

            is_recent  = False
            created_at = m.get("created_at", "")
            if created_at:
                try:
                    ts        = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    is_recent = (now - ts).days <= 7
                except Exception:
                    pass

            is_salient = (
                any(kw in text for kw in _SALIENCE_KEYWORDS)
                or ac >= 3
                or is_recent
            )

            if not is_salient:
                continue

            if not dry_run:
                try:
                    self._qdrant.set_payload(
                        collection_name=QDRANT_COLLECTION,
                        payload={"access_count": min(ac + DREAM_BOOST_AMOUNT, 255)},
                        points=[mem_id],
                    )
                except Exception as e:
                    log.warning(f"Boost failed for {mem_id}: {e}")
                    continue

            boosted += 1

        if boosted:
            log.info(f"{'(dry-run) ' if dry_run else ''}Boosted {boosted} memories.")
        return boosted

    def _dream_merge(
        self,
        mem_ids:   list[str],
        threshold: float = DREAM_MERGE_THRESHOLD,
        dry_run:   bool  = False,
    ) -> int:
        """
        Detect and collapse near-duplicate memory vectors.

        For each memory, searches Qdrant for neighbors above the cosine
        threshold. When a duplicate pair is found, the lower access_count
        copy is deleted. Tracks already-deleted IDs to avoid double-deletes
        in the same pass.

        Pinned memories are skipped as query origins and are never chosen as
        the loser in _resolve_duplicate() — a pinned memory always survives.

        Returns count of memories deleted as duplicates.
        """
        deleted_ids: set[str] = set()
        merged = 0

        for mem_id in mem_ids:
            if mem_id in deleted_ids:
                continue

            if self._is_pinned(mem_id):
                continue

            vector = self._get_vector(mem_id)
            if not vector:
                continue

            try:
                neighbors = self._qdrant.search(
                    collection_name=QDRANT_COLLECTION,
                    query_vector=vector,
                    limit=4,
                    score_threshold=threshold,
                    with_payload=True,
                )
            except Exception as e:
                log.warning(f"Similarity search failed for {mem_id}: {e}")
                continue

            for neighbor in neighbors:
                neighbor_id = str(neighbor.id)
                if neighbor_id == mem_id:
                    continue
                if neighbor_id in deleted_ids:
                    continue

                n_merged = self._resolve_duplicate(
                    mem_id, neighbor_id, neighbor.score, dry_run=dry_run
                )
                if n_merged:
                    deleted_ids.add(neighbor_id)
                    merged += 1

        if merged:
            log.info(f"{'(dry-run) ' if dry_run else ''}Merged {merged} duplicate memories.")
        return merged

    def _resolve_duplicate(
        self,
        id_a:    str,
        id_b:    str,
        score:   float,
        dry_run: bool = False,
    ) -> bool:
        """
        Compare two near-duplicate memories and delete the weaker one.

        Keeps the copy with higher access_count. On a tie, keeps id_a
        (the query origin) and deletes id_b.

        If either memory is pinned, the merge is aborted — a pinned memory
        is never deleted regardless of access_count comparison.

        Returns True if a deletion occurred (or would occur in dry_run).
        """
        if self._is_pinned(id_a) or self._is_pinned(id_b):
            log.info(f"Skipping merge: one or both of ({id_a}, {id_b}) is pinned.")
            return False

        payload_map = self._batch_get_payloads([id_a, id_b])
        ac_a, _ = payload_map.get(id_a, (0, "never"))
        ac_b, _ = payload_map.get(id_b, (0, "never"))
        loser   = id_b if ac_a >= ac_b else id_a

        if dry_run:
            log.info(
                f"(dry-run) Would merge: score={score:.3f} "
                f"ac_a={ac_a} ac_b={ac_b} → delete {loser}"
            )
            return True

        try:
            self._mem.delete(memory_id=loser)
            log.info(
                f"Merged duplicate (score={score:.3f}, "
                f"ac_a={ac_a}, ac_b={ac_b}) → deleted {loser}"
            )
            return True
        except Exception as e:
            log.warning(f"Merge delete failed for {loser}: {e}")
            return False

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def cleanup(
        self,
        user_id:   str   = USER_ID,
        threshold: float = CLEANUP_THRESHOLD,
        dry_run:   bool  = False,
    ) -> dict:
        """
        Prune decayed memories below threshold score.

        Fetches all memories, batch-retrieves Qdrant payloads (single round-trip),
        evaluates decay score via should_cleanup(), and deletes candidates directly
        via Qdrant to keep vector store in sync.

        Grace period (14 days) protects newly created memories from deletion
        even if they score below threshold.

        Pinned memories are unconditionally kept — the pinned flag overrides
        all decay scoring.

        Args:
            threshold: Override decay threshold (default: CLEANUP_THRESHOLD = 0.05).
            dry_run:   If True, report candidates without deleting.

        Returns dict with counts: deleted, kept, failed, candidates (dry_run only).
        """
        all_mems = self.get_all(user_id=user_id)
        if not all_mems:
            return {"deleted": 0, "kept": 0, "failed": 0}

        mem_ids     = [str(m.get("id", "")) for m in all_mems if m.get("id")]
        payload_map = self._batch_get_payloads(mem_ids)

        candidates = []
        kept       = 0

        for m in all_mems:
            mem_id     = str(m.get("id", ""))
            ac, la     = payload_map.get(mem_id, (0, "never"))
            created_at = m.get("created_at", "")

            if self._is_pinned(mem_id):
                kept += 1
                continue

            if should_cleanup(ac, la, created_at):
                w = compute_weighted_score(ac, la)
                candidates.append({
                    "id":               mem_id,
                    "memory":           m.get("memory", "")[:120],
                    "access_count":     ac,
                    "weighted_score":   round(w, 4),
                    "last_accessed_at": la,
                })
            else:
                kept += 1

        candidates.sort(key=lambda x: x["weighted_score"])

        if dry_run:
            log.info(f"Dry run: {len(candidates)} candidates for deletion, {kept} kept.")
            return {"deleted": 0, "kept": kept, "failed": 0, "candidates": candidates}

        deleted = []
        failed  = []
        for c in candidates:
            try:
                self._mem.delete(memory_id=c["id"])
                deleted.append(c["id"])
            except Exception as e:
                failed.append({"id": c["id"], "error": str(e)})

        log.info(f"Cleanup: deleted={len(deleted)}, kept={kept}, failed={len(failed)}")
        return {"deleted": len(deleted), "kept": kept, "failed": len(failed)}

    # ── debug ─────────────────────────────────────────────────────────────────

    def get_all(self, user_id: str = USER_ID) -> list[dict]:
        """Return all stored memories for a user (for debugging / dream pass)."""
        return self._mem.get_all(user_id=user_id)

    def clear(self, user_id: str = USER_ID) -> None:
        """Wipe all memories for a user. Use carefully."""
        self._mem.delete_all(user_id=user_id)
        log.info(f"Cleared all memories for user '{user_id}'.")

    # ── internal ──────────────────────────────────────────────────────────────

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
            log.warning(f"Batch payload fetch failed: {e}")
            return {}

    def _get_vector(self, mem_id: str) -> list[float]:
        """
        Retrieve the raw embedding vector for a single memory from Qdrant.
        Used by _dream_merge() to run similarity searches.
        Returns empty list on failure — callers should skip on empty.
        """
        try:
            pts = self._qdrant.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[mem_id],
                with_vectors=True,
            )
            if pts and pts[0].vector:
                return pts[0].vector
        except Exception as e:
            log.warning(f"Vector fetch failed for {mem_id}: {e}")
        return []

    def _is_pinned(self, mem_id: str) -> bool:
        """
        Return True if the Qdrant payload has pinned=True for this memory.

        Used as a guard in cleanup(), _dream_merge(), and _resolve_duplicate()
        to make pinned memories permanently immune to all deletion paths.
        Defaults to False on any error — safe because a False miss at worst
        leaves a memory subject to normal decay, not silently deletes it.
        """
        try:
            pts = self._qdrant.retrieve(
                collection_name=QDRANT_COLLECTION,
                ids=[mem_id],
                with_payload=True,
            )
            return bool(pts and pts[0].payload.get("pinned", False))
        except Exception:
            return False
