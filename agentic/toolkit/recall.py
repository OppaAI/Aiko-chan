"""
agentic/toolkit/recall.py

On-demand recall for the agentic loop (tool-RAG counterpart to upfront injection).

run_agentic_chat injects memory/knowledge upfront; wiki, skill, and
experience blocks are never injected automatically — when the loop needs
them it calls retrieve_context instead of guessing from the system
prompt. Output is bounded to protect the 8GB context window.
"""
from __future__ import annotations

from agentic.registry import TOOLS, tool
from system.log import get_logger

log = get_logger(__name__)

_PER_STORE_LIMIT = 5
_TEXT_CHARS = 500
_TOTAL_CHARS = 2500
_VALID_STORES = ("memory", "knowledge", "experience", "wiki", "skill")


@tool(
    TOOLS["retrieve_context"] if "retrieve_context" in TOOLS else "retrieve_context",
    description="Pull long-term memory, learned knowledge, past task experience, wiki operating cards, or skill descriptions on demand mid-task. Use when upfront context was omitted for low relevance or you need specifics — do not guess from the system prompt. stores: comma-separated subset of memory,knowledge,experience,wiki,skill.",
    graph=True,
    react=True,
    domain="memory",
    always_on=True,
)
def retrieve_context(query: str = "", stores: str = "memory,knowledge", limit: int = 3) -> str:
    """Query personal memory / learned KB / past task runs mid-loop. Returns compact JSON."""
    from agentic.toolkit.common import json_block
    from system.userspace import current_user_id

    query = (query or "").strip()
    if not query:
        return json_block("retrieve_context", {"ok": False, "error": "query required"})
    try:
        limit = max(1, min(int(limit), _PER_STORE_LIMIT))
    except (TypeError, ValueError):
        limit = 3
    wanted = [s.strip().lower() for s in (stores or "").split(",")]
    wanted = [s for s in wanted if s in _VALID_STORES] or ["memory"]
    uid = current_user_id()
    out: dict = {"ok": True, "query": query, "stores": {}}
    try:
        if "memory" in wanted:
            from cognition.memory.memorize import AikoMemorize
            try:
                hits = AikoMemorize().search(query, user_id=uid, limit=limit) or []
            except Exception as e:
                log.warning("retrieve_context memory failed: %s", e)
                hits = []
            out["stores"]["memory"] = [
                {
                    "score": round(float(h.get("_recall_score", 0.0) or 0.0), 4),
                    "text": (h.get("memory") or h.get("text") or "")[:_TEXT_CHARS],
                }
                for h in hits
            ]
        if "knowledge" in wanted or "experience" in wanted or "wiki" in wanted or "skill" in wanted:
            try:
                from cognition.memory.vecstore import HarrierEmbedder
                emb = HarrierEmbedder()
            except Exception:
                emb = None
        else:
            emb = None
        if "knowledge" in wanted:
            from cognition.knowledge import search_knowledge
            try:
                hits = search_knowledge(query, limit=limit, embedder=emb, user_id=uid) or []
            except Exception as e:
                log.warning("retrieve_context knowledge failed: %s", e)
                hits = []
            out["stores"]["knowledge"] = [
                {
                    "title": h.get("title", ""),
                    "score": round(float(h.get("score", 0.0) or 0.0), 4),
                    "text": (h.get("text") or "")[:_TEXT_CHARS],
                }
                for h in hits
            ]
        if "experience" in wanted:
            from agentic.experience import search_experience
            try:
                hits = search_experience(query, limit=limit, embedder=emb, user_id=uid) or []
            except Exception as e:
                log.warning("retrieve_context experience failed: %s", e)
                hits = []
            out["stores"]["experience"] = [
                {
                    "goal": (h.get("goal") or "")[:200],
                    "outcome": h.get("outcome") or "",
                    "score": round(float(h.get("recall_score", h.get("score", 0.0)) or 0.0), 4),
                    "text": (h.get("record_text") or h.get("answer_excerpt") or "")[:_TEXT_CHARS],
                }
                for h in hits
            ]
        if "wiki" in wanted:
            from agentic.wiki import search_wiki
            try:
                items = search_wiki(query, limit=limit, embedder=emb) or []
            except Exception as e:
                log.warning("retrieve_context wiki failed: %s", e)
                items = []
            out["stores"]["wiki"] = [
                {
                    "id": getattr(it, "item_id", ""),
                    "kind": getattr(it, "kind", ""),
                    "title": getattr(it, "title", "") or "",
                    "text": (getattr(it, "text", "") or "")[:_TEXT_CHARS],
                }
                for it in items
            ]
        if "skill" in wanted:
            from agentic.skills import search_skillsets
            try:
                docs = search_skillsets(query, limit=limit, embedder=emb) or []
            except Exception as e:
                log.warning("retrieve_context skill failed: %s", e)
                docs = []
            out["stores"]["skill"] = [
                {
                    "skill_id": d.skill_id,
                    "name": d.name,
                    "summary": (d.summary or "")[:_TEXT_CHARS],
                    "hint": f"call load_skill(\"{d.name}\") for the full instructions",
                }
                for d in docs
            ]
    except Exception as e:
        log.warning("retrieve_context failed: %s", e)
        return json_block("retrieve_context", {"ok": False, "error": str(e), "query": query})
    blob = str(out)
    if len(blob) > _TOTAL_CHARS:
        # Trim longest store texts first; keep the envelope intact.
        for _store in out["stores"]:
            for _hit in out["stores"][_store]:
                if "text" in _hit and isinstance(_hit["text"], str):
                    _hit["text"] = _hit["text"][:200]
            if len(str(out)) <= _TOTAL_CHARS:
                break
    return json_block("retrieve_context", out)
