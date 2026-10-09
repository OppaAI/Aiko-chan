"""
cognition/__init__.py

Shared thread pool for concurrent context-fetch calls used across
cognition.think and agentic.agentic.

Two fetch groups use this pool:
  1. Memory + KB (cognition.think._fetch_memory_and_knowledge) — fired from
     route() only once intent has resolved to "agentic". Localchat/webchat
     recall sequentially inside chat() (memory → entity-link knowledge hop
     → conditional explicit knowledge search) and never touch this pool.
  2. (retired) Wiki / agentic-policy / skill / experience are no longer
     fetched at task start. The agentic loop pulls them explicitly with the
     retrieve_context and load_skill tools when it needs them; only a similar
     successful experience may be injected as ReAct guidance.

All of these are independent reads against separate backing stores
(memory.db, knowledge.db, wiki store, skills store, optional experience store,
persona/*.md files) keyed only on (user_input, embedder). No fetch
depends on another's output, so completion order never matters — callers
just wait for the ones they need and join the results into the prompt
afterward.

Sized for the busiest caller (agentic's post-intent context fetch)
plus headroom for an overlapping second request's smaller 2-way
pre-intent fetch.
"""
from concurrent.futures import ThreadPoolExecutor
import atexit

# ThreadPoolExecutor for concurrent memory/knowledge fetch.
# Jetson Orin Nano: 4 cores shared. Fetches are IO-bound (HTTP embeds,
# sqlite reads), so 4 workers overlap without CPU contention. Chat submits
# up to 4 (recall chain + system prompt + codebase + wiki); agentic submits
# 2 (memory + KB). Callers always join with a timeout, so a slow backend
# degrades to skipped context, never a hung turn.
CONTEXT_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ctx-fetch")
atexit.register(CONTEXT_POOL.shutdown, wait=False)
