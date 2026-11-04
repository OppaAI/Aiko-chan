# Conversation routing and context

Aiko's normal chat and tool-task paths share `LLM_CTX_SIZE` (8192 in the checked-in
configuration). Keep it aligned with the actual llama-server context size. The
project version remains 0.2.0.

## Current request and history

`include_history=False` excludes earlier dialogue, not the current task. The
current user input is always included. Request-only style instructions are sent
to the model while the raw user text is retained for recall and stored history.
Failed model requests are displayed as errors and removed from pending chat
history; their errors are not submitted for memory extraction.

## Bounded normal-chat requests

Before sending a normal-chat request, the guard reserves the requested answer
budget plus ten percent of the window for template and estimation overhead.
It removes older dialogue turns first, then omits the largest named optional
retrieval blocks if needed. Core instructions and the current request are never
truncated. An omitted block explicitly says it is missing evidence.

If required context still needs more space, the answer budget can shrink, retaining
at least 128 tokens (or the caller's smaller requested answer). If that cannot
fit, Aiko reports a context error without making a doomed model request.

The guard uses a conservative ASCII/UTF-8 estimate, not the server's tokenizer.
Actual tokenization varies by model and language, so server rejection remains
possible; this does not imply an 8K model can understand arbitrarily long input.
Agentic requests retain their separate tool-aware fitting path.

## Context workers and identity

Each shared-pool submission copies the caller's context variables, including
user identity and reasoning mode. Reused worker threads do not retain the prior
request's identity. Nested submissions in the same bounded pool run inline to
avoid workers waiting on children queued behind other waiting workers. This
trades nested parallelism for predictable progress with four workers.

Routing embeddings use classification instructions. Memory retrieval obtains its
own cached vector using retrieval instructions, so classifier vectors are not
silently treated as retrieval vectors.

## Streaming and recovery

Both primary and no-system-role retry streams filter private thinking blocks.
Streams are closed on success and failure. If a stream fails after early sentence
audio begins, pending audio is cancelled before retry/fallback; audio already
played cannot be recalled. Recovery output still goes through normal response
gating. Thinking-only fallback metadata is request-local.

Inner-speech generation remains single-flight through persistence and snapshot
updates, not just through the model call. This prevents a second background
thought from overtaking an unfinished write from the previous one.

The inner-speech store also serializes connection close with database operations.
A late background write sees a closed connection safely rather than having
SQLite closed underneath an active extension operation.
