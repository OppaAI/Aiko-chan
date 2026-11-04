# Memory storage review for Jetson Orin Nano 8 GB

## Storage decision

Keep SQLite as Aiko's durable storage backend for the current single-device
deployment. WAL, short write transactions, FTS5, and sqlite-vec provide local
durability, lexical recall, and indexed similarity search without a separate
database server. Memory architecture and retrieval quality can improve while
retaining this backend.

The current audit found failures above the database engine: invalid/colliding
embedding caches, unavailable embeddings suppressing lexical recall, an
unimportable codebase index, and worker lifecycle races. A different database
would not correct those execution paths. Network embedding calls now happen
before the learned-knowledge writer transaction; embedding caches retain a
bounded number of vectors rather than multiplying capacity by batch size.

Revisit the backend when measurements demonstrate a need: several independent
machines writing concurrently, sustained writer contention after shortening
transactions, or a vector corpus whose measured latency/RAM cannot meet the
device's budget. PostgreSQL or a dedicated vector service would introduce
additional processes and deployment work, so a migration needs benchmarks and
a concrete workload that benefits from it. SQLite does not provide unrestricted
parallel writes; WAL allows readers alongside a writer.

## Existing memory features to compare

| Capability | Current Aiko implementation |
|---|---|
| Durable facts and hybrid vector/keyword/entity recall | `cognition/memory/memorize.py`, `schema.py`, `vecstore.py` |
| Episodic traces and asynchronous embeddings | `cognition/memory/episode.py` |
| Scenes, stable identity facts, and temporal reconciliation | `memorize.py`, `schema.py`, `learn.py` |
| Supersession and lineage | `cognition/memory/lineage.py`, `narrative.py` |
| Bounded working memory and daily journal | `cognition/memory/grasp.py` |
| Preference facts and retrieval diversity | `preference_store.py`, `diversity.py` |
| Learned documents, archive, and codebase retrieval | `cognition/knowledge/` |

These are implementation locations, not evidence that every feature behaves
correctly in every live-model scenario. Area 5 remains under review.

## Aidon comparison

Reviewed [OppaAI/Aidon](https://github.com/OppaAI/Aidon) at commit
[`5ecbbdec4765f4775032c3decb111ad6cf5ea4c3`](https://github.com/OppaAI/Aidon/tree/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3).
The comparison follows implementation and call sites, including places where
module docstrings are older than the current wiring. Aidon was reviewed as
source; its models and remote services were not started.

| Aidon feature | Aiko's current position | Decision for the 8 GB Jetson |
|---|---|---|
| File hashes and changed-document refresh in `src/memory/kb.py`; content-addressed document chunks in `docstore.py` | Codebase indexing already tracks changes. Learned-library scanning previously skipped any path imported once. | Implemented source fingerprints and transactional refresh for learned files. Unchanged files avoid extraction/embedding; old text remains as superseded history. |
| Credential redaction before embedding/storage in `src/memory/memory.py` and `redaction.py` | Encryption support exists, but the reviewed personal/episodic/knowledge write paths do not share an equivalent credential-redaction boundary. | High-priority follow-up. Use one tested helper across ingress paths and caches; preserve legitimate email addresses and ordinary personal facts. Define credential patterns carefully to avoid erasing normal technical text. |
| Structured failure lessons with recurrence, failure class, and repair/test fields in `failure_lessons.py` | Experiences, verification results, and engram relations exist, but no equivalent complete recurrence/repair lifecycle was found. | High-priority follow-up for self-coding. Extend the existing SQLite experience store; include only a few relevant lessons in an 8K prompt. Require actual repair evidence rather than equating a similar later success with proof that the same failure was fixed. |
| Image memory points with file references in `memory.py:record_image` and a shared text/image encoder in `embedder.py` | The reviewed memory API stores text/episodic traces; no equivalent image-embedding record API was found. Sensory integration remains Area 7. | Useful later, subject to measured RAM/latency with Aiko's normal model and services loaded. Keep image assets on disk and captions/references in SQLite; benchmark an encoder before adding another resident model. |
| Persistent parser-needed status in `kb.py` | Failed extraction is logged and the source remains available for retry. | Useful follow-up: persist ingestion status/error/retry metadata so unsupported documents are visible to the owner. No need for another database. |
| SQLite source of truth with rebuildable FAISS indexes in `stores.py` | SQLite + sqlite-vec already supplies durable text and vector retrieval. The first Area 5 fixes repair codebase vectors after outages. | Keep sqlite-vec now. Rebuildable indexes and explicit embedding-version metadata are useful design principles; a FAISS replacement needs a benchmark and a migration plan. |
| RecallBudget and model-summary helper modules | Aiko already fits assembled requests to the configured context and has working-memory/episode/scene summaries. Aidon's `RecallBudget` and `rolling_summary_model` have no production call sites in the reviewed source; the budget helper only trims point count. | Keep Aiko's active budget enforcement. Evaluate summary quality and combined retrieval budgets before adding another summarizer or unused policy object. |
| Extracted facts, entity links, recency/retention, and long-term promotion | Aiko already has corresponding features, plus temporal validity, supersession narratives, and lineage. | Review correctness and evaluate retrieval quality in Area 5; no additional memory layer is justified solely by a similar feature name. |

Implementation evidence:

- [Aidon file manifest and scan](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/kb.py)
  and [document hashes](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/docstore.py).
- [Aidon memory ingress and image points](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/memory.py)
  and [credential redaction](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/redaction.py).
- [Failure lesson records](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/failure_lessons.py),
  [record/repair call sites](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/agentic/run_finalization.py),
  and [planner context wiring](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/agentic/loop.py).
- [SQLite, FAISS, Qdrant, and backend selection](https://github.com/OppaAI/Aidon/blob/5ecbbdec4765f4775032c3decb111ad6cf5ea4c3/src/memory/stores.py).

### Implemented adaptation: learned-file freshness

Aiko now keeps `knowledge_file_state` in the existing knowledge database,
keyed by user and canonical workspace source. SHA-256 is computed with bounded
reads. A source version records its file stamp, and a file changing during
extraction/embedding/indexing is retried rather than published as the current
version. Existing databases acquire the metadata table automatically; old
imports receive their first fingerprint after a successful refresh.

A changed file is fully parsed/chunked/embedded before publication. SQLite then
publishes its new chunks, supersedes prior source chunks, removes their obsolete
vectors from active recall, and updates the fingerprint in one transaction.
Text/history remains available in the database. Unchanged bytes skip expensive
work, including repeated scans after a restart. File versions retain their own
complete chunk/provenance set rather than losing chunks to global semantic dedup.

A parser/write failure rolls back the version and its fingerprint. Concurrent
imports of identical bytes publish only one version. Source files removed from
the library do not automatically erase learned history; forgetting remains an
explicit memory/retention operation. These choices preserve Aiko's existing
history and retention behavior while adopting Aidon's freshness tracking.

Scan/watch coverage remains an Area 5 follow-up: the recursive scan handles
nested sources, but the current inotify watcher watches the root directory.
Nested edits need an explicit/periodic scan until recursive watch coverage is
implemented. File hashes are streamed, but avoiding repeated reads of unchanged
large files through validated source stamps is another device-optimization
candidate. No claim of immediate recursive watch delivery is made here.

Validation for this adaptation: **69 focused tests passed**; the complete
unit/integration and standalone conscience run passed **2,498 tests, 5 skipped,
and 34 subtests**, with 13 existing deprecation warnings.

## Validation boundary

Cloud checks use temporary SQLite/sqlite-vec databases, controlled embedders,
and real worker threads. They cover restartable caches, migration rollback,
lexical fallback during embedding outages, user scoping, index repair, and
worker cleanup. They do not measure Jetson peak RAM, real Harrier retrieval
quality, or the owner's existing database. No live user memories were migrated
or removed during this review.
