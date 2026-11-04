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

The requested Aidon source has not yet been identified. No Aidon-specific
capabilities, performance results, or implementation changes are assumed.
Once its repository/documentation is supplied, compare it against the actual
Aiko implementation and tests, especially:

- Evidence/provenance and confidence handling for extracted memories.
- Conflict resolution, temporal updates, correction, and deletion behavior.
- Consolidation quality and preservation of important episodic details.
- Embedding/version migrations, repair after outages, and cache invalidation.
- Retrieval evaluation, bounded context selection, and measured device costs.

An addition should address a demonstrated Aiko gap and have a regression or
evaluation case. Avoid duplicating an existing memory layer under another name.

## Validation boundary

Cloud checks use temporary SQLite/sqlite-vec databases, controlled embedders,
and real worker threads. They cover restartable caches, migration rollback,
lexical fallback during embedding outages, user scoping, index repair, and
worker cleanup. They do not measure Jetson peak RAM, real Harrier retrieval
quality, or the owner's existing database. No live user memories were migrated
or removed during this review.
