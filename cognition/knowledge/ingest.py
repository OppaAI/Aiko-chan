"""Knowledge ingestion and source extraction."""
from __future__ import annotations

from cognition.memory.redaction import redact_text, redact_data

import importlib.util
import hashlib
import os
import re
import sqlite3
import threading
import uuid
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

from defusedxml import ElementTree as DET
import numpy as np

from cognition import reason
from cognition.memory.vecstore import insert_vector, user_scoped_vec_knn
from cognition.memory.memorize import extract_entities, entities_to_json
from system.log import get_logger
from system.userspace import current_user_id, user_workspace_root

from .schema import (
    connect,
    Embedder,
    KNOWLEDGE_CHUNK_CHARS,
    EMBED_DIMS,
    KNOWLEDGE_SUPERSEDE_ON_DEDUP,
    KNOWLEDGE_WORKSPACE_DIR,
    KNOWLEDGE_WRITE_DEDUP_THRESHOLD,
    KnowledgeSchema,
    now,
)
from .search import maybe_clear_knowledge_cache

log = get_logger(__name__)


class KnowledgeIngest:
    """Owns file extraction and learned-knowledge write path."""

    def __init__(self, schema: KnowledgeSchema | None = None, embedder: Embedder | None = None):
        self.schema = schema or KnowledgeSchema()
        self.embedder = embedder

    def extract_text_from_file(
        self, relative_path: str, *, user_id: str | None = None, max_chars: int = 200_000
    ) -> tuple[str, str]:
        return extract_text_from_file(relative_path, user_id=user_id, max_chars=max_chars)

    def ingest_file(
        self,
        relative_path: str,
        *,
        title: str | None = None,
        kind: str = "ingested",
        embedder: Embedder | None = None,
        user_id: str | None = None,
        image_caption: str | None = None,
    ) -> str | None:
        return ingest_file(
            relative_path,
            title=title,
            kind=kind,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
            image_caption=image_caption,
        )

    def ingest_image(self, relative_path: str, caption: str, **kwargs):
        kwargs.setdefault("embedder", self.embedder)
        kwargs.setdefault("user_id", None)
        kwargs.setdefault("schema", self.schema)
        return ingest_image(relative_path, caption, **kwargs)
    
    def ingest_text(
        self,
        title: str,
        text: str,
        *,
        source: str = "",
        kind: str = "ingested",
        embedder: Embedder | None = None,
        user_id: str | None = None,
    ) -> str | None:
        return ingest_text(
            title,
            text,
            source=source,
            kind=kind,
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
        )
    
    def ingest_workspace_knowledge_folder(
        self, *, embedder: Embedder | None = None, user_id: str | None = None
    ) -> list[str]:
        return ingest_workspace_knowledge_folder(
            embedder=embedder if embedder is not None else self.embedder,
            user_id=user_id,
            schema=self.schema,
        )

def _sanitize_text(text: str, max_chars: int = 200_000) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())[:max_chars]


def _read_plain_text(path: Path, max_chars: int) -> str:
    """Normalize incrementally so a large log is not loaded before clipping."""
    parts = []
    remaining = max(0, max_chars)
    previous_space = False
    with path.open(encoding="utf-8", errors="replace") as stream:
        while remaining > 0:
            block = stream.read(8192)
            if not block:
                break
            block = re.sub(r"\s+", " ", block)
            if not parts:
                block = block.lstrip()
            elif previous_space:
                block = block.lstrip(" ")
            block = block[:remaining]
            if block:
                parts.append(block)
                previous_space = block.endswith(" ")
                remaining -= len(block)
    return "".join(parts).rstrip()


def _safe_workspace_path(relative_path: str, user_id: str | None = None) -> Path:
    root = user_workspace_root(user_id).resolve()
    target = (root / relative_path).expanduser().resolve()
    if root not in target.parents and target != root:
        raise ValueError("path must stay inside the user workspace")
    return target


def _xml_text_from_zip(path: Path, members: Iterable[str], max_chars: int = 200_000) -> str:
    chunks: list[str] = []
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        for member in members:
            if accumulated >= max_chars:
                break
            try:
                info = zf.getinfo(member)
                remaining_budget = max_chars - accumulated
                if info.file_size > remaining_budget:
                    # Skip members that exceed remaining budget
                    continue
                data = zf.read(member)
            except KeyError:
                continue
            root = DET.fromstring(data)
            for t in root.itertext():
                if t and t.strip():
                    text = t.strip()
                    chunks.append(text)
                    accumulated += len(text) + 1  # +1 for newline
                    if accumulated >= max_chars:
                        break
    return "\n".join(chunks)


def _xlsx_text(path: Path, max_chars: int = 200_000) -> str:
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        shared: list[str] = []
        try:
            info = zf.getinfo("xl/sharedStrings.xml")
            if info.file_size <= max_chars:
                root = DET.fromstring(zf.read("xl/sharedStrings.xml"))
                shared = [" ".join(t.strip() for t in si.itertext() if t and t.strip()) for si in root]
        except KeyError:
            log.debug("knowledge: xlsx has no shared strings")
        out: list[str] = []
        for name in sorted(n for n in zf.namelist() if n.startswith("xl/worksheets/") and n.endswith(".xml")):
            if accumulated >= max_chars:
                break
            try:
                info = zf.getinfo(name)
                remaining_budget = max_chars - accumulated
                if info.file_size > remaining_budget:
                    continue
                root = DET.fromstring(zf.read(name))
            except KeyError:
                continue
            for c in root.iter():
                if not c.tag.endswith("}c"):
                    continue
                cell_type = c.attrib.get("t")
                value = None
                for child in c:
                    if child.tag.endswith("}v"):
                        value = child.text
                        break
                if value is None:
                    continue
                if cell_type == "s":
                    try:
                        value = shared[int(value)]
                    except Exception:
                        log.warning("knowledge: failed to decode shared string")
                value_str = str(value)
                out.append(value_str)
                accumulated += len(value_str) + 1  # +1 for newline
                if accumulated >= max_chars:
                    break
    return "\n".join(out)


def _epub_text(path: Path, max_chars: int = 200_000) -> str:
    texts: list[str] = []
    accumulated = 0
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if accumulated >= max_chars:
                break
            if name.casefold().endswith((".xhtml", ".html", ".htm")):
                try:
                    info = zf.getinfo(name)
                    remaining_budget = max_chars - accumulated
                    if info.file_size > remaining_budget:
                        continue
                    raw = zf.read(name).decode("utf-8", errors="replace")
                except KeyError:
                    continue
                raw = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", raw, flags=re.I)
                cleaned = re.sub(r"<[^>]+>", " ", raw)
                texts.append(cleaned)
                accumulated += len(cleaned) + 1  # +1 for newline
                if accumulated >= max_chars:
                    break
    return "\n".join(texts)


def _extract_with_markitdown(path: Path, max_chars: int = 200_000) -> str | None:
    """Convert a document to Markdown via MarkItDown if installed.

    Returns text on success, or None when MarkItDown is unavailable or
    conversion fails so the caller can fall back to built-in extractors.
    """
    if importlib.util.find_spec("markitdown") is None:
        return None
    try:
        from markitdown import MarkItDown  # type: ignore
        result = MarkItDown().convert(str(path))
        text = (getattr(result, "text_content", None) or "").strip()
    except Exception:
        return None
    if not text:
        return None
    return text[:max_chars]


def extract_text_from_file(relative_path: str, *, user_id: str | None = None, max_chars: int = 200_000) -> tuple[str, str]:
    """Extract text from a workspace document for learned-knowledge ingest.

    Document formats (html, pdf, docx, xlsx, epub, pptx, ...) are converted
    via MarkItDown when installed, falling back to built-in extractors.
    Plain text/Markdown/config files are read directly. Returns
    (text, source_path). Raises ValueError with a user-facing reason when the
    file cannot be read/extracted.
    """
    path = _safe_workspace_path(relative_path, user_id)
    if not path.is_file():
        raise ValueError(f"workspace file not found: {relative_path}")
    suffix = path.suffix.casefold()
    source = str(path.relative_to(user_workspace_root(user_id).resolve()))
    if max_chars <= 0:
        return "", source

    # Plain text / code / config formats: read directly.
    if suffix in {".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".log", ".py", ".js", ".ts", ".tex", ".latex", ".rtf"}:
        return _read_plain_text(path, max_chars), source

    # Document formats: prefer MarkItDown, fall back to built-in extractors.
    md_text = _extract_with_markitdown(path, max_chars)
    if md_text is not None:
        return md_text, source

    if suffix in {".html", ".htm"}:
        try:
            import trafilatura  # type: ignore
            raw = trafilatura.extract(path.read_text(encoding="utf-8", errors="replace"), include_links=False, include_tables=False)
        except Exception:
            raw = None
        if not raw:
            raw = path.read_text(encoding="utf-8", errors="replace")
        return _sanitize_text(raw, max_chars), source
    if suffix == ".docx":
        try:
            text = _xml_text_from_zip(path, ["word/document.xml"], max_chars)
            return _sanitize_text(text, max_chars), source
        except Exception as exc:
            raise ValueError(f"Failed to extract .docx file: {exc}") from exc
    if suffix == ".xlsx":
        try:
            return _sanitize_text(_xlsx_text(path, max_chars), max_chars), source
        except Exception as exc:
            raise ValueError(f"Failed to extract .xlsx file: {exc}") from exc
    if suffix == ".epub":
        try:
            return _sanitize_text(_epub_text(path, max_chars), max_chars), source
        except Exception as exc:
            raise ValueError(f"Failed to extract .epub file: {exc}") from exc
    if suffix == ".pdf":
        reader_cls = None
        try:
            from pypdf import PdfReader  # type: ignore
            reader_cls = PdfReader
        except Exception:
            try:
                from PyPDF2 import PdfReader  # type: ignore
                reader_cls = PdfReader
            except Exception as exc:
                raise ValueError("PDF ingest needs pypdf or PyPDF2 installed") from exc
        try:
            reader = reader_cls(str(path))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
                if sum(len(p) for p in pages) >= max_chars:
                    break
            return _sanitize_text("\n\n".join(pages), max_chars), source
        except Exception as exc:
            raise ValueError(f"Failed to read or extract PDF file: {exc}") from exc
    raise ValueError(f"unsupported knowledge file type: {suffix or 'no extension'}")


def _file_stamp(path: Path) -> tuple:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _file_fingerprint(path: Path) -> tuple[str, tuple]:
    before = _file_stamp(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(65536):
            digest.update(block)
    after = _file_stamp(path)
    if before != after:
        raise ValueError("knowledge source changed while hashing; retry next scan")
    return digest.hexdigest(), after


# Scan-only optimization. Actual ingestion still verifies bytes independently
# before publication. A restart deliberately starts with an empty cache.
_SCAN_HASH_CACHE: OrderedDict[Path, tuple[str, tuple]] = OrderedDict()
_SCAN_HASH_LOCK = threading.Lock()
_SCAN_HASH_LIMIT = 1024


def _scan_fingerprint(path: Path) -> tuple[str, tuple]:
    stamp = _file_stamp(path)
    with _SCAN_HASH_LOCK:
        cached = _SCAN_HASH_CACHE.get(path)
        if cached is not None and cached[1] == stamp:
            _SCAN_HASH_CACHE.move_to_end(path)
            return cached
    result = _file_fingerprint(path)
    with _SCAN_HASH_LOCK:
        _SCAN_HASH_CACHE[path] = result
        _SCAN_HASH_CACHE.move_to_end(path)
        while len(_SCAN_HASH_CACHE) > _SCAN_HASH_LIMIT:
            _SCAN_HASH_CACHE.popitem(last=False)
    return result


def _library_files(folder: Path):
    """Walk incrementally, excluding hidden directories and directory links."""
    for parent, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith('.')
                         and not (Path(parent) / name).is_symlink())
        for name in sorted(files):
            if not name.startswith('.'):
                path = Path(parent) / name
                if path.is_file():
                    yield path


def ingest_file(
    relative_path: str,
    *,
    title: str | None = None,
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
    schema: KnowledgeSchema | None = None,
    image_caption: str | None = None,
) -> str | None:
    """Canonical implementation that
    :class:`KnowledgeIngest.ingest_file` delegates to (not a shim — this is the live code path). Kept for the historical
    import path. Extract a workspace file and store it in learned knowledge RAG."""
    uid = user_id or current_user_id()
    if image_caption and Path(relative_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
        return ingest_image(relative_path, image_caption, title=title, kind=kind,
                            embedder=embedder, user_id=uid, schema=schema)
    try:
        path = _safe_workspace_path(relative_path, uid)
        digest, _ = _file_fingerprint(path)
        text, source = extract_text_from_file(relative_path, user_id=uid)
        after_digest, stamp = _file_fingerprint(path)
        if after_digest != digest:
            raise ValueError("knowledge source changed during extraction; retry next scan")
    except (ValueError, OSError) as exc:
        log.warning("Failed to extract knowledge file %s: %s", relative_path, exc)
        return None
    return ingest_text(title or Path(relative_path).stem.replace("_", " ").title(), text, source=redact_text(source), kind=kind, embedder=embedder, user_id=uid, schema=schema, _file_version=(digest, path, stamp))


def ingest_image(
    relative_path: str,
    caption: str,
    *,
    title: str | None = None,
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
    schema: KnowledgeSchema | None = None,
) -> str | None:
    """Index a supplied image caption and keep a workspace-relative asset link.

    This deliberately uses Aiko's existing text embedding model; it does not
    claim visual understanding. The image itself remains in the user workspace.
    """
    uid = user_id or current_user_id()
    try:
        path = _safe_workspace_path(relative_path, uid)
        if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            raise ValueError("unsupported image type")
        stamp = _file_stamp(path)
        if stamp[2] > 20 * 1024 * 1024:
            raise ValueError("image exceeds 20 MiB limit")
        digest, checked = _file_fingerprint(path)
        with path.open("rb") as stream:
            header = stream.read(16)
        valid = (header.startswith(b"\x89PNG\r\n\x1a\n") or header.startswith(b"\xff\xd8\xff")
                 or header.startswith(b"GIF87a") or header.startswith(b"GIF89a")
                 or (header[:4] == b"RIFF" and header[8:12] == b"WEBP"))
        if not valid or checked != stamp:
            raise ValueError("invalid image or image changed while reading")
    except (ValueError, OSError) as exc:
        log.warning("Failed to register knowledge image %s: %s", relative_path, exc)
        return None
    safe_caption = redact_text(caption or "").strip()[:5000]
    if not safe_caption:
        return None
    suffix = path.suffix.lower()
    media = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
             ".webp": "image/webp", ".gif": "image/gif"}[suffix]
    # Caption participates in the source fingerprint, so recaptioning an
    # unchanged file publishes a new active record as well.
    version = hashlib.sha256((digest + "\0" + safe_caption).encode()).hexdigest()
    return ingest_text(title or path.stem.replace("_", " ").title(), safe_caption,
                       source=relative_path, kind=kind, embedder=embedder, user_id=uid,
                       schema=schema, _file_version=(version, path, stamp),
                       _asset=(relative_path, media, digest, safe_caption))

def ingest_text(
    title: str,
    text: str,
    *,
    source: str = "",
    kind: str = "ingested",
    embedder: Embedder | None = None,
    user_id: str | None = None,
    schema: KnowledgeSchema | None = None,
    _file_version: tuple[str, Path, tuple] | None = None,
    _asset: tuple[str, str, str, str] | None = None,
) -> str | None:
    """Canonical implementation that
    :class:`KnowledgeIngest.ingest_text` delegates to (not a shim — this is the live code path). Kept for the historical
    import path. Chunk, embed, and persist durable learned knowledge."""
    clean = _sanitize_text(redact_text(text))
    if not clean:
        return None
    title = redact_text(title)
    source = redact_text(source or "")
    uid = user_id or current_user_id()
    doc_id = str(uuid.uuid4())
    created_at = now()
    chunks = reason.chunk_text(clean, KNOWLEDGE_CHUNK_CHARS) or [clean]
    # Do network/embedding work before taking SQLite's writer lock.
    vectors = []
    if embedder is not None:
        batch = reason.embed_batch_or_none(embedder, chunks)
        vectors = list(batch) if batch is not None and len(batch) == len(chunks) else []
        for index, value in enumerate(vectors):
            try:
                vector = np.asarray(value, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if vector.shape != (EMBED_DIMS,) or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 1e-12:
                    raise ValueError("invalid chunk embedding")
                vectors[index] = vector
            except (TypeError, ValueError, OverflowError):
                vectors[index] = None
    store = schema or KnowledgeSchema()
    conn = store.connect(uid)
    try:
        if _file_version is not None:
            conn.execute("BEGIN IMMEDIATE")
            digest, path, stamp = _file_version
            if _file_stamp(path) != stamp:
                raise ValueError("knowledge source changed during embedding; retry next scan")
            prior = conn.execute(
                "SELECT source_hash FROM knowledge_file_state WHERE user_id=? AND source=?",
                (uid, source),
            ).fetchone()
            if prior is not None and prior["source_hash"] == digest:
                conn.rollback()
                return None
        conn.execute(
            "INSERT INTO learned_docs(id,user_id,title,source,kind,created_at) VALUES(?,?,?,?,?,?)",
            (doc_id, uid, (title or "Untitled knowledge")[:200], source[:500], kind[:50], created_at),
        )
        if _asset is not None:
            conn.execute(
                "INSERT INTO knowledge_assets(doc_id,user_id,relative_path,media_type,file_hash,caption) "
                "VALUES(?,?,?,?,?,?)",
                (doc_id, uid, *_asset),
            )
        written_chunks = 0
        for index, chunk in enumerate(chunks):  # always loop chunks
            ents_json = entities_to_json(extract_entities(chunk))
            vec = vectors[index] if index < len(vectors) else None

            supersedes_id = None
            # A file version owns its complete chunk set and provenance.
            # Global semantic dedup must not omit a changed file's chunks.
            if _file_version is None and KNOWLEDGE_WRITE_DEDUP_THRESHOLD > 0 and vec is not None:
                try:
                    neighbors = user_scoped_vec_knn(
                        conn,
                        vec_table="learned_chunks_vec",
                        owner_table="learned_chunks",
                        owner_alias="c",
                        vector=vec,
                        user_id=uid,
                        limit=1,
                    )
                    if neighbors:
                        dist = float(neighbors[0]["dist"])
                        sim = 1.0 - dist
                        if sim >= KNOWLEDGE_WRITE_DEDUP_THRESHOLD:
                            old_id = str(neighbors[0]["id"])
                            if KNOWLEDGE_SUPERSEDE_ON_DEDUP and old_id:
                                row = conn.execute(
                                    "SELECT status FROM learned_chunks WHERE id = ? AND user_id = ?",
                                    (old_id, uid),
                                ).fetchone()
                                st = "active"
                                if row is not None:
                                    try:
                                        st = (row["status"] or "active")
                                    except Exception:
                                        st = "active"
                                if str(st).strip().lower() == "superseded":
                                    # Already replaced — skip insert (true dedup) or treat as no parent
                                    log.debug("knowledge dedup skip already-superseded sim=%.3f", sim)
                                    continue
                                supersedes_id = old_id
                                try:
                                    conn.execute(
                                        "UPDATE learned_chunks SET status = 'superseded' "
                                        "WHERE id = ? AND user_id = ? "
                                        "AND (status = 'active' OR status IS NULL)",
                                        (old_id, uid),
                                    )
                                except Exception as sup_exc:
                                    log.debug("knowledge supersede mark failed: %s", sup_exc)
                                log.debug("knowledge supersede chunk sim=%.3f old=%s", sim, old_id[:8])
                            else:
                                log.debug("knowledge dedup skip chunk sim=%.3f", sim)
                                continue
                except Exception as dedup_exc:
                    log.debug("knowledge dedup skipped: %s", dedup_exc)

            chunk_id = str(uuid.uuid4())
            conn.execute(
                "INSERT INTO learned_chunks(id,doc_id,user_id,chunk_index,text,created_at,entities,status,supersedes_id) VALUES(?,?,?,?,?,?,?,?,?)",
                (chunk_id, doc_id, uid, index, chunk, created_at, ents_json, "active", supersedes_id),
            )
            written_chunks += 1
            if vec is not None:
                insert_vector(conn, "learned_chunks_vec", chunk_id, vec)

        if written_chunks == 0:
            # No chunks were written, rollback the document insert
            conn.rollback()
            return None

        if _file_version is not None:
            # Preserve old document/chunk history. Only active recall changes,
            # in the same transaction that publishes the new source hash.
            conn.execute(
                "UPDATE learned_chunks SET status='superseded' WHERE user_id=? AND doc_id != ? "
                "AND (status='active' OR status IS NULL) AND (doc_id IN "
                "(SELECT doc_id FROM knowledge_file_state WHERE user_id=? AND source=?) "
                "OR doc_id IN (SELECT id FROM learned_docs WHERE user_id=? AND source=? AND ?))",
                (uid, doc_id, uid, source, uid, source, len(source) <= 500),
            )
            if _file_stamp(_file_version[1]) != _file_version[2]:
                raise ValueError("knowledge source changed during indexing; retry next scan")
            conn.execute(
                "DELETE FROM learned_chunks_vec WHERE id IN (SELECT c.id FROM learned_chunks c "
                "JOIN learned_docs d ON d.id=c.doc_id WHERE c.user_id=? AND c.status='superseded' "
                "AND (d.source=? OR d.id IN "
                "(SELECT doc_id FROM knowledge_file_state WHERE user_id=? AND source=?)))",
                (uid, source, uid, source),
            )
            conn.execute(
                "INSERT INTO knowledge_file_state(user_id,source,source_hash,doc_id) VALUES(?,?,?,?) "
                "ON CONFLICT(user_id,source) DO UPDATE SET source_hash=excluded.source_hash, doc_id=excluded.doc_id",
                (uid, source, _file_version[0], doc_id),
            )
        conn.commit()
        maybe_clear_knowledge_cache()
        return doc_id
    except Exception as exc:
        conn.rollback()
        log.warning("Failed to ingest knowledge: %s", exc)
        return None
    finally:
        conn.close()



def _knowledge_sources(conn: sqlite3.Connection, uid: str) -> dict[str, str]:
    rows = conn.execute("SELECT source, source_hash FROM knowledge_file_state WHERE user_id=?", (uid,)).fetchall()
    return {str(row["source"]): str(row["source_hash"]) for row in rows}


def ingest_workspace_knowledge_folder(*, embedder: Embedder | None = None, user_id: str | None = None) -> list[str]:
    """Ingest new files dropped under <workspace>/knowledge into the KB DB.

    Unchanged bytes are skipped; changed files publish a complete new version
    atomically. Failed or concurrently modified sources leave prior recall intact.
    Legacy documents acquire fingerprints on their first successful refresh.
    """
    uid = user_id or current_user_id()
    root = user_workspace_root(uid).resolve()
    folder = _safe_workspace_path(KNOWLEDGE_WORKSPACE_DIR, uid)
    folder.mkdir(parents=True, exist_ok=True)
    conn = connect(uid)
    try:
        known = _knowledge_sources(conn, uid)
        assets = {row['source']: row['file_hash'] for row in conn.execute(
            "SELECT s.source,a.file_hash FROM knowledge_file_state s "
            "JOIN knowledge_assets a ON a.doc_id=s.doc_id AND a.user_id=s.user_id WHERE s.user_id=?",
            (uid,))}
    finally:
        conn.close()
    doc_ids: list[str] = []
    outcomes = []
    for path in _library_files(folder):
        rel = str(path.relative_to(root))
        # Images need a caption supplied by the user or an image-capable model.
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
            try:
                safe_path = _safe_workspace_path(rel, uid)
                if _file_stamp(safe_path)[2] > 20 * 1024 * 1024:
                    raise ValueError('image exceeds 20 MiB limit')
                digest, _ = _scan_fingerprint(safe_path)
                outcomes.append((rel, "ready" if assets.get(rel) == digest else "needs_caption"))
            except (OSError, ValueError):
                outcomes.append((rel, "failed"))
            continue
        try:
            path = _safe_workspace_path(rel, uid)
            rel = str(path.relative_to(root))
            digest, _ = _scan_fingerprint(path)
            if known.get(rel) == digest:
                outcomes.append((rel, "ready"))
                continue
            doc_id = ingest_file(rel, kind="workspace_drop", embedder=embedder, user_id=uid)
        except Exception as exc:
            log.warning("skipping workspace knowledge file %s: %s", rel, exc)
            outcomes.append((rel, "failed"))
            continue
        if doc_id:
            known[rel] = digest
            doc_ids.append(doc_id)
            log.info("ingested workspace knowledge file %s as %s", rel, doc_id)
            outcomes.append((rel, "ready"))
        else:
            # Another importer may already have accepted these same bytes.
            conn = connect(uid)
            try:
                current = _knowledge_sources(conn, uid)
            finally:
                conn.close()
            outcomes.append((rel, "ready" if current.get(rel) == digest else "failed"))
    try:
        from .status import record_ingestion_statuses
        record_ingestion_statuses(outcomes, user_id=uid)
    except Exception:
        log.exception("knowledge ingestion status recording failed")
    if embedder is not None:
        try:
            from .backfill import backfill_missing_vectors
            backfill_missing_vectors(embedder=embedder, user_id=uid)
        except Exception:
            log.exception('Knowledge vector repair unavailable; retry next scan')
    return doc_ids
