"""
PhysicsFlow RAG — Ingestion Pipeline (RAGIndexer).

Manages the full lifecycle of document indexing:
  - File ingestion (PDF, LAS, CSV, .pfproj, text/MD, JSON)
  - Deduplication by content hash
  - Dual write to VectorStore (dense) + SparseStore (BM25)
  - Background file watcher (optional, uses watchdog)
  - Audit log and chat message indexing
  - Index statistics and source management

Thread-safe: all public methods acquire a single write lock.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from .document_processor import DocumentChunk, DocumentProcessor
from .sparse_store import SparseStore
from .vector_store import VectorStore

log = logging.getLogger(__name__)

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent, FileCreatedEvent
    _HAS_WATCHDOG = True
except ImportError:
    _HAS_WATCHDOG = False


# ── File watcher ───────────────────────────────────────────────────────────────

if _HAS_WATCHDOG:
    class _IndexerEventHandler(FileSystemEventHandler):
        """Re-indexes files when they are created or modified."""
        _SUPPORTED_EXTS = {".pdf", ".las", ".csv", ".pfproj", ".txt", ".md", ".rst", ".json"}

        def __init__(self, indexer: "RAGIndexer"):
            self._indexer = indexer

        def on_created(self, event):
            if not event.is_directory and self._is_supported(event.src_path):
                log.info("File watcher: new file %s", event.src_path)
                self._indexer.index_file(event.src_path)

        def on_modified(self, event):
            if not event.is_directory and self._is_supported(event.src_path):
                log.info("File watcher: modified file %s", event.src_path)
                self._indexer.index_file(event.src_path)

        def _is_supported(self, path: str) -> bool:
            return Path(path).suffix.lower() in self._SUPPORTED_EXTS


# ── RAGIndexer ─────────────────────────────────────────────────────────────────

class RAGIndexer:
    """
    Unified ingestion pipeline: file → chunks → VectorStore + SparseStore.

    Handles:
    - Single file indexing (index_file)
    - Directory scanning (index_directory)
    - Audit log batch indexing (index_audit_entries)
    - Chat message indexing (index_chat_message)
    - Source deletion (delete_source)
    - Index clearing (clear)
    - Background file watching (start_watching / stop_watching)

    Usage:
        indexer = RAGIndexer()
        n = indexer.index_file("/path/to/report.pdf")
        indexer.start_watching("/project/documents/")
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        sparse_store: Optional[SparseStore] = None,
        processor: Optional[DocumentProcessor] = None,
    ):
        self._vs   = vector_store or VectorStore()
        self._ss   = sparse_store or SparseStore()
        self._proc = processor   or DocumentProcessor()
        self._lock = threading.Lock()

        # Indexed content hashes for deduplication (source_path → set of hashes)
        self._indexed_hashes: dict[str, set[str]] = {}

        # Watchdog observer
        self._observer: Optional["Observer"] = None

    # ── File indexing ──────────────────────────────────────────────────────────

    def index_file(self, path: str | Path, force: bool = False) -> int:
        """
        Index a single file into both stores.

        Args:
            path:  Path to the file to index.
            force: Re-index even if the file was previously indexed.

        Returns:
            Number of new chunks added (0 if already up to date).
        """
        p = Path(path)
        if not p.exists():
            log.warning("index_file: path not found — %s", p)
            return 0

        source_key = str(p)
        chunks = self._proc.process_file(p)
        if not chunks:
            log.debug("index_file: no chunks from %s", p.name)
            return 0

        return self._upsert_chunks(chunks, source_key, force)

    def index_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: Optional[set[str]] = None,
    ) -> int:
        """Scan a directory and index all supported files."""
        exts = extensions or {".pdf", ".las", ".csv", ".pfproj", ".txt", ".md", ".json"}
        d = Path(directory)
        if not d.is_dir():
            log.warning("index_directory: not a directory — %s", d)
            return 0

        pattern = "**/*" if recursive else "*"
        total = 0
        for p in d.glob(pattern):
            if p.suffix.lower() in exts and p.is_file():
                total += self.index_file(p)

        log.info("Indexed directory %s: %d new chunks", d, total)
        return total

    # ── Structured data indexing ───────────────────────────────────────────────

    def index_audit_entries(self, entries: list[dict]) -> int:
        """Index SQLite audit_log rows as searchable chunks."""
        if not entries:
            return 0
        chunks = self._proc.process_audit_entries(entries)
        return self._upsert_chunks(chunks, "audit_log", force=True)

    def index_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        project_name: str = "",
    ) -> int:
        """
        Index a chat turn for long-term memory retrieval.

        Args:
            session_id:   Chat session identifier.
            role:         'user' or 'assistant'.
            content:      Message text.
            project_name: Associated project name (for filtering).
        """
        if not content or len(content.split()) < 5:
            return 0
        source = f"chat_{session_id}"
        meta   = {"role": role, "project": project_name, "session_id": session_id}
        chunks = self._proc.process_text(
            content,
            source_path=source,
            source_type="chat",
            metadata=meta,
        )
        return self._upsert_chunks(chunks, source, force=False)

    def index_text(
        self,
        text: str,
        source_path: str,
        source_type: str = "text",
        metadata: Optional[dict] = None,
    ) -> int:
        """Index arbitrary text (notes, summaries, manual entries)."""
        chunks = self._proc.process_text(text, source_path, source_type, metadata)
        return self._upsert_chunks(chunks, source_path, force=True)

    # ── Source management ──────────────────────────────────────────────────────

    def delete_source(self, source_path: str) -> None:
        """Remove all indexed chunks from a given source."""
        with self._lock:
            self._vs.delete_source(source_path)
            self._ss.delete_source(source_path)
            self._indexed_hashes.pop(source_path, None)
        log.info("Deleted source from index: %s", source_path)

    def clear(self) -> None:
        """Wipe all indexed data from both stores."""
        with self._lock:
            self._vs.clear()
            self._ss.clear()
            self._indexed_hashes.clear()
        log.info("RAG index cleared")

    # ── Statistics ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        return {
            "vector_chunks": self._vs.count(),
            "sparse_chunks": self._ss.count(),
            "indexed_sources": len(self._indexed_hashes),
            "vector_sources":  self._vs.list_sources(),
        }

    # ── File watching ──────────────────────────────────────────────────────────

    def start_watching(self, directory: str | Path) -> bool:
        """Start background file watcher. Returns False if watchdog unavailable."""
        if not _HAS_WATCHDOG:
            log.warning("watchdog not installed — file watching disabled")
            return False
        if self._observer is not None:
            return True

        handler = _IndexerEventHandler(self)
        self._observer = Observer()
        self._observer.schedule(handler, str(directory), recursive=True)
        self._observer.start()
        log.info("File watcher started on %s", directory)
        return True

    def stop_watching(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            log.info("File watcher stopped")

    # ── Internal ───────────────────────────────────────────────────────────────

    def _upsert_chunks(
        self, chunks: list[DocumentChunk], source_key: str, force: bool
    ) -> int:
        """Filter by content hash, then write to both stores."""
        if not chunks:
            return 0

        with self._lock:
            known_hashes = self._indexed_hashes.get(source_key, set())

            if not force:
                new_chunks = [c for c in chunks if c.content_hash not in known_hashes]
            else:
                new_chunks = chunks

            if not new_chunks:
                return 0

            added_v = self._vs.upsert_chunks(new_chunks)
            added_s = self._ss.upsert_chunks(new_chunks)

            # Track hashes
            if source_key not in self._indexed_hashes:
                self._indexed_hashes[source_key] = set()
            for c in new_chunks:
                self._indexed_hashes[source_key].add(c.content_hash)

            n_added = max(added_v, added_s)
            if n_added:
                log.info("Indexed %d new chunks from %s", n_added, source_key)
            return n_added
