"""
PhysicsFlow RAG — Persistent Vector Store (ChromaDB + BGE embeddings).

Uses BAAI/bge-small-en-v1.5 via sentence-transformers for embeddings.
ChromaDB provides persistent local storage with cosine similarity search.

Collection layout:
  physicsflow_main   — all chunks (unified search)
  physicsflow_<type> — per-source-type collections for filtered search

Thread-safe: uses a single lock around write operations.
"""

from __future__ import annotations

import hashlib
import logging
import os
import threading
from pathlib import Path
from typing import Optional

from .document_processor import DocumentChunk

log = logging.getLogger(__name__)

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False
    log.warning("chromadb not installed — vector store disabled")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    log.warning("sentence-transformers not installed — embeddings disabled")


# ── Embedding model ────────────────────────────────────────────────────────────

_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"   # 33M params, 512-dim, fast & accurate
_EMBED_INSTRUCTION = "Represent this reservoir engineering document for retrieval: "


class EmbeddingModel:
    """
    Singleton wrapper around BAAI/bge-small-en-v1.5.

    BGE models benefit from a query-time instruction prefix.
    Documents are embedded without the prefix (as-is).
    """
    _instance: Optional["EmbeddingModel"] = None
    _lock = threading.Lock()

    def __init__(self):
        if not _HAS_ST:
            self._model = None
            return
        log.info("Loading embedding model %s …", _EMBED_MODEL_NAME)
        self._model = SentenceTransformer(_EMBED_MODEL_NAME)
        log.info("Embedding model ready (dim=%d)", self._model.get_sentence_embedding_dimension())

    @classmethod
    def instance(cls) -> "EmbeddingModel":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            return [[0.0] * 384] * len(texts)   # zero vectors as fallback
        vecs = self._model.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        return [v.tolist() for v in vecs]

    def embed_query(self, query: str) -> list[float]:
        if self._model is None:
            return [0.0] * 384
        # BGE instruction prefix improves retrieval quality
        prefixed = _EMBED_INSTRUCTION + query
        vec = self._model.encode(
            prefixed, show_progress_bar=False, normalize_embeddings=True
        )
        return vec.tolist()


# ── ChromaDB vector store ──────────────────────────────────────────────────────

def _default_persist_dir() -> str:
    base = os.environ.get("APPDATA") or str(Path.home())
    return str(Path(base) / "PhysicsFlow" / "rag" / "chroma")


class VectorStore:
    """
    Persistent ChromaDB-backed vector store for PhysicsFlow RAG.

    Provides:
    - upsert_chunks(chunks)    — deduplicated batch upsert
    - search(query, top_k)     — cosine similarity search with optional source filter
    - delete_source(path)      — remove all chunks from a source file
    - count()                  — number of indexed chunks
    - clear()                  — wipe all data
    """

    MAIN_COLLECTION = "physicsflow_main"

    def __init__(self, persist_dir: Optional[str] = None):
        self._persist_dir = persist_dir or _default_persist_dir()
        self._embedder = EmbeddingModel.instance()
        self._lock = threading.Lock()
        self._client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self._init()

    def _init(self):
        if not _HAS_CHROMADB:
            return
        Path(self._persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.MAIN_COLLECTION,
            metadata={"hnsw:space": "cosine"},     # cosine similarity
        )
        log.info("VectorStore ready: %d chunks in %s",
                 self._collection.count(), self._persist_dir)

    # ── Write operations ───────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """
        Upsert chunks by chunk_id (idempotent).
        Returns number of new chunks added.
        """
        if not chunks or self._collection is None:
            return 0

        texts     = [c.text                          for c in chunks]
        ids       = [c.chunk_id                      for c in chunks]
        metadatas = [self._build_meta(c)             for c in chunks]

        embeddings = self._embedder.embed_documents(texts)

        with self._lock:
            # ChromaDB upsert is idempotent on id
            BATCH = 256   # avoid memory spikes on large corpora
            for i in range(0, len(chunks), BATCH):
                self._collection.upsert(
                    ids        = ids[i:i+BATCH],
                    embeddings = embeddings[i:i+BATCH],
                    documents  = texts[i:i+BATCH],
                    metadatas  = metadatas[i:i+BATCH],
                )

        log.debug("Upserted %d chunks into vector store", len(chunks))
        return len(chunks)

    def delete_source(self, source_path: str) -> None:
        """Remove all chunks originating from a given source file."""
        if self._collection is None:
            return
        with self._lock:
            self._collection.delete(
                where={"source_path": {"$eq": source_path}}
            )
        log.info("Deleted chunks for source: %s", source_path)

    def clear(self) -> None:
        """Wipe all data from the collection."""
        if self._client is None:
            return
        with self._lock:
            self._client.delete_collection(self.MAIN_COLLECTION)
            self._collection = self._client.get_or_create_collection(
                name=self.MAIN_COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
        log.info("VectorStore cleared")

    # ── Read operations ────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_type: Optional[str] = None,
        source_path: Optional[str] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Cosine similarity search.

        Returns list of (chunk, score) sorted descending by similarity.
        Score is in [0, 1] (1 = identical).
        """
        if self._collection is None or self._collection.count() == 0:
            return []

        query_vec = self._embedder.embed_query(query)

        where: dict = {}
        if source_type:
            where["source_type"] = {"$eq": source_type}
        if source_path:
            where["source_path"] = {"$eq": source_path}

        try:
            result = self._collection.query(
                query_embeddings=[query_vec],
                n_results=min(top_k, max(1, self._collection.count())),
                where=where if where else None,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            log.warning("Vector search error: %s", e)
            return []

        chunks_scores: list[tuple[DocumentChunk, float]] = []
        docs      = result["documents"][0]
        metas     = result["metadatas"][0]
        distances = result["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance: dist ∈ [0, 2]; score = 1 - dist/2
            score = max(0.0, 1.0 - dist / 2.0)
            chunk = self._meta_to_chunk(doc, meta)
            chunk.score = score
            chunks_scores.append((chunk, score))

        return sorted(chunks_scores, key=lambda x: x[1], reverse=True)

    def count(self) -> int:
        if self._collection is None:
            return 0
        return self._collection.count()

    def list_sources(self) -> list[str]:
        """Return distinct source_path values in the store."""
        if self._collection is None or self._collection.count() == 0:
            return []
        try:
            all_meta = self._collection.get(include=["metadatas"])["metadatas"]
            return list({m.get("source_path", "") for m in all_meta})
        except Exception:
            return []

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_meta(chunk: DocumentChunk) -> dict:
        return {
            "source_path": chunk.source_path,
            "source_type": chunk.source_type,
            "page":        chunk.page,
            "section":     chunk.section,
            "domain_tags": ",".join(chunk.domain_tags),
            **{k: str(v) for k, v in chunk.metadata.items()
               if isinstance(v, (str, int, float, bool))},
        }

    @staticmethod
    def _meta_to_chunk(text: str, meta: dict) -> DocumentChunk:
        tags = [t for t in meta.get("domain_tags", "").split(",") if t]
        extra_meta = {k: v for k, v in meta.items()
                      if k not in ("source_path", "source_type", "page",
                                   "section", "domain_tags")}
        return DocumentChunk(
            chunk_id    = hashlib.sha256(text.encode()).hexdigest()[:16],
            text        = text,
            source_path = meta.get("source_path", ""),
            source_type = meta.get("source_type", ""),
            page        = int(meta.get("page", 0)),
            section     = meta.get("section", ""),
            domain_tags = tags,
            metadata    = extra_meta,
        )


