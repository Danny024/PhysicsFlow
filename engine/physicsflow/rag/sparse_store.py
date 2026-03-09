"""
PhysicsFlow RAG — BM25 Sparse Retrieval Store.

Uses BM25Okapi (rank_bm25) with a reservoir-domain-aware tokenizer.
The index is persisted as a JSON file alongside ChromaDB for durability.

Reservoir-specific tokenizer:
  - Lowercases text
  - Preserves hyphenated well names (B-2H, E-1H)
  - Keeps alphanumeric tokens + underscores
  - Splits on camelCase (FNO3d → fno 3d)
  - Min token length: 2 characters
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except ImportError:
    _HAS_BM25 = False
    log.warning("rank_bm25 not installed — BM25 sparse retrieval disabled")

from .document_processor import DocumentChunk


def _default_index_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "PhysicsFlow" / "rag" / "bm25_index.json"


# ── Reservoir tokenizer ────────────────────────────────────────────────────────

# Matches well names like B-2H, E-1H, D-3BH, NORNE-A1
_WELL_NAME_RE = re.compile(r'\b[A-Z]-\d+[A-Z]{0,2}\b')
_CAMEL_RE     = re.compile(r'(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])')
_TOKEN_RE     = re.compile(r'[a-z0-9α-ωέ_\-]{2,}')


def tokenize(text: str) -> list[str]:
    """
    Reservoir-domain tokenizer.

    Steps:
    1. Preserve well names (B-2H → b-2h, kept intact)
    2. Split camelCase (FNO3d → fno3d, αREKI → αreki)
    3. Lowercase
    4. Extract tokens ≥ 2 chars (alphanumeric + hyphen for well names)
    """
    # Extract well names before lowercasing
    well_names = [w.lower() for w in _WELL_NAME_RE.findall(text)]

    # Split camelCase
    text = _CAMEL_RE.sub(' ', text)
    text = text.lower()

    # Extract tokens
    tokens = _TOKEN_RE.findall(text)

    # Re-inject well names (they may have been broken by regex)
    return tokens + well_names


# ── BM25 Sparse Store ──────────────────────────────────────────────────────────

class SparseStore:
    """
    BM25Okapi index over all indexed document chunks.

    Persistence:
    - Stores tokenised corpus and chunk metadata as JSON.
    - Rebuilds BM25Okapi on load (fast for <100k chunks).
    - Incremental updates are supported via rebuild_index().

    Usage:
        store = SparseStore()
        store.upsert_chunks(chunks)
        results = store.search("permeability E-1H", top_k=20)
    """

    def __init__(self, index_path: Optional[Path] = None):
        self._index_path = index_path or _default_index_path()
        self._lock = threading.Lock()

        # In-memory state
        self._corpus_tokens: list[list[str]] = []   # tokenised docs
        self._chunk_ids:     list[str]       = []   # chunk_id per doc
        self._chunk_texts:   list[str]       = []   # raw text per doc
        self._chunk_metas:   list[dict]      = []   # metadata per doc
        self._bm25: Optional[BM25Okapi]      = None

        self._load()

    # ── Write ──────────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add or replace chunks by chunk_id. Returns number added."""
        if not chunks:
            return 0

        with self._lock:
            existing_ids = set(self._chunk_ids)
            new_count = 0
            for chunk in chunks:
                tokens = tokenize(chunk.text)
                if not tokens:
                    continue
                if chunk.chunk_id in existing_ids:
                    # Update in-place
                    idx = self._chunk_ids.index(chunk.chunk_id)
                    self._corpus_tokens[idx] = tokens
                    self._chunk_texts[idx]   = chunk.text
                    self._chunk_metas[idx]   = self._chunk_to_meta(chunk)
                else:
                    self._corpus_tokens.append(tokens)
                    self._chunk_ids.append(chunk.chunk_id)
                    self._chunk_texts.append(chunk.text)
                    self._chunk_metas.append(self._chunk_to_meta(chunk))
                    existing_ids.add(chunk.chunk_id)
                    new_count += 1

            self._rebuild()
            self._save()

        return new_count

    def delete_source(self, source_path: str) -> None:
        """Remove all chunks from a given source file."""
        with self._lock:
            keep = [i for i, m in enumerate(self._chunk_metas)
                    if m.get("source_path") != source_path]
            self._corpus_tokens = [self._corpus_tokens[i] for i in keep]
            self._chunk_ids     = [self._chunk_ids[i]     for i in keep]
            self._chunk_texts   = [self._chunk_texts[i]   for i in keep]
            self._chunk_metas   = [self._chunk_metas[i]   for i in keep]
            self._rebuild()
            self._save()

    def clear(self) -> None:
        with self._lock:
            self._corpus_tokens = []
            self._chunk_ids     = []
            self._chunk_texts   = []
            self._chunk_metas   = []
            self._bm25          = None
            self._save()

    # ── Read ───────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        source_type: Optional[str] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        BM25 retrieval.
        Returns (chunk, score) pairs sorted descending by BM25 score.
        Scores are raw BM25 values (not normalised to [0,1]).
        """
        if not _HAS_BM25 or self._bm25 is None or not self._chunk_ids:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        with self._lock:
            scores = self._bm25.get_scores(query_tokens)

        # Apply source_type filter and build results
        results: list[tuple[DocumentChunk, float]] = []
        for i, score in enumerate(scores):
            if score <= 0:
                continue
            meta = self._chunk_metas[i]
            if source_type and meta.get("source_type") != source_type:
                continue
            chunk = self._meta_to_chunk(i)
            chunk.score = float(score)
            results.append((chunk, float(score)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def count(self) -> int:
        return len(self._chunk_ids)

    # ── Persistence ───────────────────────────────────────────────────────────

    def _rebuild(self) -> None:
        if not _HAS_BM25 or not self._corpus_tokens:
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(
            self._corpus_tokens,
            k1=1.5,     # term frequency saturation (higher = slower saturation)
            b=0.75,     # length normalisation (0=none, 1=full)
            epsilon=0.25,
        )

    def _save(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "corpus_tokens": self._corpus_tokens,
            "chunk_ids":     self._chunk_ids,
            "chunk_texts":   self._chunk_texts,
            "chunk_metas":   self._chunk_metas,
        }
        tmp = self._index_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._index_path)   # atomic rename

    def _load(self) -> None:
        if not self._index_path.exists():
            return
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
            if data.get("version") != 1:
                return
            self._corpus_tokens = data["corpus_tokens"]
            self._chunk_ids     = data["chunk_ids"]
            self._chunk_texts   = data["chunk_texts"]
            self._chunk_metas   = data["chunk_metas"]
            self._rebuild()
            log.info("BM25 index loaded: %d docs from %s",
                     len(self._chunk_ids), self._index_path)
        except Exception as e:
            log.warning("BM25 index load failed (%s) — starting fresh", e)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_to_meta(chunk: DocumentChunk) -> dict:
        return {
            "source_path": chunk.source_path,
            "source_type": chunk.source_type,
            "page":        chunk.page,
            "section":     chunk.section,
            "domain_tags": chunk.domain_tags,
        }

    def _meta_to_chunk(self, idx: int) -> DocumentChunk:
        meta = self._chunk_metas[idx]
        return DocumentChunk(
            chunk_id    = self._chunk_ids[idx],
            text        = self._chunk_texts[idx],
            source_path = meta.get("source_path", ""),
            source_type = meta.get("source_type", ""),
            page        = meta.get("page", 0),
            section     = meta.get("section", ""),
            domain_tags = meta.get("domain_tags", []),
        )
