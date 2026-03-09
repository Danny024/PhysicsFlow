"""
PhysicsFlow RAG — Hybrid Retriever.

Combines dense (ChromaDB + BGE) and sparse (BM25) retrieval using
Reciprocal Rank Fusion (RRF) for score-agnostic fusion.

Pipeline:
    query → QueryProcessor → [dense, sparse] retrieval (parallel)
          → RRF fusion → top-K combined candidates
          → CrossEncoderReranker → final ranked list

RRF formula:
    score(d) = Σ_r  1 / (k + rank_r(d))
    where k=60 (default), r iterates over each retrieval run.

All query variants from ExpandedQuery are retrieved independently
and then fused together, giving a final set of diverse, high-quality
candidates.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from .document_processor import DocumentChunk
from .query_processor import ExpandedQuery, QueryProcessor
from .sparse_store import SparseStore
from .vector_store import VectorStore

log = logging.getLogger(__name__)

# RRF rank constant (higher → penalises rank differences less)
_RRF_K = 60


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, DocumentChunk]]],
    k: int = _RRF_K,
) -> list[tuple[DocumentChunk, float]]:
    """
    Fuse multiple ranked lists via RRF.

    Args:
        ranked_lists: Each element is [(chunk_id, chunk), ...] sorted by relevance.
        k: RRF smoothing constant.

    Returns:
        List of (chunk, rrf_score) sorted descending.
    """
    scores: dict[str, float]        = defaultdict(float)
    chunks: dict[str, DocumentChunk] = {}

    for ranked in ranked_lists:
        for rank, (cid, chunk) in enumerate(ranked, start=1):
            scores[cid] += 1.0 / (k + rank)
            if cid not in chunks:
                chunks[cid] = chunk

    return sorted(
        [(chunks[cid], scores[cid]) for cid in scores],
        key=lambda x: x[1],
        reverse=True,
    )


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Parallel dense + sparse retrieval fused via RRF.

    Accepts a single query string or a pre-expanded ExpandedQuery.
    Retrieves top-K candidates from each store for each query variant,
    then fuses all result lists with RRF into a single ranked set.

    Usage:
        retriever = HybridRetriever(vector_store, sparse_store)
        chunks = retriever.retrieve("permeability distribution E-1H", top_k=10)
    """

    def __init__(
        self,
        vector_store: VectorStore,
        sparse_store: SparseStore,
        query_processor: Optional[QueryProcessor] = None,
        dense_top_k:  int = 20,
        sparse_top_k: int = 20,
    ):
        self._vs     = vector_store
        self._ss     = sparse_store
        self._qp     = query_processor or QueryProcessor()
        self._dense_k  = dense_top_k
        self._sparse_k = sparse_top_k

    def retrieve(
        self,
        query: str | ExpandedQuery,
        top_k: int = 10,
        source_type: Optional[str] = None,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Full hybrid retrieval pipeline.

        Returns (chunk, rrf_score) pairs sorted descending, limited to top_k.
        """
        if isinstance(query, str):
            expanded = self._qp.expand(query)
        else:
            expanded = query

        all_queries = expanded.all_queries
        log.debug("Retrieving for %d query variants", len(all_queries))

        # Fetch from both stores for every query variant (in parallel)
        ranked_lists: list[list[tuple[str, DocumentChunk]]] = []

        with ThreadPoolExecutor(max_workers=min(8, len(all_queries) * 2)) as ex:
            futures = {}
            for q in all_queries:
                futures[ex.submit(self._dense_search,  q, source_type)] = f"dense:{q[:30]}"
                futures[ex.submit(self._sparse_search, q, source_type)] = f"sparse:{q[:30]}"

            for fut in as_completed(futures):
                label = futures[fut]
                try:
                    result = fut.result()
                    if result:
                        ranked_lists.append(result)
                        log.debug("%s → %d results", label, len(result))
                except Exception as e:
                    log.warning("Retrieval error [%s]: %s", label, e)

        if not ranked_lists:
            return []

        fused = reciprocal_rank_fusion(ranked_lists)
        return fused[:top_k]

    # ── Store-level search ─────────────────────────────────────────────────────

    def _dense_search(
        self, query: str, source_type: Optional[str]
    ) -> list[tuple[str, DocumentChunk]]:
        results = self._vs.search(query, top_k=self._dense_k, source_type=source_type)
        return [(c.chunk_id, c) for c, _ in results]

    def _sparse_search(
        self, query: str, source_type: Optional[str]
    ) -> list[tuple[str, DocumentChunk]]:
        results = self._ss.search(query, top_k=self._sparse_k, source_type=source_type)
        return [(c.chunk_id, c) for c, _ in results]
