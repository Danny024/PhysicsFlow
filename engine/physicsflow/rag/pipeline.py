"""
PhysicsFlow RAG — High-Level Pipeline.

RAGPipeline wires together all components into a single object
that the ReservoirAgent calls:

    context = rag.retrieve_and_build(query, top_k=5)
    system_prompt += context_builder.format_for_prompt(context)

The pipeline is a singleton (one shared index per process).
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .context_builder import ContextBuilder, RetrievedContext
from .document_processor import DocumentChunk
from .indexer import RAGIndexer
from .query_processor import QueryProcessor
from .reranker import CrossEncoderReranker
from .retriever import HybridRetriever
from .sparse_store import SparseStore
from .vector_store import VectorStore

log = logging.getLogger(__name__)


class RAGPipeline:
    """
    Full hybrid RAG pipeline as a single composable object.

    Components:
        indexer   — RAGIndexer  (file ingestion → both stores)
        retriever — HybridRetriever  (dense + sparse + RRF)
        reranker  — CrossEncoderReranker
        builder   — ContextBuilder

    Singleton pattern — share one index across the process:
        rag = RAGPipeline.instance()

    Usage:
        rag.indexer.index_file("report.pdf")
        ctx = rag.retrieve_and_build("permeability uncertainty B-2H")
        prompt += ctx.context_block
    """

    _instance: Optional["RAGPipeline"] = None
    _lock = threading.Lock()

    def __init__(
        self,
        llm_model:    str = "phi3:mini",
        top_k_fetch:  int = 20,
        top_k_rerank: int = 5,
        token_budget: int = 1800,
        use_hyde:     bool = True,
        use_reranker: bool = True,
    ):
        self._top_k_fetch  = top_k_fetch
        self._top_k_rerank = top_k_rerank
        self._use_reranker = use_reranker

        # Shared stores
        vector_store = VectorStore()
        sparse_store = SparseStore()

        # Components
        self.indexer   = RAGIndexer(vector_store, sparse_store)
        self.retriever = HybridRetriever(
            vector_store, sparse_store,
            query_processor=QueryProcessor(model=llm_model, use_hyde=use_hyde),
            dense_top_k=top_k_fetch,
            sparse_top_k=top_k_fetch,
        )
        self.reranker = CrossEncoderReranker.instance()
        self.builder  = ContextBuilder(token_budget=token_budget)

        log.info(
            "RAGPipeline ready — vector:%d sparse:%d",
            vector_store.count(), sparse_store.count(),
        )

    @classmethod
    def instance(cls, **kwargs) -> "RAGPipeline":
        """Get or create the singleton RAGPipeline."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    def retrieve_and_build(
        self,
        query:       str,
        top_k:       Optional[int] = None,
        source_type: Optional[str] = None,
        tool_data:   Optional[dict] = None,
    ) -> RetrievedContext:
        """
        Full pipeline: expand query → hybrid retrieve → rerank → build context.

        Args:
            query:       User's question.
            top_k:       Final number of chunks to include in context.
            source_type: Optional filter (e.g. "pdf", "las", "audit").
            tool_data:   Live simulation data to prepend to context.

        Returns:
            RetrievedContext ready for injection into the system prompt.
        """
        k_fetch  = self._top_k_fetch
        k_final  = top_k or self._top_k_rerank

        # 1. Hybrid retrieval (dense + sparse + RRF)
        candidates = self.retriever.retrieve(
            query, top_k=k_fetch, source_type=source_type
        )
        if not candidates:
            log.debug("RAG: no candidates for query: %s", query[:60])
            return RetrievedContext("", "", 0, [])

        # 2. Cross-encoder reranking
        if self._use_reranker:
            reranked = self.reranker.rerank(query, candidates, top_k=k_final)
        else:
            reranked = candidates[:k_final]

        # 3. Build formatted context
        ctx = self.builder.build(reranked, tool_data=tool_data)
        log.debug("RAG: %d chunks → %d reranked → context built", len(candidates), len(reranked))
        return ctx

    def stats(self) -> dict:
        """Return index statistics."""
        return self.indexer.stats()
