"""
PhysicsFlow RAG — Retrieval-Augmented Generation Package.

Hybrid retrieval pipeline combining dense (ChromaDB + BGE embeddings)
and sparse (BM25) retrieval with cross-encoder reranking.

Public API
----------
RAGPipeline   — High-level pipeline: retrieve + rerank + build context
RAGIndexer    — Ingest files, text, audit_log entries into both stores
HybridRetriever — Low-level: RRF fusion of dense + sparse results
QueryProcessor  — Multi-query expansion, HyDE, keyword extraction
ContextBuilder  — Format retrieved chunks into LLM-ready context

Quick start
-----------
    from physicsflow.rag import RAGPipeline

    rag = RAGPipeline()
    rag.indexer.index_file("docs/norne_report.pdf")

    context = rag.retrieve_and_build("Why is well B-2H cutting water?")
    print(context.context_block)
"""

from .document_processor import DocumentChunk, DocumentProcessor, TextChunker
from .vector_store        import VectorStore, EmbeddingModel
from .sparse_store        import SparseStore, tokenize
from .query_processor     import QueryProcessor, ExpandedQuery, classify_query
from .retriever           import HybridRetriever, reciprocal_rank_fusion
from .reranker            import CrossEncoderReranker
from .context_builder     import ContextBuilder, RetrievedContext
from .indexer             import RAGIndexer
from .pipeline            import RAGPipeline

__all__ = [
    # Core data types
    "DocumentChunk",
    "DocumentProcessor",
    "TextChunker",
    "ExpandedQuery",
    "RetrievedContext",
    # Stores
    "VectorStore",
    "EmbeddingModel",
    "SparseStore",
    "tokenize",
    # Pipeline components
    "QueryProcessor",
    "classify_query",
    "HybridRetriever",
    "reciprocal_rank_fusion",
    "CrossEncoderReranker",
    "ContextBuilder",
    "RAGIndexer",
    # High-level
    "RAGPipeline",
]
