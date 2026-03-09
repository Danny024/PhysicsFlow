"""
PhysicsFlow RAG — Cross-Encoder Reranker.

Uses cross-encoder/ms-marco-MiniLM-L-6-v2 to rerank the fused
candidate set from HybridRetriever.

The cross-encoder jointly encodes (query, document) pairs — producing
scores that are much more accurate than bi-encoder cosine similarity
or BM25, at the cost of being O(n) in candidate count.

Graceful degradation:
  - sentence-transformers not installed → pass-through (RRF scores kept)
  - Model download fails → pass-through
"""

from __future__ import annotations

import logging
from typing import Optional

from .document_processor import DocumentChunk

log = logging.getLogger(__name__)

_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

try:
    from sentence_transformers import CrossEncoder
    _HAS_CROSS_ENCODER = True
except ImportError:
    _HAS_CROSS_ENCODER = False
    log.warning("sentence-transformers not installed — cross-encoder reranking disabled")


class CrossEncoderReranker:
    """
    Reranks (query, chunk) pairs using a cross-encoder model.

    The model is loaded lazily on first use and shared as a singleton.

    Usage:
        reranker = CrossEncoderReranker()
        reranked = reranker.rerank("permeability near fault", candidates, top_k=5)
    """

    _instance: Optional["CrossEncoderReranker"] = None
    _model = None  # shared cross-encoder instance

    def __init__(self, model_name: str = _CROSS_ENCODER_MODEL):
        self._model_name = model_name

    @classmethod
    def instance(cls) -> "CrossEncoderReranker":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        if self._model is not None or not _HAS_CROSS_ENCODER:
            return
        try:
            log.info("Loading cross-encoder %s …", self._model_name)
            CrossEncoderReranker._model = CrossEncoder(
                self._model_name, max_length=512
            )
            log.info("Cross-encoder ready")
        except Exception as e:
            log.warning("Cross-encoder load failed (%s) — using pass-through", e)

    def rerank(
        self,
        query: str,
        candidates: list[tuple[DocumentChunk, float]],
        top_k: int = 5,
    ) -> list[tuple[DocumentChunk, float]]:
        """
        Rerank candidate (chunk, score) pairs against query.

        Returns top_k (chunk, score) sorted descending.
        Scores are cross-encoder logits (higher = more relevant).
        Falls back to original scores if model unavailable.
        """
        if not candidates:
            return []

        if not _HAS_CROSS_ENCODER:
            return candidates[:top_k]

        self._load_model()
        if CrossEncoderReranker._model is None:
            return candidates[:top_k]

        chunks  = [c for c, _ in candidates]
        pairs   = [(query, c.text[:512]) for c in chunks]

        try:
            scores  = CrossEncoderReranker._model.predict(pairs, show_progress_bar=False)
            ranked  = sorted(
                zip(chunks, scores.tolist()),
                key=lambda x: x[1],
                reverse=True,
            )
            return ranked[:top_k]
        except Exception as e:
            log.warning("Cross-encoder rerank failed (%s) — using RRF scores", e)
            return candidates[:top_k]
