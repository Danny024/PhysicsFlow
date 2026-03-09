"""
PhysicsFlow KG — High-level Pipeline.

KGPipeline wires together:
    graph        — ReservoirGraph (networkx)
    builder      — KGBuilder (population from pfproj / DB / context)
    query_engine — KGQueryEngine (pattern → graph traversal)
    serializer   — JSON persistence

Singleton pattern — one shared graph per process.

Usage:
    kg = KGPipeline.instance()
    answer = kg.query("Which wells perforate layer K-9?")
    print(answer.answer)

    # Rebuild after loading a new project
    kg.rebuild(pfproj_path="project.pfproj", db_service=db)
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Optional

from . import serializer
from .builder import KGBuilder
from .graph import ReservoirGraph
from .query_engine import KGAnswer, KGQueryEngine

log = logging.getLogger(__name__)


class KGPipeline:
    """
    Singleton knowledge graph pipeline.

    Lifecycle:
        1. On first access: try to load from disk; fall back to Norne base.
        2. rebuild(**sources) enriches from pfproj / DB / context.
        3. save() persists to disk.
        4. query(text) returns a KGAnswer.
    """

    _instance: Optional["KGPipeline"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._graph_lock = threading.RLock()
        self.graph        = self._load_or_build()
        self.query_engine = KGQueryEngine(self.graph)

    @classmethod
    def instance(cls) -> "KGPipeline":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, text: str) -> KGAnswer:
        """Run a natural language query against the graph."""
        with self._graph_lock:
            return self.query_engine.query(text)

    def is_kg_query(self, text: str) -> bool:
        """Return True if the text looks like a graph-structured query."""
        return self.query_engine.is_kg_query(text)

    def rebuild(
        self,
        pfproj_path: Optional[str | Path] = None,
        db_service=None,
        context_provider=None,
        save_to_disk: bool = True,
    ) -> None:
        """
        Rebuild the graph from all available sources.
        Starts fresh from the Norne base and layers in project-specific data.
        """
        with self._graph_lock:
            graph = ReservoirGraph()
            KGBuilder.build_norne_base(graph)

            if pfproj_path:
                KGBuilder.from_pfproj(pfproj_path, graph)

            if db_service:
                KGBuilder.from_db(db_service, graph)

            if context_provider:
                KGBuilder.from_context_provider(context_provider, graph)

            self.graph        = graph
            self.query_engine = KGQueryEngine(graph)

            if save_to_disk:
                try:
                    serializer.save(graph)
                except Exception as e:
                    log.warning("KG save failed: %s", e)

            log.info("KG rebuilt — %s", graph.summary())

    def update_from_context(self, context_provider) -> None:
        """Lightweight update: sync live per-well RMSE without full rebuild."""
        with self._graph_lock:
            KGBuilder.from_context_provider(context_provider, self.graph)

    def save(self, path: Optional[Path] = None) -> None:
        with self._graph_lock:
            serializer.save(self.graph, path)

    def stats(self) -> dict:
        with self._graph_lock:
            return self.graph.summary()

    # ── Internal ──────────────────────────────────────────────────────────────

    @staticmethod
    def _load_or_build() -> ReservoirGraph:
        """Try disk first; fall back to Norne base if not found/corrupt."""
        graph = serializer.load()
        if graph is None:
            log.info("No saved KG found — building Norne base")
            graph = ReservoirGraph()
            KGBuilder.build_norne_base(graph)
            try:
                serializer.save(graph)
            except Exception as e:
                log.debug("KG initial save skipped: %s", e)
        return graph
