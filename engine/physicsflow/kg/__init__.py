"""
PhysicsFlow KG — Reservoir Knowledge Graph Package.

Structured graph of reservoir entities (wells, layers, segments, faults,
simulation runs, parameters) and their typed relationships, enabling
precise relational queries the LLM agent cannot reliably answer via RAG.

Public API
----------
KGPipeline    — Singleton: query(text) → KGAnswer  |  rebuild(...)
ReservoirGraph — Core networkx-backed typed graph
KGBuilder     — Populates graph from pfproj / SQLite / live context
KGQueryEngine — 20-pattern NL → graph traversal dispatcher
KGAnswer      — Structured answer: .answer (str), .entities, .data

Quick start
-----------
    from physicsflow.kg import KGPipeline

    kg = KGPipeline.instance()
    ans = kg.query("Which wells perforate layer K-9?")
    print(ans.answer)
    # → "Layer K9 is perforated by 3 well(s): B-2H, B-4BH, D-2H."

    ans = kg.query("Which injectors support B-2H?")
    print(ans.answer)
    # → "2 injector(s) provide pressure support to producer B-2H: F-1H, F-2H."
"""

from .graph        import ReservoirGraph, NodeType, EdgeType, WellType
from .builder      import KGBuilder
from .query_engine import KGQueryEngine, KGAnswer
from .serializer   import save as save_graph, load as load_graph
from .pipeline     import KGPipeline

__all__ = [
    "ReservoirGraph",
    "NodeType",
    "EdgeType",
    "WellType",
    "KGBuilder",
    "KGQueryEngine",
    "KGAnswer",
    "KGPipeline",
    "save_graph",
    "load_graph",
]
