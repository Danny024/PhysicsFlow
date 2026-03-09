"""
PhysicsFlow KG — JSON Persistence.

Saves/loads the ReservoirGraph as a networkx node-link JSON file.
Uses atomic write (temp → rename) to prevent corruption.

Default path: %APPDATA%/PhysicsFlow/kg/reservoir_graph.json
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from .graph import ReservoirGraph

log = logging.getLogger(__name__)


def _default_path() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / "PhysicsFlow" / "kg" / "reservoir_graph.json"


def save(graph: ReservoirGraph, path: Optional[Path] = None) -> Path:
    """
    Persist the graph to JSON (atomic write).
    Returns the path written to.
    """
    out = Path(path) if path else _default_path()
    out.parent.mkdir(parents=True, exist_ok=True)

    data = graph.to_dict()
    tmp  = out.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(data, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    tmp.replace(out)
    log.info("KG saved: %d nodes, %d edges → %s",
             data.get("nodes", []).__len__(),
             data.get("links", []).__len__(), out)
    return out


def load(path: Optional[Path] = None) -> Optional[ReservoirGraph]:
    """
    Load graph from JSON. Returns None if file not found or corrupt.
    """
    src = Path(path) if path else _default_path()
    if not src.exists():
        return None
    try:
        raw  = json.loads(src.read_text(encoding="utf-8"))
        graph = ReservoirGraph.from_dict(raw)
        s    = graph.summary()
        log.info("KG loaded: %d nodes, %d edges from %s",
                 s["total_nodes"], s["total_edges"], src)
        return graph
    except Exception as e:
        log.warning("KG load failed (%s) — will rebuild from scratch", e)
        return None


def _json_default(obj):
    """Handle non-serialisable types (numpy scalars, etc.)."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
