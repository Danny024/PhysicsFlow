"""
PhysicsFlow — Reservoir Knowledge Graph (core graph structure).

Uses networkx.MultiDiGraph as the in-memory backend.

Node types model the physical and computational entities:
    FIELD        — The field itself (e.g. Norne)
    SEGMENT      — Reservoir segment / geological unit (A, B, C, D, E)
    LAYER        — Reservoir layer (K1..K22)
    WELL         — Producer or injector well (B-2H, E-1H, …)
    PERFORATION  — Specific completion interval (well × layer)
    FAULT        — Fault plane (F-1..F-53)
    SIM_RUN      — Simulation or HM run (run_id)
    PARAMETER    — Uncertain model parameter (perm, poro, fault_mult)
    QUANTITY     — Observable output (WOPR, BHP, WCT, …)

Edge types model the relationships:
    PERFORATES       Well       → Layer        (well completes in layer)
    IN_SEGMENT       Well/Layer → Segment      (entity belongs to segment)
    BOUNDS           Fault      → Segment      (fault separates segments)
    CONNECTS         Segment    → Segment      (flow connectivity)
    SUPPORTS         Well       → Well         (injector → producer)
    HAS_RUN          Field      → SIM_RUN
    CONVERGED_AT     SIM_RUN    → iteration (int, stored as attribute)
    INFLUENCES       PARAMETER  → QUANTITY     (sensitivity direction)
    UPDATED_IN       PARAMETER  → SIM_RUN      (parameter was updated in run)
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Any, Iterator, Optional

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

log = logging.getLogger(__name__)


# ── Enumerations ──────────────────────────────────────────────────────────────

class NodeType(str, Enum):
    FIELD       = "field"
    SEGMENT     = "segment"
    LAYER       = "layer"
    WELL        = "well"
    PERFORATION = "perforation"
    FAULT       = "fault"
    SIM_RUN     = "sim_run"
    PARAMETER   = "parameter"
    QUANTITY    = "quantity"


class EdgeType(str, Enum):
    PERFORATES   = "PERFORATES"
    IN_SEGMENT   = "IN_SEGMENT"
    BOUNDS       = "BOUNDS"
    CONNECTS     = "CONNECTS"
    SUPPORTS     = "SUPPORTS"
    HAS_RUN      = "HAS_RUN"
    INFLUENCES   = "INFLUENCES"
    UPDATED_IN   = "UPDATED_IN"


class WellType(str, Enum):
    PRODUCER = "producer"
    INJECTOR = "injector"
    UNKNOWN  = "unknown"


# ── Node ID helpers ───────────────────────────────────────────────────────────

def _nid(node_type: NodeType, name: str) -> str:
    """Canonical node identifier: '<type>/<name>'."""
    return f"{node_type.value}/{name.upper()}"


# ── ReservoirGraph ────────────────────────────────────────────────────────────

class ReservoirGraph:
    """
    Typed knowledge graph for reservoir entities and their relationships.

    All public methods are thread-safe via an internal lock.

    Usage:
        g = ReservoirGraph()
        g.add_well("B-2H", well_type=WellType.PRODUCER, segment="B")
        g.add_perforation("B-2H", "K3")
        wells = g.wells_in_layer("K3")
    """

    def __init__(self):
        if not _HAS_NX:
            raise ImportError("networkx is required for the knowledge graph. "
                              "Install with: pip install networkx")
        self._g: nx.MultiDiGraph = nx.MultiDiGraph()

    # ── Low-level node / edge access ──────────────────────────────────────────

    def add_node(
        self,
        node_type: NodeType,
        name: str,
        **attrs: Any,
    ) -> str:
        """Add (or update) a typed node. Returns the node id."""
        nid = _nid(node_type, name)
        if self._g.has_node(nid):
            self._g.nodes[nid].update(attrs)
        else:
            self._g.add_node(nid, node_type=node_type.value, name=name, **attrs)
        return nid

    def add_edge(
        self,
        src_type: NodeType,
        src_name: str,
        dst_type: NodeType,
        dst_name: str,
        edge_type: EdgeType,
        **attrs: Any,
    ) -> None:
        """Add a typed directed edge (idempotent by edge_type between same pair)."""
        src = _nid(src_type, src_name)
        dst = _nid(dst_type, dst_name)

        # Ensure both nodes exist (minimal stub if not)
        if not self._g.has_node(src):
            self._g.add_node(src, node_type=src_type.value, name=src_name)
        if not self._g.has_node(dst):
            self._g.add_node(dst, node_type=dst_type.value, name=dst_name)

        # Check if this edge_type already exists between these nodes
        existing = self._g.get_edge_data(src, dst) or {}
        for key, data in existing.items():
            if data.get("edge_type") == edge_type.value:
                self._g[src][dst][key].update(attrs)
                return

        self._g.add_edge(src, dst, edge_type=edge_type.value, **attrs)

    def get_node(self, node_type: NodeType, name: str) -> Optional[dict]:
        """Return node attribute dict, or None if not present."""
        nid = _nid(node_type, name)
        return dict(self._g.nodes[nid]) if self._g.has_node(nid) else None

    def node_exists(self, node_type: NodeType, name: str) -> bool:
        return self._g.has_node(_nid(node_type, name))

    # ── Typed accessors ───────────────────────────────────────────────────────

    def nodes_of_type(self, node_type: NodeType) -> list[dict]:
        """Return all nodes of a given type as attribute dicts."""
        return [
            {**data, "_id": nid}
            for nid, data in self._g.nodes(data=True)
            if data.get("node_type") == node_type.value
        ]

    def names_of_type(self, node_type: NodeType) -> list[str]:
        """Return sorted list of names for a given node type."""
        return sorted(
            data["name"]
            for _, data in self._g.nodes(data=True)
            if data.get("node_type") == node_type.value
        )

    def successors_of_type(
        self,
        node_type: NodeType,
        name: str,
        edge_type: EdgeType,
        target_type: Optional[NodeType] = None,
    ) -> list[dict]:
        """
        Return nodes reachable from (node_type, name) via edge_type,
        optionally filtered by target_type.
        """
        src = _nid(node_type, name)
        if not self._g.has_node(src):
            return []
        result = []
        for _, dst, data in self._g.out_edges(src, data=True):
            if data.get("edge_type") != edge_type.value:
                continue
            dst_data = self._g.nodes[dst]
            if target_type and dst_data.get("node_type") != target_type.value:
                continue
            result.append({**dst_data, "_id": dst})
        return result

    def predecessors_of_type(
        self,
        node_type: NodeType,
        name: str,
        edge_type: EdgeType,
        source_type: Optional[NodeType] = None,
    ) -> list[dict]:
        """Return nodes that point TO (node_type, name) via edge_type."""
        dst = _nid(node_type, name)
        if not self._g.has_node(dst):
            return []
        result = []
        for src, _, data in self._g.in_edges(dst, data=True):
            if data.get("edge_type") != edge_type.value:
                continue
            src_data = self._g.nodes[src]
            if source_type and src_data.get("node_type") != source_type.value:
                continue
            result.append({**src_data, "_id": src})
        return result

    # ── Domain-level helper builders ──────────────────────────────────────────

    def add_field(self, name: str, **attrs) -> None:
        self.add_node(NodeType.FIELD, name, **attrs)

    def add_segment(self, name: str, **attrs) -> None:
        self.add_node(NodeType.SEGMENT, name, **attrs)

    def add_layer(self, name: str, index: int = 0, **attrs) -> None:
        self.add_node(NodeType.LAYER, name, index=index, **attrs)

    def add_well(
        self,
        name: str,
        well_type: WellType = WellType.UNKNOWN,
        segment: Optional[str] = None,
        **attrs,
    ) -> None:
        self.add_node(NodeType.WELL, name, well_type=well_type.value, **attrs)
        if segment:
            self.add_segment(segment)
            self.add_edge(
                NodeType.WELL, name,
                NodeType.SEGMENT, segment,
                EdgeType.IN_SEGMENT,
            )

    def add_perforation(
        self,
        well_name: str,
        layer_name: str,
        k_top: int = 0,
        k_bot: int = 0,
    ) -> None:
        self.add_edge(
            NodeType.WELL, well_name,
            NodeType.LAYER, layer_name,
            EdgeType.PERFORATES,
            k_top=k_top, k_bot=k_bot,
        )
        # Also ensure layer is in the well's segment (if known)
        well_data = self.get_node(NodeType.WELL, well_name)
        if well_data:
            segs = self.successors_of_type(
                NodeType.WELL, well_name, EdgeType.IN_SEGMENT
            )
            for seg in segs:
                self.add_edge(
                    NodeType.LAYER, layer_name,
                    NodeType.SEGMENT, seg["name"],
                    EdgeType.IN_SEGMENT,
                )

    def add_fault(
        self,
        fault_name: str,
        segment_a: Optional[str] = None,
        segment_b: Optional[str] = None,
        transmissibility_mult: float = 1.0,
    ) -> None:
        self.add_node(NodeType.FAULT, fault_name,
                      transmissibility_mult=transmissibility_mult)
        if segment_a:
            self.add_edge(
                NodeType.FAULT, fault_name,
                NodeType.SEGMENT, segment_a,
                EdgeType.BOUNDS,
            )
        if segment_b:
            self.add_edge(
                NodeType.FAULT, fault_name,
                NodeType.SEGMENT, segment_b,
                EdgeType.BOUNDS,
            )

    def add_segment_connection(self, seg_a: str, seg_b: str) -> None:
        self.add_edge(
            NodeType.SEGMENT, seg_a,
            NodeType.SEGMENT, seg_b,
            EdgeType.CONNECTS,
        )
        self.add_edge(
            NodeType.SEGMENT, seg_b,
            NodeType.SEGMENT, seg_a,
            EdgeType.CONNECTS,
        )

    def add_parameter(
        self,
        name: str,
        param_type: str = "continuous",
        typical_range: Optional[str] = None,
        influences: Optional[list[str]] = None,
    ) -> None:
        self.add_node(NodeType.PARAMETER, name,
                      param_type=param_type,
                      typical_range=typical_range or "")
        for qty in (influences or []):
            self.add_node(NodeType.QUANTITY, qty)
            self.add_edge(
                NodeType.PARAMETER, name,
                NodeType.QUANTITY, qty,
                EdgeType.INFLUENCES,
            )

    def add_sim_run(
        self,
        run_id: str,
        run_type: str = "hm",
        converged: bool = False,
        converged_at_iter: Optional[int] = None,
        best_mismatch: Optional[float] = None,
        n_ensemble: Optional[int] = None,
        **attrs,
    ) -> None:
        self.add_node(
            NodeType.SIM_RUN, run_id,
            run_type=run_type,
            converged=converged,
            converged_at_iter=converged_at_iter,
            best_mismatch=best_mismatch,
            n_ensemble=n_ensemble,
            **attrs,
        )

    def add_injector_support(self, injector: str, producer: str) -> None:
        """Declare that an injector provides pressure support to a producer."""
        self.add_edge(
            NodeType.WELL, injector,
            NodeType.WELL, producer,
            EdgeType.SUPPORTS,
        )

    # ── Domain-level queries ──────────────────────────────────────────────────

    def wells_in_layer(self, layer: str) -> list[str]:
        """All wells that perforate a given layer."""
        nodes = self.predecessors_of_type(
            NodeType.LAYER, layer, EdgeType.PERFORATES, NodeType.WELL
        )
        return sorted(n["name"] for n in nodes)

    def layers_of_well(self, well: str) -> list[str]:
        """All layers perforated by a given well."""
        nodes = self.successors_of_type(
            NodeType.WELL, well, EdgeType.PERFORATES, NodeType.LAYER
        )
        return sorted(
            (n["name"] for n in nodes),
            key=lambda x: int(x.lstrip("Kk")) if x.lstrip("Kk").isdigit() else 999,
        )

    def wells_in_segment(self, segment: str) -> list[str]:
        """All wells in a reservoir segment."""
        nodes = self.predecessors_of_type(
            NodeType.SEGMENT, segment, EdgeType.IN_SEGMENT, NodeType.WELL
        )
        return sorted(n["name"] for n in nodes)

    def segment_of_well(self, well: str) -> Optional[str]:
        """The segment a well belongs to."""
        nodes = self.successors_of_type(
            NodeType.WELL, well, EdgeType.IN_SEGMENT, NodeType.SEGMENT
        )
        return nodes[0]["name"] if nodes else None

    def faults_bounding_segment(self, segment: str) -> list[str]:
        """All faults that bound a given segment."""
        nodes = self.predecessors_of_type(
            NodeType.SEGMENT, segment, EdgeType.BOUNDS, NodeType.FAULT
        )
        return sorted(n["name"] for n in nodes)

    def segments_of_fault(self, fault: str) -> list[str]:
        """Segments separated by a fault."""
        nodes = self.successors_of_type(
            NodeType.FAULT, fault, EdgeType.BOUNDS, NodeType.SEGMENT
        )
        return [n["name"] for n in nodes]

    def injectors_supporting(self, producer: str) -> list[str]:
        """Injectors providing pressure support to a producer."""
        nodes = self.predecessors_of_type(
            NodeType.WELL, producer, EdgeType.SUPPORTS, NodeType.WELL
        )
        return sorted(n["name"] for n in nodes)

    def producers_supported_by(self, injector: str) -> list[str]:
        """Producers supported by an injector."""
        nodes = self.successors_of_type(
            NodeType.WELL, injector, EdgeType.SUPPORTS, NodeType.WELL
        )
        return sorted(n["name"] for n in nodes)

    def connected_segments(self, segment: str) -> list[str]:
        """Segments with flow connectivity to a given segment."""
        nodes = self.successors_of_type(
            NodeType.SEGMENT, segment, EdgeType.CONNECTS, NodeType.SEGMENT
        )
        return sorted(n["name"] for n in nodes)

    def parameters_influencing(self, quantity: str) -> list[str]:
        """Parameters that influence a given output quantity."""
        nodes = self.predecessors_of_type(
            NodeType.QUANTITY, quantity, EdgeType.INFLUENCES, NodeType.PARAMETER
        )
        return sorted(n["name"] for n in nodes)

    def converged_runs(self) -> list[dict]:
        """All simulation runs that converged."""
        return [
            n for n in self.nodes_of_type(NodeType.SIM_RUN)
            if n.get("converged", False)
        ]

    def wells_by_type(self, well_type: WellType) -> list[str]:
        """All producers or injectors."""
        return sorted(
            n["name"]
            for n in self.nodes_of_type(NodeType.WELL)
            if n.get("well_type") == well_type.value
        )

    # ── Statistics ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for _, data in self._g.nodes(data=True):
            t = data.get("node_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return {
            "total_nodes": self._g.number_of_nodes(),
            "total_edges": self._g.number_of_edges(),
            "by_type": counts,
        }

    # ── Serialisation (networkx node-link JSON) ───────────────────────────────

    def to_dict(self) -> dict:
        if not _HAS_NX:
            return {}
        return nx.node_link_data(self._g)

    @classmethod
    def from_dict(cls, data: dict) -> "ReservoirGraph":
        g = cls()
        g._g = nx.node_link_graph(data, directed=True, multigraph=True)
        return g

    def __repr__(self) -> str:
        s = self.summary()
        return (f"<ReservoirGraph nodes={s['total_nodes']} "
                f"edges={s['total_edges']} types={s['by_type']}>")
