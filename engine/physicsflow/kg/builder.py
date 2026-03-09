"""
PhysicsFlow KG — Knowledge Graph Builder.

Populates a ReservoirGraph from multiple sources:

    1. build_norne_base()      — Hard-coded Norne structural knowledge
                                 (wells, layers, segments, faults, connectivity)
    2. from_pfproj(path)       — .pfproj JSON project file (well list, notes)
    3. from_db(db_service)     — SQLite: simulation runs, HM iterations,
                                 well observations, model versions
    4. from_context_provider() — Live simulation state (current mismatch,
                                 per-well RMSE from last HM run)

Build order: base → pfproj → db → context (each layer enriches the graph).
All methods are idempotent (calling twice is safe).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from .graph import EdgeType, NodeType, ReservoirGraph, WellType

log = logging.getLogger(__name__)


# ── Norne structural constants ─────────────────────────────────────────────────
#
# Source: Norne Benchmark Case documentation (SINTEF / SPE)
# https://www.opm-project.org/norne/

_NORNE_SEGMENTS = ["A", "B", "C", "D", "E"]

# Approximate segment connectivity (flow paths across faults)
_NORNE_SEGMENT_CONNECTIONS = [
    ("A", "B"), ("B", "C"), ("B", "D"), ("C", "E"), ("D", "E"),
]

_NORNE_LAYERS = [f"K{i}" for i in range(1, 23)]   # K1..K22

# Producer wells with their primary segment and typical perforation layers
# (simplified — exact completions read from COMPDAT in Eclipse deck)
_NORNE_PRODUCERS: list[tuple[str, str, list[str]]] = [
    # (well_name, segment, perforation_layers)
    ("B-1H",  "B", ["K6", "K7", "K8"]),
    ("B-2H",  "B", ["K9", "K10", "K11", "K12"]),
    ("B-4BH", "B", ["K7", "K8", "K9"]),
    ("B-4DH", "B", ["K10", "K11"]),
    ("C-1H",  "C", ["K8", "K9", "K10"]),
    ("C-2H",  "C", ["K7", "K8"]),
    ("C-3H",  "C", ["K9", "K10", "K11"]),
    ("D-1CH", "D", ["K10", "K11", "K12"]),
    ("D-2H",  "D", ["K8", "K9", "K10"]),
    ("D-3BH", "D", ["K11", "K12"]),
    ("D-4H",  "D", ["K9", "K10"]),
    ("E-1H",  "E", ["K8", "K9", "K10", "K11"]),
    ("E-2H",  "E", ["K7", "K8", "K9"]),
    ("E-3AH", "E", ["K10", "K11"]),
    ("E-3CH", "E", ["K9", "K10"]),
    ("E-4AH", "E", ["K8", "K9", "K10"]),
    ("K-3H",  "C", ["K6", "K7", "K8"]),
]

# Injector wells with segment and perforation layers
_NORNE_INJECTORS: list[tuple[str, str, list[str]]] = [
    ("C-4AH", "C", ["K9", "K10", "K11"]),
    ("C-4H",  "C", ["K8", "K9"]),
    ("F-1H",  "B", ["K8", "K9", "K10"]),
    ("F-2H",  "B", ["K9", "K10"]),
    ("F-4H",  "D", ["K10", "K11"]),
]

# Injector → producer support pairs (pressure maintenance)
_NORNE_INJECTOR_SUPPORT: list[tuple[str, str]] = [
    ("C-4AH", "C-1H"), ("C-4AH", "C-2H"), ("C-4AH", "C-3H"), ("C-4AH", "K-3H"),
    ("C-4H",  "C-1H"), ("C-4H",  "C-2H"),
    ("F-1H",  "B-1H"), ("F-1H",  "B-2H"), ("F-1H",  "B-4BH"),
    ("F-2H",  "B-2H"), ("F-2H",  "B-4DH"),
    ("F-4H",  "D-2H"), ("F-4H",  "D-3BH"), ("F-4H",  "D-4H"),
]

# Major Norne faults (first 12 named; full 53 indexed)
_NORNE_FAULTS: list[tuple[str, str, str]] = [
    # (fault_id, segment_a, segment_b)
    ("F-NE",  "A", "B"),
    ("F-NW",  "B", "C"),
    ("F-SE",  "C", "D"),
    ("F-SW",  "D", "E"),
    ("F-MAIN","B", "D"),
    ("F-12",  "A", "C"),
    ("F-23",  "B", "E"),
    ("F-34",  "C", "E"),
]
# Remaining 45 faults added as generic F-09 .. F-53
for _i in range(9, 54):
    _NORNE_FAULTS.append((f"F-{_i:02d}", "B", "C"))   # approximate

# Uncertain parameters and what they influence
_NORNE_PARAMETERS: list[tuple[str, str, list[str]]] = [
    # (name, type, influenced_quantities)
    ("perm_i",   "field_array", ["WOPR", "WWPR", "BHP"]),
    ("perm_j",   "field_array", ["WOPR", "WWPR", "BHP"]),
    ("perm_k",   "field_array", ["WOPR", "BHP"]),
    ("poro",     "field_array", ["EUR", "WOPR"]),
    *[(f"fault_mult_{f[0]}", "scalar", ["BHP", "WOPR"])
      for f in _NORNE_FAULTS[:8]],
    ("kr_oil",   "table",       ["WOPR", "WWCT"]),
    ("kr_water", "table",       ["WWPR", "WWCT"]),
    ("kr_gas",   "table",       ["WGPR"]),
    ("pvt_bo",   "table",       ["WOPR", "EUR"]),
    ("pvt_bg",   "table",       ["WGPR"]),
    ("pvt_rs",   "table",       ["WGPR", "WOPR"]),
]

# Output quantities
_NORNE_QUANTITIES = ["WOPR", "WWPR", "WGPR", "BHP", "WCT", "EUR", "WWCT"]


# ── Builder ────────────────────────────────────────────────────────────────────

class KGBuilder:
    """
    Stateless builder — all methods take a ReservoirGraph and mutate it.
    Call methods in order: base → pfproj → db → context.
    """

    # ── Step 1: Norne structural base ─────────────────────────────────────────

    @staticmethod
    def build_norne_base(graph: ReservoirGraph) -> None:
        """Populate the graph with the hard-coded Norne structural knowledge."""
        # Field node
        graph.add_field(
            "NORNE",
            description="Norne Oil Field, Norwegian Sea (PL128)",
            nx=46, ny=112, nz=22,
            initial_pressure_bar=277,
            datum_depth_m=2609,
        )

        # Segments
        segment_desc = {
            "A": "Northern tilted fault block, Åre/Ile formation",
            "B": "Central main segment, highest STOIIP",
            "C": "Western Garn/Ile segment, near C/K-wells",
            "D": "Southern segment, D/F-wells",
            "E": "Eastern pinch-out segment, E-wells",
        }
        for seg in _NORNE_SEGMENTS:
            graph.add_segment(seg, description=segment_desc.get(seg, ""))

        # Segment connectivity
        for sa, sb in _NORNE_SEGMENT_CONNECTIONS:
            graph.add_segment_connection(sa, sb)

        # Layers
        layer_names = {
            "K1":  "Åre Formation (top)",
            "K6":  "Ile Formation (upper)",
            "K12": "Garn Formation",
            "K17": "Not Formation (lower)",
            "K22": "Åre Formation (base)",
        }
        for i, lyr in enumerate(_NORNE_LAYERS, start=1):
            graph.add_layer(
                lyr, index=i,
                description=layer_names.get(lyr, f"Reservoir layer {i}"),
            )

        # Output quantities
        for qty in _NORNE_QUANTITIES:
            graph.add_node(NodeType.QUANTITY, qty, unit=_qty_unit(qty))

        # Producer wells
        for well, seg, layers in _NORNE_PRODUCERS:
            graph.add_well(well, WellType.PRODUCER, segment=seg)
            for lyr in layers:
                graph.add_perforation(well, lyr)

        # Injector wells
        for well, seg, layers in _NORNE_INJECTORS:
            graph.add_well(well, WellType.INJECTOR, segment=seg)
            for lyr in layers:
                graph.add_perforation(well, lyr)

        # Injector → producer support
        for inj, prod in _NORNE_INJECTOR_SUPPORT:
            graph.add_injector_support(inj, prod)

        # Faults
        for fault_id, sa, sb in _NORNE_FAULTS:
            graph.add_fault(fault_id, segment_a=sa, segment_b=sb)

        # Uncertain parameters
        for pname, ptype, influences in _NORNE_PARAMETERS:
            graph.add_parameter(pname, param_type=ptype, influences=influences)

        log.info("KGBuilder: Norne base populated — %s", graph.summary())

    # ── Step 2: .pfproj project file ──────────────────────────────────────────

    @staticmethod
    def from_pfproj(path: str | Path, graph: ReservoirGraph) -> None:
        """
        Enrich the graph from a .pfproj JSON project file.
        Updates well list, perforation depths, project notes.
        """
        p = Path(path)
        if not p.exists():
            log.warning("KGBuilder.from_pfproj: path not found — %s", p)
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("KGBuilder.from_pfproj: parse error — %s", e)
            return

        project_name = data.get("project_name", p.stem)
        graph.add_field(
            project_name,
            pfproj_path=str(p),
            nx=data.get("grid", {}).get("nx"),
            ny=data.get("grid", {}).get("ny"),
            nz=data.get("grid", {}).get("nz"),
            eclipse_deck=data.get("eclipse_deck_path", ""),
            notes=data.get("notes", ""),
        )

        # Wells
        for w in data.get("wells", []):
            name      = w.get("name", "")
            wtype_str = w.get("type", "unknown").lower()
            wtype = WellType.PRODUCER if "prod" in wtype_str else (
                    WellType.INJECTOR if "inj"  in wtype_str else WellType.UNKNOWN)
            if name:
                graph.add_well(name, wtype)
                k_top = w.get("k_top", 0)
                k_bot = w.get("k_bot", 0)
                for k in range(k_top, k_bot + 1):
                    if k > 0:
                        graph.add_perforation(name, f"K{k}", k_top=k_top, k_bot=k_bot)

        # HM results
        hm = data.get("hm_results", {})
        if hm:
            graph.add_node(
                NodeType.SIM_RUN, f"{project_name}_hm",
                run_type="hm",
                converged=hm.get("converged", False),
                converged_at_iter=hm.get("n_iterations"),
                best_mismatch=hm.get("best_mismatch"),
                eur_p10=hm.get("eur_p10"),
                eur_p50=hm.get("eur_p50"),
                eur_p90=hm.get("eur_p90"),
            )

        log.info("KGBuilder.from_pfproj: enriched from %s", p.name)

    # ── Step 3: SQLite database ───────────────────────────────────────────────

    @staticmethod
    def from_db(db_service, graph: ReservoirGraph) -> None:
        """
        Enrich the graph from the PhysicsFlow SQLite database.
        db_service: AppDbService (optional — skipped if None or unavailable).
        """
        if db_service is None:
            return
        try:
            KGBuilder._sync_runs_from_db(db_service, graph)
            KGBuilder._sync_well_observations_from_db(db_service, graph)
        except Exception as e:
            log.warning("KGBuilder.from_db: failed — %s", e)

    @staticmethod
    def _sync_runs_from_db(db_service, graph: ReservoirGraph) -> None:
        # Use the module-level get_session context manager (not a method on db_service)
        try:
            from physicsflow.db.database import get_session
            from physicsflow.db.models import SimulationRun, HMIteration
            with get_session() as session:
                runs = session.query(SimulationRun).all()
                for run in runs:
                    # Find if any HM iteration for this run converged
                    hm_iters = (
                        session.query(HMIteration)
                        .filter(HMIteration.hm_run_id == run.id)
                        .order_by(HMIteration.iteration.desc())
                        .first()
                    )
                    converged = hm_iters.converged if hm_iters else False
                    conv_iter = hm_iters.iteration  if hm_iters else None
                    mismatch  = hm_iters.mismatch   if hm_iters else None

                    graph.add_sim_run(
                        run.id,
                        run_type=run.run_type,
                        converged=converged,
                        converged_at_iter=conv_iter,
                        best_mismatch=mismatch,
                        n_ensemble=run.n_ensemble,
                        status=run.status,
                    )
        except Exception as e:
            log.debug("KGBuilder._sync_runs_from_db: %s", e)

    @staticmethod
    def _sync_well_observations_from_db(db_service, graph: ReservoirGraph) -> None:
        """Add any well names from observations that aren't already in the graph."""
        try:
            from physicsflow.db.database import get_session
            from physicsflow.db.models import WellObservation
            with get_session() as session:
                well_names = (
                    session.query(WellObservation.well_name)
                    .distinct()
                    .all()
                )
                for (name,) in well_names:
                    if not graph.node_exists(NodeType.WELL, name):
                        graph.add_well(name, WellType.UNKNOWN)
        except Exception as e:
            log.debug("KGBuilder._sync_well_observations_from_db: %s", e)

    # ── Step 4: Live context provider ─────────────────────────────────────────

    @staticmethod
    def from_context_provider(ctx, graph: ReservoirGraph) -> None:
        """
        Sync live per-well RMSE from the most recent HM run into the graph.
        ctx: ReservoirContextProvider
        """
        if ctx is None:
            return
        try:
            per_well = ctx.per_well_mismatch
            for well_name, mismatch_data in per_well.items():
                if graph.node_exists(NodeType.WELL, well_name):
                    rmse = float(mismatch_data.get("total", 0.0)) \
                        if isinstance(mismatch_data, dict) else float(mismatch_data)
                    graph.add_node(NodeType.WELL, well_name,
                                   last_rmse=round(rmse, 4))

            hm_hist = ctx.hm_history
            if hm_hist:
                last = hm_hist[-1]
                graph.add_node(NodeType.FIELD, "NORNE",
                               last_hm_mismatch=last.get("data_mismatch"),
                               last_hm_iter=last.get("iteration"))
        except Exception as e:
            log.debug("KGBuilder.from_context_provider: %s", e)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _qty_unit(qty: str) -> str:
    units = {
        "WOPR": "stb/day", "WWPR": "stb/day", "WGPR": "Mscf/day",
        "BHP":  "bar",     "WCT":  "fraction", "EUR":  "MMstb",
        "WWCT": "fraction",
    }
    return units.get(qty, "")
