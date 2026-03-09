"""
PhysicsFlow KG — Query Engine.

Pattern-based natural language → graph traversal engine.
Each pattern matches a class of questions and dispatches to a
typed graph query, returning a structured answer dict.

Pattern registry (ordered — first match wins):
  P01  layers_of_well       "what layers does B-2H perforate?"
  P02  wells_in_layer       "which wells are in layer K-9?"
  P03  wells_in_segment     "which wells are in segment D?"
  P04  segment_of_well      "what segment is B-2H in?"
  P05  injectors_for        "which injectors support B-2H?"
  P06  producers_by         "which producers does F-1H support?"
  P07  faults_of_segment    "which faults bound segment C?"
  P08  segments_of_fault    "what segments does F-NE separate?"
  P09  connected_segments   "which segments connect to B?"
  P10  parameters_for_qty   "which parameters influence WOPR?"
  P11  converged_runs       "which runs converged?"
  P12  all_producers        "list all producer wells"
  P13  all_injectors        "list all injector wells"
  P14  all_wells            "list all wells"
  P15  all_parameters       "what are the uncertain parameters?"
  P16  all_faults           "list all faults"
  P17  all_layers           "list all layers"
  P18  connectivity_path    "how are B-2H and E-1H connected?"
  P19  well_type            "is D-4H a producer or injector?"
  P20  worst_match_wells    "which wells have the worst history match?"

Returns:
    dict with keys: answer (str), entities (list), query_type (str),
                    data (dict), confidence (float)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Pattern

from .graph import NodeType, ReservoirGraph, WellType

log = logging.getLogger(__name__)

# ── Result type ───────────────────────────────────────────────────────────────

@dataclass
class KGAnswer:
    answer:     str
    entities:   list[str]         = field(default_factory=list)
    query_type: str                = ""
    data:       dict[str, Any]     = field(default_factory=dict)
    confidence: float              = 1.0
    matched:    bool               = True


_NO_MATCH = KGAnswer(
    answer="",
    matched=False,
    confidence=0.0,
    query_type="no_match",
)


# ── Well-name extractor ────────────────────────────────────────────────────────

_WELL_RE   = re.compile(r'\b([A-Z]-\d+[A-Z]{0,2})\b')
_LAYER_RE  = re.compile(r'\bk[-_]?(\d+)\b', re.IGNORECASE)
_SEGMENT_RE = re.compile(r'\bsegment\s+([A-E])\b', re.IGNORECASE)
_FAULT_RE  = re.compile(r'\b(F-[A-Z0-9]+)\b', re.IGNORECASE)


def _extract_well(text: str) -> Optional[str]:
    m = _WELL_RE.search(text)
    return m.group(1).upper() if m else None


def _extract_layer(text: str) -> Optional[str]:
    m = _LAYER_RE.search(text)
    return f"K{m.group(1)}" if m else None


def _extract_segment(text: str) -> Optional[str]:
    m = _SEGMENT_RE.search(text)
    return m.group(1).upper() if m else None


def _extract_fault(text: str) -> Optional[str]:
    m = _FAULT_RE.search(text)
    return m.group(1).upper() if m else None


def _extract_quantity(text: str) -> Optional[str]:
    qtys = ["WOPR", "WWPR", "WGPR", "BHP", "WCT", "EUR", "WWCT"]
    t = text.upper()
    for q in qtys:
        if q in t:
            return q
    # Aliases
    aliases = {
        "OIL RATE": "WOPR", "WATER RATE": "WWPR", "GAS RATE": "WGPR",
        "BOTTOM HOLE": "BHP", "WATER CUT": "WCT", "RECOVERY": "EUR",
    }
    for alias, qty in aliases.items():
        if alias in t:
            return qty
    return None


# ── Pattern registry ──────────────────────────────────────────────────────────

@dataclass
class _Pattern:
    name:    str
    regex:   Pattern
    handler: Callable[["KGQueryEngine", re.Match, str], KGAnswer]
    priority: int = 0   # lower = checked first


class KGQueryEngine:
    """
    Dispatches natural language questions to typed graph queries.

    Usage:
        engine = KGQueryEngine(graph)
        answer = engine.query("Which wells perforate layer K-9?")
        print(answer.answer)
    """

    def __init__(self, graph: ReservoirGraph):
        self.graph = graph
        self._patterns = self._build_patterns()

    # ── Public API ────────────────────────────────────────────────────────────

    def query(self, text: str) -> KGAnswer:
        """
        Try each pattern in order; return the first match.
        Returns KGAnswer with matched=False if no pattern fires.
        """
        t = text.strip()
        for pat in self._patterns:
            m = pat.regex.search(t)
            if m:
                try:
                    answer = pat.handler(self, m, t)
                    answer.query_type = pat.name
                    log.debug("KG pattern '%s' matched: %s", pat.name, t[:50])
                    return answer
                except Exception as e:
                    log.warning("KG pattern '%s' error: %s", pat.name, e)
        return _NO_MATCH

    def is_kg_query(self, text: str) -> bool:
        """Quick check: does this question look like a graph query?"""
        return any(pat.regex.search(text) for pat in self._patterns)

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _h_layers_of_well(self, m: re.Match, text: str) -> KGAnswer:
        well = _extract_well(text)
        if not well:
            return _NO_MATCH
        layers = self.graph.layers_of_well(well)
        if not layers:
            return KGAnswer(
                answer=f"No perforation data found for well {well} in the knowledge graph.",
                entities=[well], confidence=0.6,
            )
        return KGAnswer(
            answer=(f"Well {well} perforates {len(layers)} layer(s): "
                    f"{', '.join(layers)}."),
            entities=[well] + layers,
            data={"well": well, "layers": layers},
        )

    def _h_wells_in_layer(self, m: re.Match, text: str) -> KGAnswer:
        layer = _extract_layer(text)
        if not layer:
            return _NO_MATCH
        wells = self.graph.wells_in_layer(layer)
        if not wells:
            return KGAnswer(
                answer=f"No wells found perforating layer {layer}.",
                entities=[layer], confidence=0.6,
            )
        return KGAnswer(
            answer=(f"Layer {layer} is perforated by {len(wells)} well(s): "
                    f"{', '.join(wells)}."),
            entities=wells + [layer],
            data={"layer": layer, "wells": wells},
        )

    def _h_wells_in_segment(self, m: re.Match, text: str) -> KGAnswer:
        seg = _extract_segment(text) or m.group(1).upper()
        wells = self.graph.wells_in_segment(seg)
        if not wells:
            return KGAnswer(
                answer=f"No wells found in segment {seg}.",
                entities=[f"Segment_{seg}"], confidence=0.6,
            )
        # Split by type
        prods = [w for w in wells
                 if self.graph.get_node(NodeType.WELL, w)
                    and self.graph.get_node(NodeType.WELL, w).get("well_type") == "producer"]
        injs  = [w for w in wells if w not in prods]
        parts = []
        if prods:
            parts.append(f"{len(prods)} producer(s): {', '.join(prods)}")
        if injs:
            parts.append(f"{len(injs)} injector(s): {', '.join(injs)}")
        return KGAnswer(
            answer=(f"Segment {seg} contains {len(wells)} well(s) — "
                    + "; ".join(parts) + "."),
            entities=wells + [f"Segment_{seg}"],
            data={"segment": seg, "wells": wells, "producers": prods, "injectors": injs},
        )

    def _h_segment_of_well(self, m: re.Match, text: str) -> KGAnswer:
        well = _extract_well(text)
        if not well:
            return _NO_MATCH
        seg = self.graph.segment_of_well(well)
        if not seg:
            return KGAnswer(
                answer=f"Segment assignment not found for well {well}.",
                entities=[well], confidence=0.5,
            )
        neighbours = self.graph.connected_segments(seg)
        return KGAnswer(
            answer=(f"Well {well} is in Segment {seg}. "
                    f"Segment {seg} has flow connectivity to: "
                    f"{', '.join(neighbours) if neighbours else 'none recorded'}."),
            entities=[well, f"Segment_{seg}"],
            data={"well": well, "segment": seg, "connected_segments": neighbours},
        )

    def _h_injectors_for(self, m: re.Match, text: str) -> KGAnswer:
        producer = _extract_well(text)
        if not producer:
            return _NO_MATCH
        injectors = self.graph.injectors_supporting(producer)
        if not injectors:
            # Fallback: show injectors in same segment
            seg = self.graph.segment_of_well(producer)
            if seg:
                seg_wells = self.graph.wells_in_segment(seg)
                injectors = [
                    w for w in seg_wells
                    if self.graph.get_node(NodeType.WELL, w)
                       and self.graph.get_node(NodeType.WELL, w).get("well_type") == "injector"
                ]
                return KGAnswer(
                    answer=(f"No direct injector support mapped for {producer}. "
                            f"Injectors in the same segment ({seg}): "
                            f"{', '.join(injectors) if injectors else 'none'}."),
                    entities=[producer] + injectors,
                    data={"producer": producer, "segment": seg,
                          "segment_injectors": injectors},
                    confidence=0.7,
                )
            return KGAnswer(
                answer=f"No injector support mapped for {producer}.",
                entities=[producer], confidence=0.5,
            )
        return KGAnswer(
            answer=(f"{len(injectors)} injector(s) provide pressure support to "
                    f"producer {producer}: {', '.join(injectors)}."),
            entities=[producer] + injectors,
            data={"producer": producer, "injectors": injectors},
        )

    def _h_producers_by(self, m: re.Match, text: str) -> KGAnswer:
        injector = _extract_well(text)
        if not injector:
            return _NO_MATCH
        producers = self.graph.producers_supported_by(injector)
        if not producers:
            return KGAnswer(
                answer=f"No producer support mapped for injector {injector}.",
                entities=[injector], confidence=0.5,
            )
        return KGAnswer(
            answer=(f"Injector {injector} supports {len(producers)} producer(s): "
                    f"{', '.join(producers)}."),
            entities=[injector] + producers,
            data={"injector": injector, "producers": producers},
        )

    def _h_faults_of_segment(self, m: re.Match, text: str) -> KGAnswer:
        seg = _extract_segment(text) or m.group(1).upper()
        faults = self.graph.faults_bounding_segment(seg)
        n = len(faults)
        shown = faults[:8]
        tail  = f" (+ {n - 8} more)" if n > 8 else ""
        return KGAnswer(
            answer=(f"Segment {seg} is bounded by {n} fault(s): "
                    f"{', '.join(shown)}{tail}."),
            entities=[f"Segment_{seg}"] + shown,
            data={"segment": seg, "faults": faults},
        )

    def _h_segments_of_fault(self, m: re.Match, text: str) -> KGAnswer:
        fault = _extract_fault(text) or m.group(0).upper()
        segs  = self.graph.segments_of_fault(fault)
        if not segs:
            return KGAnswer(
                answer=f"No segment data found for fault {fault}.",
                entities=[fault], confidence=0.5,
            )
        return KGAnswer(
            answer=(f"Fault {fault} separates segments: {', '.join(segs)}."),
            entities=[fault] + segs,
            data={"fault": fault, "segments": segs},
        )

    def _h_connected_segments(self, m: re.Match, text: str) -> KGAnswer:
        seg = _extract_segment(text) or m.group(1).upper()
        connected = self.graph.connected_segments(seg)
        return KGAnswer(
            answer=(f"Segment {seg} has direct flow connectivity to: "
                    f"{', '.join(connected) if connected else 'no other segments'}."),
            entities=[f"Segment_{seg}"] + connected,
            data={"segment": seg, "connected": connected},
        )

    def _h_params_for_qty(self, m: re.Match, text: str) -> KGAnswer:
        qty = _extract_quantity(text)
        if not qty:
            return _NO_MATCH
        params = self.graph.parameters_influencing(qty)
        return KGAnswer(
            answer=(f"{len(params)} uncertain parameter(s) influence {qty}: "
                    f"{', '.join(params)}."),
            entities=params + [qty],
            data={"quantity": qty, "parameters": params},
        )

    def _h_converged_runs(self, m: re.Match, text: str) -> KGAnswer:
        runs = self.graph.converged_runs()
        if not runs:
            return KGAnswer(
                answer="No completed simulation runs found in the knowledge graph.",
                entities=[], data={"runs": []}, confidence=0.7,
            )
        lines = []
        for r in runs[:10]:
            parts = [f"run {r['name']}"]
            if r.get("converged_at_iter"):
                parts.append(f"converged at iter {r['converged_at_iter']}")
            if r.get("best_mismatch") is not None:
                parts.append(f"mismatch={r['best_mismatch']:.4f}")
            if r.get("n_ensemble"):
                parts.append(f"N={r['n_ensemble']}")
            lines.append(" | ".join(parts))
        tail = f"\n... and {len(runs) - 10} more." if len(runs) > 10 else ""
        return KGAnswer(
            answer=f"{len(runs)} converged run(s) found:\n" + "\n".join(lines) + tail,
            entities=[r["name"] for r in runs[:10]],
            data={"converged_runs": runs},
        )

    def _h_all_producers(self, m: re.Match, text: str) -> KGAnswer:
        wells = self.graph.wells_by_type(WellType.PRODUCER)
        return KGAnswer(
            answer=f"{len(wells)} producer well(s): {', '.join(wells)}.",
            entities=wells,
            data={"producers": wells},
        )

    def _h_all_injectors(self, m: re.Match, text: str) -> KGAnswer:
        wells = self.graph.wells_by_type(WellType.INJECTOR)
        return KGAnswer(
            answer=f"{len(wells)} injector well(s): {', '.join(wells)}.",
            entities=wells,
            data={"injectors": wells},
        )

    def _h_all_wells(self, m: re.Match, text: str) -> KGAnswer:
        prods = self.graph.wells_by_type(WellType.PRODUCER)
        injs  = self.graph.wells_by_type(WellType.INJECTOR)
        all_w = prods + injs
        return KGAnswer(
            answer=(f"{len(all_w)} total wells — "
                    f"{len(prods)} producers: {', '.join(prods)}; "
                    f"{len(injs)} injectors: {', '.join(injs)}."),
            entities=all_w,
            data={"producers": prods, "injectors": injs},
        )

    def _h_all_parameters(self, m: re.Match, text: str) -> KGAnswer:
        params = self.graph.names_of_type(NodeType.PARAMETER)
        return KGAnswer(
            answer=(f"{len(params)} uncertain parameter(s) in the model: "
                    f"{', '.join(params)}."),
            entities=params,
            data={"parameters": params},
        )

    def _h_all_faults(self, m: re.Match, text: str) -> KGAnswer:
        faults = self.graph.names_of_type(NodeType.FAULT)
        n = len(faults)
        shown = faults[:12]
        tail  = f" ... ({n - 12} more)" if n > 12 else ""
        return KGAnswer(
            answer=f"{n} fault(s) in the model: {', '.join(shown)}{tail}.",
            entities=shown,
            data={"faults": faults, "total": n},
        )

    def _h_all_layers(self, m: re.Match, text: str) -> KGAnswer:
        layers = self.graph.names_of_type(NodeType.LAYER)
        return KGAnswer(
            answer=f"{len(layers)} reservoir layer(s): {', '.join(layers)}.",
            entities=layers,
            data={"layers": layers},
        )

    def _h_connectivity_path(self, m: re.Match, text: str) -> KGAnswer:
        """How are two wells connected? Via shared segments."""
        wells = _WELL_RE.findall(text.upper())
        if len(wells) < 2:
            return _NO_MATCH
        w1, w2 = wells[0], wells[1]
        seg1 = self.graph.segment_of_well(w1)
        seg2 = self.graph.segment_of_well(w2)
        if seg1 and seg2 and seg1 == seg2:
            return KGAnswer(
                answer=(f"Wells {w1} and {w2} are in the same segment ({seg1}) — "
                        f"they share the same drainage volume."),
                entities=[w1, w2, f"Segment_{seg1}"],
                data={"w1": w1, "w2": w2, "shared_segment": seg1},
            )
        elif seg1 and seg2:
            connected = self.graph.connected_segments(seg1)
            if seg2 in connected:
                return KGAnswer(
                    answer=(f"Wells {w1} (Segment {seg1}) and {w2} (Segment {seg2}) "
                            f"are in directly connected segments — potential flow "
                            f"communication across the inter-segment boundary."),
                    entities=[w1, w2, f"Segment_{seg1}", f"Segment_{seg2}"],
                    data={"w1": w1, "w2": w2, "seg1": seg1, "seg2": seg2,
                          "direct_connection": True},
                )
            else:
                return KGAnswer(
                    answer=(f"Wells {w1} (Segment {seg1}) and {w2} (Segment {seg2}) "
                            f"are in different, non-directly-connected segments — "
                            f"limited flow communication expected."),
                    entities=[w1, w2, f"Segment_{seg1}", f"Segment_{seg2}"],
                    data={"w1": w1, "w2": w2, "seg1": seg1, "seg2": seg2,
                          "direct_connection": False},
                )
        return KGAnswer(
            answer=f"Cannot determine connectivity — segment data missing for one or both wells.",
            entities=[w1, w2], confidence=0.4,
        )

    def _h_well_type(self, m: re.Match, text: str) -> KGAnswer:
        well = _extract_well(text)
        if not well:
            return _NO_MATCH
        node = self.graph.get_node(NodeType.WELL, well)
        if not node:
            return KGAnswer(
                answer=f"Well {well} not found in the knowledge graph.",
                entities=[well], confidence=0.5,
            )
        wtype = node.get("well_type", "unknown")
        seg   = self.graph.segment_of_well(well) or "unknown"
        return KGAnswer(
            answer=f"{well} is a {wtype} in Segment {seg}.",
            entities=[well],
            data={"well": well, "type": wtype, "segment": seg},
        )

    def _h_worst_match_wells(self, m: re.Match, text: str) -> KGAnswer:
        wells = self.graph.nodes_of_type(NodeType.WELL)
        rmse_wells = [
            (n["name"], n["last_rmse"])
            for n in wells
            if "last_rmse" in n
        ]
        if not rmse_wells:
            return KGAnswer(
                answer="No RMSE data available in the graph — run history matching first.",
                entities=[], data={}, confidence=0.6,
            )
        rmse_wells.sort(key=lambda x: x[1], reverse=True)
        top5 = rmse_wells[:5]
        lines = [f"{w}: RMSE={r:.4f}" for w, r in top5]
        return KGAnswer(
            answer=("Wells with worst history match fit (highest RMSE):\n"
                    + "\n".join(lines)),
            entities=[w for w, _ in top5],
            data={"worst_match": [{"well": w, "rmse": r} for w, r in top5]},
        )

    # ── Pattern table ─────────────────────────────────────────────────────────

    def _build_patterns(self) -> list[_Pattern]:
        return [
            _Pattern("layers_of_well",
                re.compile(r'(?:what|which|list)\s+layer[s]?\s+(?:does?\s+)?[A-Z]-\d+[A-Z]{0,2}|'
                           r'perforation[s]?\s+(?:of|for)\s+[A-Z]-\d+', re.I),
                KGQueryEngine._h_layers_of_well),

            _Pattern("wells_in_layer",
                re.compile(r'(?:which|what|list)\s+well[s]?\s+.{0,20}(?:layer|k[-_]?\d+)|'
                           r'layer\s+k[-_]?\d+\s+.{0,20}(?:well[s]?|perforate)', re.I),
                KGQueryEngine._h_wells_in_layer),

            _Pattern("wells_in_segment",
                re.compile(r'(?:which|what|list)\s+well[s]?\s+.{0,20}segment\s+([A-E])|'
                           r'segment\s+([A-E])\s+.{0,20}well[s]?', re.I),
                KGQueryEngine._h_wells_in_segment),

            _Pattern("segment_of_well",
                re.compile(r'(?:what|which)\s+segment\s+.{0,20}[A-Z]-\d+|'
                           r'[A-Z]-\d+[A-Z]{0,2}\s+.{0,20}(?:in|belong|part of)\s+.{0,10}segment',
                           re.I),
                KGQueryEngine._h_segment_of_well),

            _Pattern("injectors_for",
                re.compile(r'(?:which|what)\s+injector[s]?\s+.{0,30}(?:support|feed|maintain|'
                           r'drive)\s+.{0,10}[A-Z]-\d+|'
                           r'injector[s]?\s+(?:for|of|supporting)\s+[A-Z]-\d+', re.I),
                KGQueryEngine._h_injectors_for),

            _Pattern("producers_by",
                re.compile(r'(?:which|what)\s+producer[s]?\s+.{0,30}[A-Z]-\d+\s+support|'
                           r'(?:which|what)\s+(?:producer[s]?|well[s]?)\s+(?:does?|is|are)\s+'
                           r'[A-Z]-\d+.{0,10}support', re.I),
                KGQueryEngine._h_producers_by),

            _Pattern("faults_of_segment",
                re.compile(r'(?:which|what|list)\s+fault[s]?\s+.{0,20}segment\s+([A-E])|'
                           r'fault[s]?\s+.{0,20}bound.{0,20}segment\s+([A-E])', re.I),
                KGQueryEngine._h_faults_of_segment),

            _Pattern("segments_of_fault",
                re.compile(r'(?:what|which)\s+segment[s]?\s+.{0,20}F-[A-Z0-9]+|'
                           r'F-[A-Z0-9]+\s+.{0,20}(?:separate[s]?|bound[s]?|between)', re.I),
                KGQueryEngine._h_segments_of_fault),

            _Pattern("connected_segments",
                re.compile(r'(?:which|what)\s+segment[s]?\s+.{0,30}connect.{0,10}segment\s+([A-E])|'
                           r'segment\s+([A-E]).{0,30}connect', re.I),
                KGQueryEngine._h_connected_segments),

            _Pattern("params_for_qty",
                re.compile(r'(?:which|what)\s+param.{0,20}(?:influ|affect|control|drive|impact)\s+'
                           r'.{0,10}(?:WOPR|WWPR|WGPR|BHP|WCT|EUR|oil rate|water rate|'
                           r'gas rate|pressure|recovery)', re.I),
                KGQueryEngine._h_params_for_qty),

            _Pattern("converged_runs",
                re.compile(r'(?:which|what|list)\s+run[s]?\s+.{0,20}converg|'
                           r'converg.{0,20}run[s]?|'
                           r'(?:which|what)\s+(?:history\s+match|hm)\s+.{0,20}converg', re.I),
                KGQueryEngine._h_converged_runs),

            _Pattern("all_producers",
                re.compile(r'(?:list|show|all|give me)\s+(?:all\s+)?producer[s]?\s*(?:well[s]?)?|'
                           r'(?:which|what)\s+well[s]?\s+are\s+producer[s]?', re.I),
                KGQueryEngine._h_all_producers),

            _Pattern("all_injectors",
                re.compile(r'(?:list|show|all|give me)\s+(?:all\s+)?injector[s]?\s*(?:well[s]?)?|'
                           r'(?:which|what)\s+well[s]?\s+are\s+injector[s]?', re.I),
                KGQueryEngine._h_all_injectors),

            _Pattern("all_wells",
                re.compile(r'(?:list|show)\s+all\s+well[s]?|all\s+well[s]?\s+in\s+(?:the\s+)?'
                           r'(?:field|project|norne)', re.I),
                KGQueryEngine._h_all_wells),

            _Pattern("all_parameters",
                re.compile(r'(?:list|show|what are)\s+(?:the\s+)?uncertain\s+param|'
                           r'(?:all|which)\s+param.{0,20}(?:uncertain|history match|update|kalman)',
                           re.I),
                KGQueryEngine._h_all_parameters),

            _Pattern("all_faults",
                re.compile(r'(?:list|show|all)\s+fault[s]?|'
                           r'how many\s+fault[s]?', re.I),
                KGQueryEngine._h_all_faults),

            _Pattern("all_layers",
                re.compile(r'(?:list|show|all)\s+(?:reservoir\s+)?layer[s]?|'
                           r'how many\s+layer[s]?', re.I),
                KGQueryEngine._h_all_layers),

            _Pattern("connectivity_path",
                re.compile(r'(?:how|what)\s+.{0,30}connect.{0,20}[A-Z]-\d+[A-Z]{0,2}'
                           r'.{0,20}[A-Z]-\d+|'
                           r'[A-Z]-\d+[A-Z]{0,2}\s+and\s+[A-Z]-\d+[A-Z]{0,2}'
                           r'\s+.{0,20}connect', re.I),
                KGQueryEngine._h_connectivity_path),

            _Pattern("well_type",
                re.compile(r'is\s+[A-Z]-\d+[A-Z]{0,2}\s+(?:a\s+)?(?:producer|injector)|'
                           r'[A-Z]-\d+[A-Z]{0,2}\s+.{0,20}(?:producer|injector|type)', re.I),
                KGQueryEngine._h_well_type),

            _Pattern("worst_match_wells",
                re.compile(r'(?:which|what)\s+well[s]?\s+.{0,30}(?:worst|poor|bad|high)\s+'
                           r'.{0,20}(?:match|rmse|mismatch|fit)|'
                           r'worst\s+(?:history\s+)?match', re.I),
                KGQueryEngine._h_worst_match_wells),
        ]
