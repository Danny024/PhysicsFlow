"""
Agent tool definitions and implementations.

These tools give the LLM access to live simulation data so it can
answer questions grounded in actual reservoir results.

Each tool returns a JSON-serialisable dict and is registered in
TOOL_DEFINITIONS for Ollama tool-calling.
"""

from __future__ import annotations
import json
from typing import Any
from .context_provider import ReservoirContextProvider

try:
    from physicsflow.rag.pipeline import RAGPipeline as _RAGPipeline
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False

try:
    from physicsflow.kg.pipeline import KGPipeline as _KGPipeline
    _HAS_KG = True
except Exception:
    _HAS_KG = False


# ── Ollama tool definitions (OpenAI-compatible function schema) ───────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "get_simulation_status",
            "description": (
                "Get the current status of the running simulation or history match. "
                "Returns progress, elapsed time, current iteration, and status message."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_well_performance",
            "description": (
                "Get production/injection time series for a specific well or all wells. "
                "Returns WOPR (oil rate), WWPR (water rate), WGPR (gas rate) in field units."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "well_name": {
                        "type": "string",
                        "description": "Well name (e.g. 'B-2H'). Use 'all' for all wells.",
                    }
                },
                "required": ["well_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_hm_iteration_summary",
            "description": (
                "Get history matching convergence summary: data mismatch per iteration, "
                "alpha values, ensemble spread, and convergence status."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_ensemble_statistics",
            "description": (
                "Get P10/P50/P90 ensemble statistics for production quantities "
                "(WOPR, WWPR, WGPR, EUR) across all wells."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "quantity": {
                        "type": "string",
                        "description": "One of: 'wopr', 'wwpr', 'wgpr', 'eur', 'pressure'",
                    },
                    "well_name": {
                        "type": "string",
                        "description": "Well name, or 'all' for field totals",
                    },
                },
                "required": ["quantity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_data_mismatch_per_well",
            "description": (
                "Get per-well data mismatch (RMSE) breakdown showing which wells "
                "are matching poorly and which quantities (oil/water/gas) have issues."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_field_property",
            "description": (
                "Get reservoir property value at a specific grid cell (i, j, k). "
                "Properties: 'perm' (mD), 'poro', 'pressure' (psia), 'sw', 'sg', 'so'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "property": {"type": "string"},
                    "i": {"type": "integer", "description": "I grid index (1-based)"},
                    "j": {"type": "integer", "description": "J grid index"},
                    "k": {"type": "integer", "description": "K grid index (layer)"},
                    "timestep": {"type": "integer", "description": "Time step (0=initial, -1=final)"},
                },
                "required": ["property", "i", "j", "k"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_parameter",
            "description": (
                "Get a technical explanation of any reservoir simulation parameter, "
                "including its physical meaning, typical ranges, and role in the model."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "parameter": {
                        "type": "string",
                        "description": (
                            "Parameter name, e.g.: 'alpha', 'kalman_gain', 'perm', 'poro', "
                            "'gaspari_cohn', 'fno', 'pino', 'areki', 'vcae', 'ddim', "
                            "'stone_ii', 'peacemann', 'ccr', 'fault_mult', 'pvt', 'bo', 'bg'"
                        ),
                    }
                },
                "required": ["parameter"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_reservoir_graph",
            "description": (
                "Query the reservoir knowledge graph for structured relational facts: "
                "well completions (perforation layers), segment membership, injector–producer "
                "support relationships, fault–segment boundaries, uncertain parameters, "
                "and simulation run convergence. Use this for questions about reservoir "
                "topology, well connectivity, and structural relationships — not for "
                "time-series production data (use get_well_performance for that)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": (
                            "Natural language question about reservoir structure, e.g.: "
                            "'Which wells perforate layer K-9?', "
                            "'Which injectors support B-2H?', "
                            "'What segment is D-3BH in?', "
                            "'Which faults bound segment C?', "
                            "'Which runs converged?'"
                        ),
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_project_knowledge",
            "description": (
                "Search the project knowledge base (indexed PDFs, well logs, reports, "
                "project files, and previous analyses) using hybrid semantic + keyword "
                "retrieval. Use this to find relevant background information, technical "
                "reports, LAS well data, or prior history matching results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'permeability uncertainty B-2H'",
                    },
                    "source_type": {
                        "type": "string",
                        "description": (
                            "Optional filter: 'pdf', 'las', 'csv', 'pfproj', "
                            "'text', 'audit', 'chat'. Omit to search all."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return (default: 5, max: 10)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_project_summary",
            "description": (
                "Get a full summary of the current project: field name, grid dimensions, "
                "number of wells, simulation period, training status, and HM status."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ── Tool implementations ──────────────────────────────────────────────────────

class ReservoirTools:
    """
    Implementation of all agent tools.
    Reads live data from the shared ReservoirContextProvider.
    """

    def __init__(self, ctx: ReservoirContextProvider):
        self.ctx = ctx

    def get_simulation_status(self) -> dict:
        s = self.ctx.simulation_state
        if s is None:
            return {"status": "No simulation running", "progress": 0}
        return {
            "status": s.get("status", "unknown"),
            "progress_pct": s.get("progress_pct", 0),
            "elapsed_sec": s.get("elapsed_sec", 0),
            "eta_sec": s.get("eta_sec", 0),
            "current_step": s.get("current_step", 0),
            "total_steps": s.get("total_steps", 0),
            "last_loss": s.get("last_loss", None),
        }

    def get_well_performance(self, well_name: str) -> dict:
        wells_data = self.ctx.well_results
        if not wells_data:
            return {"error": "No production data available. Load a project or run a simulation."}

        if well_name.lower() == "all":
            # Return compact summary for all producers — avoid huge payload
            summary = {}
            time_d  = self.ctx.time_days
            for w, d in wells_data.items():
                if d.get("status") == "injector":
                    continue
                summary[w] = {
                    "peak_wopr_stbd":   round(max(d.get("wopr", [0])), 0),
                    "cum_oil_stb":      self._cumulative(d.get("wopr", [])),
                    "current_wopr_stbd":round(d.get("wopr", [0])[-1], 0) if d.get("wopr") else 0,
                    "current_wcut":     self._water_cut(d),
                    "status":           d.get("status", "unknown"),
                    "source":           d.get("source", "simulation"),
                }
            return {
                "well_summary": summary,
                "time_days":    time_d,
                "data_source":  next(iter(wells_data.values())).get("source", "simulation"),
                "note": (
                    "source='synthetic_baseline' means reference decline-curve profiles; "
                    "run a PINO simulation to replace with physics-based results."
                    if next(iter(wells_data.values())).get("source") == "synthetic_baseline"
                    else "Live simulation results."
                ),
            }

        well = wells_data.get(well_name)
        if well is None:
            available = [w for w in wells_data if w != "FIELD"][:12]
            return {
                "error": f"Well '{well_name}' not found.",
                "available_wells": available,
            }

        source   = well.get("source", "simulation")
        status   = well.get("status", "on_target")
        time_d   = self.ctx.time_days
        wopr     = well.get("wopr", [])
        wwpr     = well.get("wwpr", [])
        wgpr     = well.get("wgpr", [])
        cum_oil  = self._cumulative(wopr)
        wcut_now = self._water_cut(well)

        return {
            "well_name":       well_name,
            "performance_status": status,
            "data_source":     source,
            "time_days":       time_d,
            "wopr_stb_day":    wopr,
            "wwpr_stb_day":    wwpr,
            "wgpr_mscfd":      wgpr,
            "peak_wopr_stbd":  round(max(wopr), 0) if wopr else 0,
            "cum_oil_stb":     cum_oil,
            "current_wcut":    wcut_now,
            "chart": {
                "chart_type": "line",
                "title": f"{well_name} Production Profile [{source}]",
                "x_label": "Time (days)",
                "y_label": "Rate (STB/day)",
                "series": [
                    {"name": "Oil Rate",   "x": time_d, "y": wopr,  "color": "#228B22"},
                    {"name": "Water Rate", "x": time_d, "y": wwpr,  "color": "#1E90FF"},
                    {"name": "Gas Rate (×0.001)", "x": time_d,
                     "y": [g * 0.001 for g in wgpr], "color": "#FF6B35"},
                ],
            },
        }

    def get_hm_iteration_summary(self) -> dict:
        hm = self.ctx.hm_history
        if not hm:
            # No HM run yet — return a concrete pre-run status using baseline mismatch
            mismatch = self.ctx.per_well_mismatch
            baseline_rmse = self.ctx.overall_rmse
            above = [w for w, v in mismatch.items() if v.get("status") == "above_expectation"]
            below = [w for w, v in mismatch.items() if v.get("status") == "below_expectation"]
            return {
                "hm_status": "not_started",
                "n_iterations": 0,
                "converged": False,
                "baseline_rmse": round(baseline_rmse, 4) if baseline_rmse else "N/A",
                "wells_above_expectation": above,
                "wells_below_expectation": below,
                "note": (
                    "No αREKI history matching run yet. "
                    f"Baseline mismatch RMSE: {round(baseline_rmse, 4) if baseline_rmse else 'N/A'}. "
                    "Navigate to History Match panel and click 'Start αREKI' to begin."
                ),
            }

        iterations = [h["iteration"] for h in hm]
        mismatches = [h["data_mismatch"] for h in hm]
        alphas = [h["alpha"] for h in hm]
        spreads = [h["ensemble_spread"] for h in hm]

        improvement = 0.0
        if len(mismatches) >= 2:
            improvement = (mismatches[0] - mismatches[-1]) / mismatches[0] * 100

        return {
            "n_iterations": len(hm),
            "initial_mismatch": mismatches[0] if mismatches else None,
            "final_mismatch": mismatches[-1] if mismatches else None,
            "improvement_pct": round(improvement, 1),
            "converged": hm[-1].get("converged", False),
            "iterations": iterations,
            "mismatches": [round(m, 4) for m in mismatches],
            "alphas": [round(a, 4) for a in alphas],
            "ensemble_spreads": [round(s, 4) for s in spreads],
            "chart": {
                "chart_type": "line",
                "title": "αREKI Convergence",
                "x_label": "Iteration",
                "y_label": "Data Mismatch (RMSE)",
                "series": [
                    {"name": "Data Mismatch", "x": iterations,
                     "y": [round(m, 4) for m in mismatches], "color": "#E74C3C"},
                    {"name": "Ensemble Spread", "x": iterations,
                     "y": [round(s, 4) for s in spreads], "color": "#3498DB",
                     "line_style": "dashed"},
                ],
            },
        }

    def get_ensemble_statistics(self, quantity: str, well_name: str = "all") -> dict:
        stats = self.ctx.ensemble_stats
        if not stats:
            return {"error": "No ensemble statistics available."}

        q = quantity.lower()
        key = f"{well_name}_{q}" if well_name != "all" else f"field_{q}"
        data = stats.get(key, stats.get(q, {}))

        if not data:
            return {
                "error": f"No statistics for '{quantity}' / '{well_name}'.",
                "available": list(stats.keys()),
            }

        return {
            "quantity": quantity,
            "well": well_name,
            "p10": data.get("p10"),
            "p50": data.get("p50"),
            "p90": data.get("p90"),
            "time_days": self.ctx.time_days,
            "chart": {
                "chart_type": "fan",
                "title": f"P10/P50/P90 — {quantity.upper()} ({well_name})",
                "x_label": "Time (days)",
                "y_label": f"{quantity.upper()} (STB/day)",
                "series": [{
                    "name": "P50", "x": self.ctx.time_days,
                    "y": data.get("p50", []),
                    "y_lower": data.get("p10", []),
                    "y_upper": data.get("p90", []),
                    "color": "#2ECC71",
                }],
            },
        }

    def get_data_mismatch_per_well(self) -> dict:
        mismatch = self.ctx.per_well_mismatch
        if not mismatch:
            return {"error": "No mismatch data available. Load a project first."}

        sorted_wells = sorted(mismatch.items(), key=lambda x: x[1].get("total", 0), reverse=True)
        above = [w for w, v in mismatch.items() if v.get("status") == "above_expectation"]
        below = [w for w, v in mismatch.items() if v.get("status") == "below_expectation"]
        on_tgt = [w for w, v in mismatch.items() if v.get("status") == "on_target"]

        source = next(iter(mismatch.values())).get("source", "simulation")
        return {
            "per_well":              mismatch,
            "worst_wells":           [w[0] for w in sorted_wells[:5]],
            "best_wells":            [w[0] for w in sorted_wells[-3:]],
            "above_expectation":     above,
            "below_expectation":     below,
            "on_target":             on_tgt,
            "overall_rmse":          self.ctx.overall_rmse,
            "data_source":           source,
            "note": (
                "source='synthetic_baseline' — RMSE values are pre-simulation estimates. "
                "Run history matching to obtain real per-well data mismatch."
                if source == "synthetic_baseline"
                else "Live history matching results."
            ),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _water_cut(well: dict) -> float:
        """Return current water cut fraction (last timestep)."""
        wopr = well.get("wopr", [])
        wwpr = well.get("wwpr", [])
        if not wopr or not wwpr:
            return 0.0
        q_o = wopr[-1]
        q_w = wwpr[-1]
        total = q_o + q_w
        return round(q_w / total, 3) if total > 0 else 0.0

    @staticmethod
    def _cumulative(rates: list[float], dt: float = 100.0) -> float:
        """Approximate cumulative production from rate array (trapezoid rule)."""
        if not rates:
            return 0.0
        total = 0.0
        for i in range(len(rates) - 1):
            total += 0.5 * (rates[i] + rates[i + 1]) * dt
        return round(total, 0)

    def get_field_property(
        self, property: str, i: int, j: int, k: int, timestep: int = -1
    ) -> dict:
        grid = self.ctx.grid_data
        if grid is None:
            return {"error": "No grid data loaded."}

        # Convert to 0-based
        i0, j0, k0 = i - 1, j - 1, k - 1
        prop = property.lower()
        value = None

        try:
            if prop == "perm":
                value = float(grid["perm_x"][i0, j0, k0])
                unit = "mD"
            elif prop == "poro":
                value = float(grid["poro"][i0, j0, k0])
                unit = "fraction"
            elif prop == "pressure" and self.ctx.pressure_field is not None:
                value = float(self.ctx.pressure_field[i0, j0, k0, timestep])
                unit = "psia"
            elif prop == "sw" and self.ctx.sw_field is not None:
                value = float(self.ctx.sw_field[i0, j0, k0, timestep])
                unit = "fraction"
            elif prop == "sg" and self.ctx.sg_field is not None:
                value = float(self.ctx.sg_field[i0, j0, k0, timestep])
                unit = "fraction"
            elif prop == "so" and self.ctx.sw_field is not None and self.ctx.sg_field is not None:
                sw = float(self.ctx.sw_field[i0, j0, k0, timestep])
                sg = float(self.ctx.sg_field[i0, j0, k0, timestep])
                value = 1.0 - sw - sg
                unit = "fraction"
            else:
                return {"error": f"Property '{property}' not available."}
        except IndexError:
            return {"error": f"Grid index ({i},{j},{k}) out of range."}

        return {
            "property": property,
            "i": i, "j": j, "k": k,
            "value": round(value, 4) if value is not None else None,
            "unit": unit,
            "timestep": timestep,
        }

    def explain_parameter(self, parameter: str) -> dict:
        explanations = {
            "alpha": (
                "Regularisation parameter α in αREKI controls the step size of each "
                "Kalman update. Large α → small update (conservative). Small α → "
                "aggressive update. Computed adaptively via the discrepancy principle: "
                "α = N_obs / (2·Φ) where Φ is the current data mismatch squared."
            ),
            "kalman_gain": (
                "The Kalman gain K = Cyd·(CnGG + α·Γ)⁻¹ determines how much the "
                "ensemble is updated based on the data mismatch. Cyd is the cross-"
                "covariance between parameters and predicted data; CnGG is the "
                "predicted data covariance; Γ is the measurement error covariance."
            ),
            "gaspari_cohn": (
                "The Gaspari-Cohn function is a smooth cutoff applied to the Kalman "
                "gain matrix (Schur product / localisation) to suppress spurious "
                "long-range correlations that arise in small ensembles. It goes from "
                "1.0 at distance=0 to 0.0 at distance=2·radius, using a 5th-order "
                "piecewise polynomial for numerical stability."
            ),
            "pino": (
                "PINO (Physics-Informed Neural Operator) is an FNO trained with both "
                "data loss (match OPM FLOW outputs) and PDE residual loss (Darcy flow "
                "equations). This enforces physical consistency between training points, "
                "improving generalisation. Three separate PINOs predict pressure, "
                "water saturation, and gas saturation fields."
            ),
            "fno": (
                "FNO (Fourier Neural Operator) maps input fields (K, φ, wells, time) "
                "to output fields (pressure, Sw, Sg) by lifting to Fourier space, "
                "applying linear transforms to low-frequency modes, and projecting back. "
                "Resolution-independent: trained at one resolution, works at others."
            ),
            "areki": (
                "αREKI (Adaptive Regularised Ensemble Kalman Inversion) is an "
                "ensemble-based iterative method for solving the inverse problem "
                "(history matching). It adaptively adjusts α to ensure convergence "
                "and uses SVD-based inversion for numerical stability. Ensemble "
                "spread quantifies uncertainty."
            ),
            "vcae": (
                "VCAE (Variational Convolutional Autoencoder) compresses the "
                "permeability field K into a low-dimensional latent vector z via "
                "encoder K→(μ,σ)→z. The ELBO loss = reconstruction error + KL "
                "divergence. In αREKI, Kalman updates are applied in latent space "
                "to preserve geological realism."
            ),
            "ddim": (
                "DDIM (Denoising Diffusion Implicit Model) is the decoder that "
                "converts updated latent vectors z back into physically realistic "
                "non-Gaussian permeability fields K. Uses deterministic reverse "
                "diffusion: xt-1 = f(xt, t) without randomness, for reproducibility."
            ),
            "stone_ii": (
                "Stone II model computes three-phase relative permeabilities. "
                "krw = krwmax·((Sw-Swi)/denom)^n; kro is a product of two-phase "
                "terms (oil-water and oil-gas); krg = krgmax·(Sg/denom)^m. "
                "For Norne: Swi=0.1, Sor=0.2, krwmax=0.3, kromax=0.9, krgmax=0.8."
            ),
            "peacemann": (
                "Peacemann (1978) well model: J = 2πKkrDZ/(μB(ln(RE/rw)+skin)). "
                "The productivity index J [STB/day/psia] times drawdown (p-pwf) "
                "gives well rate. For Norne producers: pwf=100 psia, rwell=200 ft, skin=0."
            ),
            "ccr": (
                "CCR (Cluster-Classify-Regress) is a Mixture of Experts surrogate "
                "for the Peacemann well model. Stage 1: cluster ensemble members "
                "(K-means/GMM). Stage 2: XGBoost classifier assigns new inputs to "
                "clusters. Stage 3: per-cluster XGBoost or sparse GP predicts well "
                "rates (WOPR/WWPR/WGPR for all 22 producers = 66 values)."
            ),
            "fault_mult": (
                "Fault transmissibility multipliers (FTM) scale the flow across "
                "fault planes. Values 0–1 (0=sealing, 1=fully open). Norne has 53 "
                "faults. FTMs are uncertain parameters updated in αREKI alongside "
                "permeability and porosity."
            ),
            "pvt": (
                "PVT (Pressure-Volume-Temperature) relations describe how fluids "
                "behave at reservoir conditions. Key properties: Bo (oil swelling/shrinkage), "
                "Bg (gas expansion), Rs (dissolved GOR), μo/μg (viscosities). "
                "All implemented as differentiable PyTorch formulas in PhysicsFlow."
            ),
        }

        p = parameter.lower().replace("-", "_").replace(" ", "_")
        explanation = explanations.get(p, explanations.get(p.split("_")[0]))

        if explanation is None:
            return {
                "parameter": parameter,
                "explanation": f"No specific explanation available for '{parameter}'. "
                               "Please check the PhysicsFlow documentation.",
                "available_parameters": list(explanations.keys()),
            }

        return {"parameter": parameter, "explanation": explanation}

    def query_reservoir_graph(self, question: str) -> dict:
        """
        Query the reservoir knowledge graph for structural/relational facts.
        Returns a structured answer with entity names and supporting data.
        """
        if not _HAS_KG:
            return {"error": "Knowledge graph not available. Install networkx."}
        try:
            kg  = _KGPipeline.instance()
            ans = kg.query(question)
            if not ans.matched:
                return {
                    "message": (
                        "The knowledge graph does not have a pattern for this question. "
                        "Try asking about: well perforations, segment membership, "
                        "injector support, fault boundaries, or converged runs."
                    ),
                    "question": question,
                    "kg_stats": kg.stats(),
                }
            return {
                "answer":     ans.answer,
                "entities":   ans.entities,
                "query_type": ans.query_type,
                "confidence": ans.confidence,
                "data":       ans.data,
            }
        except Exception as e:
            return {"error": f"Knowledge graph query failed: {e}"}

    def search_project_knowledge(
        self,
        query: str,
        source_type: str | None = None,
        top_k: int = 5,
    ) -> dict:
        """
        Hybrid RAG search over indexed project knowledge base.
        Returns relevant text snippets with source citations.
        """
        if not _HAS_RAG:
            return {"error": "RAG pipeline not available. Install chromadb and sentence-transformers."}
        try:
            rag = _RAGPipeline.instance()
            if rag.stats()["vector_chunks"] == 0:
                return {
                    "message": "Knowledge base is empty. Index project files first.",
                    "hint": "Use the document indexer to add PDFs, LAS files, or reports.",
                }
            k = min(max(1, top_k), 10)
            ctx = rag.retrieve_and_build(query, top_k=k, source_type=source_type or None)
            if ctx.chunk_count == 0:
                return {"message": "No relevant documents found for this query.", "query": query}
            return {
                "query":      query,
                "chunks_found": ctx.chunk_count,
                "sources":    ctx.sources,
                "context":    ctx.context_block,
                "citations":  ctx.citations,
            }
        except Exception as e:
            return {"error": f"Knowledge base search failed: {e}"}

    def get_project_summary(self) -> dict:
        return self.ctx.get_project_summary_dict()
