"""
Ollama-powered Reservoir Engineering AI Assistant.

Uses local LLM (phi3:mini, llama3.2:3b, etc.) with tool-calling to answer
reservoir engineering questions grounded in live simulation data.

The agent has access to tools that query:
- Current simulation status and progress
- Well production time series (WOPR, WWPR, WGPR)
- History matching convergence metrics
- Ensemble P10/P50/P90 statistics
- Local reservoir properties (K, φ, pressure, saturations)
- Data mismatch breakdown per well

The system prompt establishes the agent as a senior reservoir engineer
who interprets the data in context of the active project.
"""

from __future__ import annotations
import json
import re
import time
from typing import Generator, Any, Optional
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed. Agent will use mock responses.")

from .tools import ReservoirTools, TOOL_DEFINITIONS
from .context_provider import ReservoirContextProvider

# RAG pipeline — imported lazily to avoid startup cost when not needed
try:
    from physicsflow.rag.pipeline import RAGPipeline
    _HAS_RAG = True
except Exception:
    _HAS_RAG = False
    logger.info("RAG pipeline unavailable (install chromadb + sentence-transformers)")

# Knowledge graph
try:
    from physicsflow.kg.pipeline import KGPipeline
    _HAS_KG = True
except Exception:
    _HAS_KG = False
    logger.info("Knowledge graph unavailable (install networkx)")


SYSTEM_PROMPT = """You are PhysicsFlow Assistant, an expert reservoir engineer embedded in the PhysicsFlow simulation platform.

━━ CRITICAL RULES ━━
1. ALWAYS call a tool FIRST for any question about wells, rates, pressure, mismatch, or simulation results.
   Do NOT describe how to navigate the UI — retrieve the data and report it.
2. Be SHORT and DIRECT. 3-6 bullet points or 2-4 sentences max. No waffle, no preamble.
3. Never show reasoning or thinking steps.
4. Quote actual numbers from tool results or from the Active Project Context below. Use field units (STB/day, psia, mD, %).
5. When a tool returns hm_status="not_started" or n_iterations=0:
   → Report the baseline RMSE and well lists from the result. Do NOT give tutorial steps.
   → Example: "No HM run yet. Baseline RMSE: 0.0821. Above expectation: B-2H, C-4H. Below: D-1H, F-3H. Start αREKI from History Match panel."
6. When a tool returns an error or "not available":
   → One sentence only: state what is missing and which panel starts it. No step-by-step guides.
7. The Active Project Context section below already contains HM status, well performance and mismatch data — use those numbers directly if you cannot call a tool.

━━ TOOL USE — call these for data questions ━━
• "Which wells are above/below expectations?" or "production profiles"
  → call get_well_performance(well_name="all")
  → then call get_data_mismatch_per_well()
  → report above_expectation and below_expectation lists with peak rates and water cut

• "Show [WELL] production" or "how is [WELL] doing?"
  → call get_well_performance(well_name="<well>")
  → report peak WOPR, current water cut, cumulative oil, and status

• "History match status" or "convergence" or "mismatch"
  → call get_hm_status()
  → then call get_data_mismatch_per_well()

• "Training loss" or "surrogate status"
  → call get_simulation_status()

• "Ensemble statistics" or "P10/P50/P90"
  → call get_ensemble_stats()

• "Project summary" or "what's loaded"
  → call get_project_summary()

━━ PhysicsFlow navigation (only answer when user asks HOW to use the app) ━━
1. Dashboard — project overview, well map, surrogate and HM status cards
2. Project (sidebar) — 5-step wizard: Grid → Wells → PVT → Schedule → Save (.pfproj)
3. Train PINO (sidebar) — epochs/LR/PDE weight, Start button, loss curve
4. History Match (sidebar) — ensemble size / localisation radius, Start αREKI
5. Forecast (sidebar) — P10/P50/P90 fan charts, EUR, export to Excel/PDF
6. 3D Viewer — voxel pressure/Sw/K, animated timestep playback
7. Cross Section — I/J/K plane slices, colourmap selector
8. Settings (gear icon) — engine address, Ollama model, default project folder
"""


class ReservoirAgent:
    """
    Ollama-based reservoir engineering assistant with tool-calling.

    Maintains conversation history per session for context continuity.
    Streams response tokens back to the caller for real-time UI updates.
    """

    def __init__(
        self,
        model: str = "phi3:mini",
        context_provider: ReservoirContextProvider | None = None,
        max_tool_calls: int = 5,
        use_rag: bool = True,
        rag_top_k: int = 5,
    ):
        self.model = model
        self.context_provider = context_provider or ReservoirContextProvider()
        self.tools = ReservoirTools(self.context_provider)
        self.max_tool_calls = max_tool_calls
        self.use_rag  = use_rag and _HAS_RAG
        self.rag_top_k = rag_top_k

        # Lazy pipeline references (initialised on first use)
        self._rag: Optional["RAGPipeline"] = None
        self._kg:  Optional["KGPipeline"]  = None

        # Per-session conversation histories
        self._histories: dict[str, list[dict]] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def chat(
        self,
        session_id: str,
        message: str,
        project_path: str | None = None,
    ) -> Generator[dict, None, None]:
        """
        Process a user message and stream back response tokens.

        Yields dicts with keys:
            token        : str — next token to display
            is_tool_call : bool
            tool_name    : str
            tool_result  : str (JSON)
            is_done      : bool
            full_response: str (only on is_done=True)
            chart_data   : dict | None
        """
        if not OLLAMA_AVAILABLE:
            yield from self._mock_response(message)
            return

        # Update project context
        if project_path:
            self.context_provider.set_project(project_path)

        history = self._get_history(session_id)

        # Add system context about current project state
        project_summary = self.context_provider.get_project_summary()

        # ── KG: inject structured graph facts for relational questions ────
        kg_context = ""
        try:
            kg = self._get_kg()
            if kg and kg.is_kg_query(message):
                ans = kg.query(message)
                if ans.matched and ans.answer:
                    kg_context = (
                        "## Knowledge Graph — Structured Reservoir Facts\n"
                        f"{ans.answer}\n"
                        "_Source: PhysicsFlow Reservoir Knowledge Graph_"
                    )
                    logger.debug("KG: injected '%s' answer", ans.query_type)
        except Exception as e:
            logger.warning("KG query failed: %s", e)

        # ── RAG: retrieve relevant knowledge base context ──────────────────
        rag_context = ""
        if self.use_rag:
            try:
                rag = self._get_rag()
                if rag and rag.stats()["vector_chunks"] > 0:
                    ctx = rag.retrieve_and_build(message, top_k=self.rag_top_k)
                    if ctx.chunk_count > 0:
                        rag_context = rag.builder.format_for_prompt(ctx)
                        logger.debug("RAG: injected %d chunks", ctx.chunk_count)
            except Exception as e:
                logger.warning("RAG retrieval failed: %s", e)

        # ── Proactive tool calling ─────────────────────────────────────────
        # Call data tools based on question keywords BEFORE the LLM sees the
        # message. This guarantees real numbers are available even for models
        # that don't support tool-calling or ignore system-prompt instructions.
        proactive_context = self._proactive_tool_context(message)
        if proactive_context:
            logger.debug("Proactive tool context injected (%d chars)", len(proactive_context))

        # ── Direct answer bypass ───────────────────────────────────────────
        # For well-defined data queries, build the formatted answer directly
        # from tool results without involving the LLM.  Small models
        # (phi3:mini, etc.) hallucinate even when the data is injected in the
        # system prompt, so for these questions we skip the model entirely.
        direct_gen = self._try_direct_answer(session_id, message)
        if direct_gen is not None:
            yield from direct_gen
            return

        # Append user message to history (LLM path only)
        history.append({"role": "user", "content": message})

        full_response = ""
        chart_data = None
        tool_calls_made = 0

        # Agentic loop: run until no more tool calls needed
        while tool_calls_made <= self.max_tool_calls:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=self._build_messages(
                        history, project_summary, rag_context, kg_context,
                        proactive_context,
                    ),
                    tools=TOOL_DEFINITIONS,
                    stream=False,   # get full response to check for tool calls
                )
            except Exception as e:
                err = str(e)
                # Some models (phi3:mini, deepseek-r1, etc.) don't support tool-calling.
                # Fall back to plain chat so the user still gets a response.
                if "does not support tools" in err or (
                    "status code: 400" in err and "tool" in err.lower()
                ):
                    logger.warning(
                        "Model '%s' does not support tools — retrying without tool definitions.",
                        self.model,
                    )
                    try:
                        response = ollama.chat(
                            model=self.model,
                            messages=self._build_messages(
                                history, project_summary, rag_context, kg_context,
                                proactive_context,
                            ),
                            stream=False,
                        )
                    except Exception as e2:
                        logger.error(f"Ollama error (no-tools fallback): {e2}")
                        yield {"token": f"\n[Error: {e2}]", "is_done": True, "full_response": str(e2)}
                        return
                else:
                    logger.error(f"Ollama error: {e}")
                    yield {"token": f"\n[Error: {e}]", "is_done": True, "full_response": str(e)}
                    return

            msg = response.message

            # ── Handle tool calls ─────────────────────────────────────────────
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = tc.function.arguments or {}
                    tool_calls_made += 1

                    yield {
                        "token": "",
                        "is_tool_call": True,
                        "tool_name": tool_name,
                        "tool_result": "",
                        "is_done": False,
                    }

                    # Execute tool
                    result = self._call_tool(tool_name, tool_args)
                    result_json = json.dumps(result, indent=2)

                    yield {
                        "token": "",
                        "is_tool_call": True,
                        "tool_name": tool_name,
                        "tool_result": result_json,
                        "is_done": False,
                    }

                    # If result includes chart data, extract it
                    if isinstance(result, dict) and "chart" in result:
                        chart_data = result.pop("chart")

                    # Add tool result to history for next loop
                    history.append({"role": "assistant", "content": None,
                                    "tool_calls": [tc]})
                    history.append({
                        "role": "tool",
                        "content": result_json,
                        "name": tool_name,
                    })
                continue   # Loop back to get next LLM response with tool results

            # ── Stream final text response ─────────────────────────────────────
            final_text = msg.content or ""

            # Strip <think>...</think> blocks produced by reasoning models
            # (deepseek-r1, qwq, etc.) — users should see only the answer.
            final_text = re.sub(r"<think>.*?</think>", "", final_text,
                                flags=re.DOTALL).strip()

            # Stream token by token (simulate streaming for smooth UI)
            words = final_text.split(" ")
            for i, word in enumerate(words):
                token = word + (" " if i < len(words) - 1 else "")
                full_response += token
                yield {
                    "token": token,
                    "is_tool_call": False,
                    "tool_name": "",
                    "tool_result": "",
                    "is_done": False,
                    "chart_data": None,
                }
                time.sleep(0.01)  # slight delay for natural streaming feel

            # Final message
            history.append({"role": "assistant", "content": final_text})
            self._set_history(session_id, history)

            yield {
                "token": "",
                "is_tool_call": False,
                "is_done": True,
                "full_response": full_response,
                "chart_data": chart_data,
            }
            break

    def clear_history(self, session_id: str) -> None:
        self._histories.pop(session_id, None)

    def list_models(self) -> list[str]:
        if not OLLAMA_AVAILABLE:
            return ["phi3:mini (unavailable — install ollama)"]
        try:
            models = ollama.list()
            return [m.model for m in models.models]
        except Exception:
            return []

    def set_model(self, model_name: str) -> bool:
        self.model = model_name
        return True

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> list[dict]:
        if session_id not in self._histories:
            self._histories[session_id] = []
        return self._histories[session_id]

    def _set_history(self, session_id: str, history: list[dict]) -> None:
        # Keep last 20 messages to avoid context overflow
        self._histories[session_id] = history[-20:]

    def _get_kg(self) -> Optional["KGPipeline"]:
        """Return the shared KGPipeline, initialising it lazily."""
        if not _HAS_KG:
            return None
        if self._kg is None:
            try:
                self._kg = KGPipeline.instance()
            except Exception as e:
                logger.warning("KGPipeline init failed: %s", e)
        return self._kg

    @property
    def kg(self) -> Optional["KGPipeline"]:
        """Expose KG pipeline for direct rebuild from outside the agent."""
        return self._get_kg()

    def _get_rag(self) -> Optional["RAGPipeline"]:
        """Return the shared RAGPipeline, initialising it lazily."""
        if not _HAS_RAG:
            return None
        if self._rag is None:
            try:
                self._rag = RAGPipeline.instance(llm_model=self.model)
            except Exception as e:
                logger.warning("RAGPipeline init failed: %s", e)
        return self._rag

    @property
    def rag(self) -> Optional["RAGPipeline"]:
        """Expose RAG pipeline for direct indexing from outside the agent."""
        return self._get_rag()

    def _proactive_tool_context(self, message: str) -> str:
        """
        Keyword-triggered tool calls executed before the LLM sees the message.

        Guarantees real data is present in the context even when the model
        cannot call tools or ignores tool-call instructions.
        Results have chart/time-series data stripped to keep payload small.
        """
        m = message.lower()
        results: list[tuple[str, dict]] = []

        # HM / convergence / mismatch
        if any(k in m for k in [
            "mismatch", "convergence", "areki", "history match",
            "hm status", "hm result", "iteration", "summarise", "summarize",
            "matching poorly", "poorly match", "poorly performing",
        ]):
            try:
                results.append(("get_hm_iteration_summary",
                                 self.tools.get_hm_iteration_summary()))
            except Exception as e:
                logger.warning("Proactive hm_summary failed: %s", e)
            try:
                results.append(("get_data_mismatch_per_well",
                                 self.tools.get_data_mismatch_per_well()))
            except Exception as e:
                logger.warning("Proactive mismatch_per_well failed: %s", e)

        # Well performance / production profiles — broad keyword set
        # "expect" catches expectation/expectations/expected/expecting
        # "above"/"below" catch "above and below expectations" phrasing
        if any(k in m for k in [
            "well", "production", "profile", "wopr", "water cut", "wcut",
            "expect",           # expectation / expectations / expected
            "above", "below",   # "above and below expectations"
            "performing", "performance", "oil rate", "producer", "injector",
            "which well", "show me", "rates",
        ]):
            try:
                results.append(("get_well_performance_all",
                                 self.tools.get_well_performance("all")))
            except Exception as e:
                logger.warning("Proactive well_performance failed: %s", e)
            # Also pull per-well mismatch for above/below questions
            if any(k in m for k in ["expect", "above", "below", "performing",
                                     "mismatch", "match"]):
                try:
                    results.append(("get_data_mismatch_per_well",
                                     self.tools.get_data_mismatch_per_well()))
                except Exception as e:
                    logger.warning("Proactive mismatch_per_well (well block) failed: %s", e)

        # Simulation / training status
        if any(k in m for k in [
            "simulation", "training", "pino", "surrogate", "progress",
            "status", "loss", "epoch",
        ]):
            try:
                results.append(("get_simulation_status",
                                 self.tools.get_simulation_status()))
            except Exception as e:
                logger.warning("Proactive sim_status failed: %s", e)

        if not results:
            return ""

        def _strip(data: dict) -> dict:
            """Remove chart payloads and raw time-series to keep size small."""
            out = {k: v for k, v in data.items() if k not in ("chart",)}
            # For per-well dicts keep only summary fields, not rate arrays
            if "per_well" in out:
                out["per_well"] = {
                    w: {fk: fv for fk, fv in wv.items()
                        if fk not in ("wopr", "wwpr", "wgpr")}
                    for w, wv in out["per_well"].items()
                }
            return out

        lines = [
            "## ── LIVE DATA (answer directly from this — do NOT describe the UI) ──"
        ]
        for name, data in results:
            cleaned = _strip(data) if isinstance(data, dict) else data
            lines.append(f"\n### {name}\n```json\n"
                         f"{json.dumps(cleaned, indent=2)}\n```")
        return "\n".join(lines)

    def _build_messages(
        self,
        history: list[dict],
        project_summary: str,
        rag_context: str = "",
        kg_context: str = "",
        proactive_context: str = "",
    ) -> list[dict]:
        system_with_context = SYSTEM_PROMPT
        if project_summary:
            system_with_context += f"\n\n## Active Project Context\n{project_summary}"
        if kg_context:
            system_with_context += f"\n\n{kg_context}"
        if rag_context:
            system_with_context += f"\n\n{rag_context}"
        if proactive_context:
            # Injected last so it is closest to the user message and hardest to ignore
            system_with_context += f"\n\n{proactive_context}"

        return [
            {"role": "system", "content": system_with_context},
            *history,
        ]

    def _call_tool(self, tool_name: str, args: dict) -> Any:
        """Dispatch tool call to ReservoirTools."""
        tool_fn = getattr(self.tools, tool_name, None)
        if tool_fn is None:
            return {"error": f"Unknown tool: {tool_name}"}
        try:
            return tool_fn(**args)
        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return {"error": str(e)}

    def _try_direct_answer(
        self, session_id: str, message: str
    ) -> "Generator[dict, None, None] | None":
        """
        Bypass the LLM for well-defined data queries.

        Keyword-matches the message, calls relevant tools, builds a formatted
        markdown answer, and returns a streaming Generator.  Returns None if
        the message does not match any data-retrieval pattern (caller then
        falls through to the normal LLM loop).
        """
        m = message.lower()

        is_well_perf = any(k in m for k in [
            "above", "below", "expect",              # above/below expectations
            "production profile", "show me",
            "well",                                  # catches "which wells", "well performance", etc.
            "performing", "performance",
            "oil rate", "water cut", "wcut", "wopr",
            "producer", "best well", "worst well",
            "production data", "rates",
        ])

        is_hm = any(k in m for k in [
            "mismatch", "convergence", "history match",
            "hm status", "hm result", "iteration", "areki",
            "summarise", "summarize",
            "matching poorly", "poorly match",
        ])

        if not (is_well_perf or is_hm):
            return None

        lines: list[str] = []

        if is_well_perf:
            try:
                perf = self.tools.get_well_performance("all")
                mm   = self.tools.get_data_mismatch_per_well()
            except Exception as e:
                logger.warning("Direct answer — well tools failed: %s", e)
                perf, mm = {}, {}
            lines.extend(self._format_well_perf_section(perf, mm))

        if is_hm:
            try:
                hm = self.tools.get_hm_iteration_summary()
            except Exception as e:
                logger.warning("Direct answer — hm_summary failed: %s", e)
                hm = {}
            if lines:
                lines.append("")
            lines.extend(self._format_hm_section(hm))

        if not lines:
            return None

        answer = "\n".join(lines)

        # Save to history so follow-up questions have context
        history = self._get_history(session_id)
        history.append({"role": "user",      "content": message})
        history.append({"role": "assistant", "content": answer})
        self._set_history(session_id, history)

        logger.debug("Direct answer delivered (%d chars), LLM bypassed.", len(answer))
        return self._stream_text(answer)

    def _format_well_perf_section(self, perf: dict, mismatch: dict) -> list[str]:
        """Build markdown lines for well performance / above-below question."""
        if "error" in perf:
            return [f"_{perf['error']}_"]

        well_summary = perf.get("well_summary", {})
        if not well_summary:
            return ["_(No production data available — run a simulation first.)_"]

        source       = perf.get("data_source", "simulation")
        above_exp    = mismatch.get("above_expectation", []) if isinstance(mismatch, dict) else []
        below_exp    = mismatch.get("below_expectation", []) if isinstance(mismatch, dict) else []
        worst_wells  = mismatch.get("worst_wells",        []) if isinstance(mismatch, dict) else []
        per_well_mm  = mismatch.get("per_well",           {}) if isinstance(mismatch, dict) else {}
        overall_rmse = mismatch.get("overall_rmse")           if isinstance(mismatch, dict) else None

        def fmt_well(w: str) -> str:
            d    = well_summary.get(w, {})
            peak = d.get("peak_wopr_stbd")
            wcut = d.get("current_wcut")
            cum  = d.get("cum_oil_stb")
            mm_v = per_well_mm.get(w, {}).get("total") if per_well_mm else None
            parts = [f"**{w}**"]
            if isinstance(peak, (int, float)):
                parts.append(f"peak {peak:,.0f} STB/day")
            if isinstance(wcut, (int, float)):
                parts.append(f"wcut {wcut:.0%}")
            if isinstance(cum, (int, float)):
                parts.append(f"cum {cum:,.0f} STB")
            if isinstance(mm_v, (int, float)):
                parts.append(f"RMSE {mm_v:.3f}")
            return "  • " + " | ".join(parts)

        lines: list[str] = []

        if above_exp:
            lines.append(f"**Above expectation — {len(above_exp)} wells:**")
            for w in above_exp[:12]:
                lines.append(fmt_well(w))

        if below_exp:
            if lines:
                lines.append("")
            lines.append(f"**Below expectation — {len(below_exp)} wells:**")
            for w in below_exp[:12]:
                lines.append(fmt_well(w))

        if not above_exp and not below_exp:
            # No expectation classification — list all producers
            lines.append("**Production summary (all producers):**")
            for w in list(well_summary)[:15]:
                lines.append(fmt_well(w))

        if worst_wells:
            lines.append(f"\n**Highest data mismatch:** {', '.join(worst_wells[:5])}")

        if isinstance(overall_rmse, (int, float)):
            lines.append(f"**Overall RMSE:** {overall_rmse:.4f}")

        src_note = (
            "_Data source: synthetic baseline — run PINO simulation for physics-based results._"
            if source == "synthetic_baseline"
            else "_Data source: live simulation results._"
        )
        lines.append(f"\n{src_note}")
        return lines

    def _format_hm_section(self, hm: dict) -> list[str]:
        """Build markdown lines for history matching / convergence question."""
        if not hm or "error" in hm:
            msg = hm.get("error", "History matching data not available.") if hm else "History matching data not available."
            return [f"_{msg}_"]

        lines: list[str] = []

        if hm.get("hm_status") == "not_started":
            rmse  = hm.get("baseline_rmse", "N/A")
            above = hm.get("wells_above_expectation", [])
            below = hm.get("wells_below_expectation", [])
            lines.append("**History matching:** Not started.")
            lines.append(f"**Baseline RMSE:** {rmse}")
            if above:
                lines.append(f"**Above expectation:** {', '.join(above)}")
            if below:
                lines.append(f"**Below expectation:** {', '.join(below)}")
            lines.append("_Start αREKI from the History Match panel to begin calibration._")
        else:
            n         = hm.get("n_iterations", 0)
            init_mm   = hm.get("initial_mismatch")
            final_mm  = hm.get("final_mismatch")
            impr      = hm.get("improvement_pct", 0)
            converged = hm.get("converged", False)
            conv_str  = "CONVERGED ✓" if converged else "still converging"
            lines.append(f"**History matching:** {n} αREKI iterations — {conv_str}")
            if isinstance(init_mm, float) and isinstance(final_mm, float):
                lines.append(
                    f"**Mismatch:** {init_mm:.4f} → {final_mm:.4f} "
                    f"({impr:.1f}% improvement)"
                )

        return lines

    def _stream_text(self, text: str) -> Generator[dict, None, None]:
        """Stream a pre-built text answer token by token."""
        words    = text.split(" ")
        full     = ""
        for i, word in enumerate(words):
            token  = word + (" " if i < len(words) - 1 else "")
            full  += token
            yield {
                "token":       token,
                "is_tool_call": False,
                "tool_name":   "",
                "tool_result": "",
                "is_done":     False,
                "chart_data":  None,
            }
            time.sleep(0.01)
        yield {
            "token":        "",
            "is_tool_call": False,
            "is_done":      True,
            "full_response": full,
            "chart_data":   None,
        }

    def _mock_response(self, message: str) -> Generator[dict, None, None]:
        """Fallback when Ollama is not installed."""
        response = (
            "Ollama is not installed or not running. Please install it from "
            "https://ollama.com and pull a model: `ollama pull phi3:mini`"
        )
        for word in response.split():
            yield {"token": word + " ", "is_tool_call": False, "is_done": False}
            time.sleep(0.05)
        yield {"token": "", "is_done": True, "full_response": response, "chart_data": None}
