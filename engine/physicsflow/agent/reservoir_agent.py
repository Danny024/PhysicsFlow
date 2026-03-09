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
import time
from typing import Generator, Any
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logger.warning("ollama package not installed. Agent will use mock responses.")

from .tools import ReservoirTools, TOOL_DEFINITIONS
from .context_provider import ReservoirContextProvider


SYSTEM_PROMPT = """You are PhysicsFlow Assistant, an expert reservoir engineering AI
embedded in the PhysicsFlow reservoir simulation platform.

You help petroleum engineers interpret simulation results, understand history
matching outcomes, and make decisions about reservoir management.

Your expertise covers:
- Black-oil and compositional reservoir simulation
- Physics-Informed Neural Operator (PINO) surrogate modelling
- Ensemble-based history matching (αREKI, ES-MDA)
- Well performance analysis (Peacemann model, productivity index)
- Uncertainty quantification (P10/P50/P90 ensemble statistics)
- Relative permeability, PVT correlations, and rock physics
- Darcy flow, transmissibility, and reservoir connectivity

When answering:
1. Always ground your answers in the actual simulation data from tool calls
2. Use field units (psia, STB/day, mD, ft) unless the user asks for metric
3. Be concise but technically precise — engineers value accuracy over verbosity
4. When results look unusual, flag potential issues proactively
5. Suggest actionable next steps where appropriate
6. If you don't have enough data, say so clearly and suggest what to run

You have access to live project data through tool functions.
Always call the relevant tool before answering quantitative questions.
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
    ):
        self.model = model
        self.context_provider = context_provider or ReservoirContextProvider()
        self.tools = ReservoirTools(self.context_provider)
        self.max_tool_calls = max_tool_calls

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

        # Append user message
        history.append({"role": "user", "content": message})

        full_response = ""
        chart_data = None
        tool_calls_made = 0

        # Agentic loop: run until no more tool calls needed
        while tool_calls_made <= self.max_tool_calls:
            try:
                response = ollama.chat(
                    model=self.model,
                    messages=self._build_messages(history, project_summary),
                    tools=TOOL_DEFINITIONS,
                    stream=False,   # get full response to check for tool calls
                )
            except Exception as e:
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

    def _build_messages(self, history: list[dict], project_summary: str) -> list[dict]:
        system_with_context = SYSTEM_PROMPT
        if project_summary:
            system_with_context += f"\n\n## Active Project Context\n{project_summary}"

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
