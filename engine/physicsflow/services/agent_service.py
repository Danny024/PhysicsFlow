"""
PhysicsFlow — AgentService gRPC handler.

Implements:
    Chat        (server-streaming) — token-by-token LLM response
    ListModels  — available Ollama models
    SetModel    — switch active model
    ClearHistory — wipe session history
    GetToolLog  — return recent tool calls
"""

from __future__ import annotations

import logging
from typing import Iterator

from ..agent.context_provider import ReservoirContextProvider
from ..config import EngineConfig

log = logging.getLogger(__name__)

try:
    from ..proto import agent_pb2 as pb
    from ..proto import agent_pb2_grpc as pbg
    _PROTO_AVAILABLE = True
except ImportError:
    _PROTO_AVAILABLE = False
    pb = None
    pbg = None


class AgentServicer:
    """
    gRPC AgentService — bridges the .NET AI panel to the Python LLM agent.
    """

    def __init__(self, cfg: EngineConfig, context: ReservoirContextProvider):
        self.cfg = cfg
        self.ctx = context
        self._agent = None

    def _get_agent(self):
        if self._agent is None:
            from ..agent.reservoir_agent import ReservoirAgent
            self._agent = ReservoirAgent(
                model=self.cfg.default_llm_model,
                ollama_host=self.cfg.ollama_host,
                context=self.ctx,
            )
        return self._agent

    # ── Chat (server-streaming) ───────────────────────────────────────────

    def Chat(self, request, context) -> Iterator:
        """Stream LLM tokens back to the .NET client."""
        if not _PROTO_AVAILABLE:
            return

        agent = self._get_agent()
        session_id = request.session_id or 'default'
        message    = request.message

        log.info("Chat[%s]: %s...", session_id, message[:80])

        try:
            for chunk in agent.chat(session_id=session_id, message=message):
                token     = chunk.get('token', '')
                is_tool   = chunk.get('is_tool_call', False)
                is_done   = chunk.get('is_done', False)
                tool_name = chunk.get('tool_name', '')
                tool_res  = chunk.get('tool_result', '')
                full_resp = chunk.get('full_response', '')
                chart_raw = chunk.get('chart_data')

                # Map chart data
                chart_proto = None
                if chart_raw and _PROTO_AVAILABLE:
                    chart_proto = pb.ChartData(
                        chart_type=chart_raw.get('type', 'line'),
                        title=chart_raw.get('title', ''),
                        series=[
                            pb.ChartSeries(
                                name=s.get('name', ''),
                                x_values=s.get('x', []),
                                y_values=s.get('y', []),
                            )
                            for s in chart_raw.get('series', [])
                        ],
                    )

                yield pb.ChatToken(
                    token=token,
                    is_tool_call=is_tool,
                    tool_name=tool_name,
                    tool_result=str(tool_res),
                    is_done=is_done,
                    full_response=full_resp,
                    chart=chart_proto,
                )

        except Exception as exc:
            log.exception("Chat failed: %s", exc)
            yield pb.ChatToken(
                token=f"\n[Error: {exc}]",
                is_done=True,
            )

    # ── ListModels ────────────────────────────────────────────────────────

    def ListModels(self, request, context):
        try:
            import ollama
            resp = ollama.Client(host=self.cfg.ollama_host).list()
            names = [m['name'] for m in resp.get('models', [])]
        except Exception:
            names = [self.cfg.default_llm_model]

        return pb.ListModelsResponse(model_names=names)

    # ── SetModel ──────────────────────────────────────────────────────────

    def SetModel(self, request, context):
        agent = self._get_agent()
        agent.model = request.model_name
        log.info("Switched to model: %s", request.model_name)
        return pb.SetModelResponse(success=True, active_model=request.model_name)

    # ── ClearHistory ──────────────────────────────────────────────────────

    def ClearHistory(self, request, context):
        agent = self._get_agent()
        session_id = request.session_id or 'default'
        agent.clear_history(session_id)
        return pb.ClearHistoryResponse(success=True)

    # ── GetToolLog ────────────────────────────────────────────────────────

    def GetToolLog(self, request, context):
        agent = self._get_agent()
        entries = agent.tool_log[-50:]   # last 50 tool calls
        return pb.ToolLogResponse(entries=[
            pb.ToolLogEntry(
                tool_name=e.get('tool', ''),
                timestamp=str(e.get('ts', '')),
                result_summary=str(e.get('result', ''))[:200],
            )
            for e in entries
        ])
