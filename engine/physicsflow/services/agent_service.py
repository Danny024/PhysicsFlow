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
import threading
from pathlib import Path
from typing import Iterator

from ..agent.context_provider import ReservoirContextProvider
from ..config import EngineConfig

# Project documentation to auto-index into RAG on first agent use.
# Paths are resolved relative to the project root (two levels above engine/).
_ENGINE_DIR   = Path(__file__).parent.parent.parent          # .../PhysicsFlow/engine/
_PROJECT_ROOT = _ENGINE_DIR.parent                           # .../PhysicsFlow/
_KNOWLEDGE_DOCS: list[Path] = [
    _PROJECT_ROOT / "PhysicsFlow_UserManual_v201.docx",      # full user guide
    _PROJECT_ROOT / "README.md",                             # architecture & API overview
    _PROJECT_ROOT / "CHANGELOG.md",                          # version history & features
    _ENGINE_DIR   / "PhysicsFlow_UserManual_v201.docx",      # fallback if built in engine/
]

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
                context_provider=self.ctx,
            )
            # Index product documentation in the background so the first chat
            # isn't blocked — the RAG context will be available from the second
            # message onward (usually within a few seconds).
            threading.Thread(
                target=self._index_knowledge_base,
                daemon=True,
                name="rag-indexer",
            ).start()
        return self._agent

    def _index_knowledge_base(self) -> None:
        """Index PhysicsFlow documentation into the RAG pipeline."""
        try:
            rag = self._agent.rag  # type: ignore[union-attr]
            if rag is None:
                log.info("RAG pipeline unavailable — skipping knowledge base indexing.")
                return

            # Deduplicate: only index each file once (RAG uses upsert internally)
            indexed_names: set[str] = set()
            for doc_path in _KNOWLEDGE_DOCS:
                if doc_path.name in indexed_names:
                    continue  # skip duplicate fallback paths
                if doc_path.exists():
                    try:
                        rag.index_file(str(doc_path))
                        indexed_names.add(doc_path.name)
                        log.info("RAG: indexed '%s' (%d chunks total)",
                                 doc_path.name, rag.stats().get("vector_chunks", "?"))
                    except Exception as e:
                        log.warning("RAG: failed to index '%s': %s", doc_path.name, e)
                else:
                    log.debug("RAG: doc not found, skipping: %s", doc_path)
        except Exception as e:
            log.warning("Knowledge base indexing failed: %s", e)

    # ── Chat (server-streaming) ───────────────────────────────────────────

    def Chat(self, request, context) -> Iterator:
        """Stream LLM tokens back to the .NET client."""
        if not _PROTO_AVAILABLE:
            return

        agent = self._get_agent()
        session_id = request.session_id or 'default'
        message    = request.message

        log.info("Chat[%s]: %s...", session_id, message[:80])

        project_path = request.context_project or None

        try:
            for chunk in agent.chat(
                session_id=session_id,
                message=message,
                project_path=project_path,
            ):
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
            # ollama SDK >= 0.3 returns an object with .models list of objects (.model attr)
            # older versions return a dict with 'models' list of dicts ('name' key)
            raw = getattr(resp, 'models', None) or resp.get('models', [])
            names = []
            for m in raw:
                name = getattr(m, 'model', None) or m.get('name') or m.get('model', '')
                if name:
                    names.append(name)
        except Exception as exc:
            log.warning("ListModels failed: %s", exc)
            names = [self.cfg.default_llm_model]

        return pb.ListModelsResponse(models=names)

    # ── SetModel ──────────────────────────────────────────────────────────

    def SetModel(self, request, context):
        agent = self._get_agent()
        agent.model = request.model_name
        log.info("Switched to model: %s", request.model_name)
        return pb.SetModelResponse(success=True, message=request.model_name)

    # ── ClearHistory ──────────────────────────────────────────────────────

    def ClearHistory(self, request, context):
        agent = self._get_agent()
        session_id = request.session_id or 'default'
        agent.clear_history(session_id)
        return pb.ClearHistoryResponse(cleared=True)

    # ── GetToolLog ────────────────────────────────────────────────────────

    def GetToolLog(self, request, context):
        agent = self._get_agent()
        entries = agent.tool_log[-50:]   # last 50 tool calls
        return pb.ToolLogResponse(calls=[
            pb.ToolCall(
                tool_name=e.get('tool', ''),
                arguments=str(e.get('args', '')),
                result=str(e.get('result', ''))[:200],
                timestamp=str(e.get('ts', '')),
            )
            for e in entries
        ])
