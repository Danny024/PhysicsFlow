"""
/api/v1/agent — AI Reservoir Assistant via REST.

Exposes the same ReservoirAgent used by the gRPC AgentService, giving
Jupyter notebooks and scripts direct access to the 3-layer intelligence
(Live tools + Knowledge Graph + Hybrid RAG).

Two chat modes:
  POST /chat         — synchronous single-turn, returns full response
  POST /chat/stream  — Server-Sent Events streaming tokens (for web UIs)
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from ..auth import require_api_key
from ..schemas import ChatRequest, ChatResponse

router = APIRouter(prefix="/agent", tags=["agent"])
_AUTH = [Depends(require_api_key)]


def _get_agent(request: Request):
    """Lazy-init the ReservoirAgent (shares context with gRPC path)."""
    if not hasattr(request.app.state, "_agent") or request.app.state._agent is None:
        from physicsflow.agent.reservoir_agent import ReservoirAgent
        cfg     = request.app.state.cfg
        context = request.app.state.context
        request.app.state._agent = ReservoirAgent(
            model=cfg.default_llm_model,
            context_provider=context,
        )
    return request.app.state._agent


@router.post("/chat", response_model=ChatResponse, dependencies=_AUTH)
async def chat(request: Request, body: ChatRequest):
    """
    Single-turn synchronous chat.  Collects the full streamed response
    before returning — suitable for Jupyter notebooks.
    """
    agent = _get_agent(request)
    full_response = ""
    tool_calls: list[dict] = []

    for chunk in agent.chat(session_id=body.session_id, message=body.message):
        if chunk.get("is_done"):
            full_response = chunk.get("full_response", full_response)
        elif chunk.get("is_tool_call"):
            tool_calls.append({
                "tool":   chunk.get("tool_name", ""),
                "result": chunk.get("tool_result", ""),
            })
        else:
            full_response += chunk.get("token", "")

    return ChatResponse(
        response=full_response,
        session_id=body.session_id,
        tool_calls=tool_calls,
    )


@router.post("/chat/stream", dependencies=_AUTH)
async def chat_stream(request: Request, body: ChatRequest):
    """
    Server-Sent Events streaming chat.
    Each event is a JSON object: {"token": "...", "is_done": false, ...}

    JavaScript / Python httpx streaming clients can consume this directly.
    """
    agent = _get_agent(request)

    def _generate():
        for chunk in agent.chat(session_id=body.session_id, message=body.message):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/context", dependencies=_AUTH)
async def agent_context(request: Request):
    """
    Return a JSON summary of everything the AI assistant currently knows:
    live simulation state, last HM iteration, and project metadata.
    Useful for building a status dashboard in Jupyter.
    """
    context = request.app.state.context
    if hasattr(context, "get_summary"):
        return context.get_summary()
    # Graceful fallback for partial context implementations
    return {
        "simulation_status": getattr(context, "simulation_status", "unknown"),
        "hm_active":         getattr(context, "hm_active", False),
        "training_active":   getattr(context, "training_active", False),
    }


@router.get("/models", dependencies=_AUTH)
async def list_ollama_models(request: Request):
    """List available Ollama models on this server."""
    cfg = request.app.state.cfg
    try:
        import ollama
        resp  = ollama.Client(host=cfg.ollama_host).list()
        names = [m["name"] for m in resp.get("models", [])]
    except Exception:
        names = [cfg.default_llm_model]
    return {"models": names, "active": cfg.default_llm_model}
