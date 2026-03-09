"""
/api/v1/projects — Project CRUD and audit log.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from ..auth import require_api_key
from ..schemas import (
    AuditLogSchema, ProjectCreateRequest, ProjectListResponse,
    ProjectSchema, ProjectUpdateRequest, StatusResponse,
)

router = APIRouter(prefix="/projects", tags=["projects"])
_AUTH = [Depends(require_api_key)]


@router.get("", response_model=ProjectListResponse, dependencies=_AUTH)
async def list_projects(
    request: Request,
    limit: int = Query(default=20, le=200),
    offset: int = Query(default=0, ge=0),
):
    """List all registered projects, most recently modified first."""
    db_svc = request.app.state.db_svc
    projects = db_svc.list_projects(limit=limit, offset=offset)
    return ProjectListResponse(
        projects=[ProjectSchema.model_validate(p) for p in projects],
        total=len(projects),
    )


@router.post("", response_model=ProjectSchema,
             status_code=status.HTTP_201_CREATED, dependencies=_AUTH)
async def create_project(request: Request, body: ProjectCreateRequest):
    """Register a new project in the database."""
    db_svc = request.app.state.db_svc
    project = db_svc.register_project_from_dict(body.model_dump())
    return ProjectSchema.model_validate(project)


@router.get("/{project_id}", response_model=ProjectSchema, dependencies=_AUTH)
async def get_project(request: Request, project_id: str):
    """Get a single project by ID."""
    db_svc = request.app.state.db_svc
    project = db_svc.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    return ProjectSchema.model_validate(project)


@router.put("/{project_id}", response_model=ProjectSchema, dependencies=_AUTH)
async def update_project(
    request: Request, project_id: str, body: ProjectUpdateRequest,
):
    """Update project name or notes."""
    db_svc = request.app.state.db_svc
    project = db_svc.update_project(
        project_id,
        **{k: v for k, v in body.model_dump().items() if v is not None},
    )
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    return ProjectSchema.model_validate(project)


@router.delete("/{project_id}", response_model=StatusResponse, dependencies=_AUTH)
async def delete_project(request: Request, project_id: str):
    """Delete a project and all its child records."""
    db_svc = request.app.state.db_svc
    ok = db_svc.delete_project(project_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")
    return StatusResponse(status="deleted", message=f"Project {project_id} deleted.")


@router.get("/{project_id}/summary", dependencies=_AUTH)
async def project_summary(request: Request, project_id: str):
    """
    Return a rich summary combining DB metadata and live context provider state.
    This is the endpoint the Jupyter integration uses to display a project dashboard.
    """
    db_svc  = request.app.state.db_svc
    context = request.app.state.context

    project = db_svc.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project {project_id!r} not found.")

    ctx_summary = context.get_summary() if hasattr(context, "get_summary") else {}

    return {
        "project": ProjectSchema.model_validate(project).model_dump(),
        "live_context": ctx_summary,
    }


@router.get("/{project_id}/audit", response_model=list[AuditLogSchema],
            dependencies=_AUTH)
async def project_audit(
    request: Request,
    project_id: str,
    limit: int = Query(default=50, le=500),
):
    """Return recent audit log entries for a project."""
    db_svc = request.app.state.db_svc
    entries = db_svc.get_audit_log(project_id=project_id, limit=limit)
    return [AuditLogSchema.model_validate(e) for e in entries]
