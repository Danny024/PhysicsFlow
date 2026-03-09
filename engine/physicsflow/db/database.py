"""
PhysicsFlow — Database engine and session management.

Supports two backends selected by PHYSICSFLOW_DB_URL environment variable:

  SQLite   (default, single-user):
      sqlite:///path/to/physicsflow.db
      Falls back to %APPDATA%\\PhysicsFlow\\physicsflow.db on Windows
      or ~/.physicsflow/physicsflow.db on Linux/macOS.

  PostgreSQL (team / on-premise server):
      postgresql+psycopg2://user:password@host:5432/physicsflow

Usage:
    from physicsflow.db.database import get_session, init_db

    init_db()                       # creates tables if not exist
    with get_session() as session:  # auto-commit on clean exit
        session.add(...)
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker, Session

from .models import Base

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# URL resolution
# ─────────────────────────────────────────────────────────────────────────────

def _default_sqlite_path() -> Path:
    """Resolve SQLite path: env var → OS app-data dir."""
    env_path = os.environ.get("PHYSICSFLOW_DB_PATH")
    if env_path:
        return Path(env_path)
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / "PhysicsFlow"
    else:
        base = Path.home() / ".physicsflow"
    base.mkdir(parents=True, exist_ok=True)
    return base / "physicsflow.db"


def _resolve_db_url() -> str:
    """
    Determine the active database URL in priority order:
      1. PHYSICSFLOW_DB_URL environment variable
      2. config.db_url (if non-empty)
      3. PHYSICSFLOW_DB_PATH  → sqlite:///path
      4. OS default path      → sqlite:///~/.physicsflow/physicsflow.db
    """
    env_url = os.environ.get("PHYSICSFLOW_DB_URL", "")
    if env_url:
        return env_url
    try:
        from ..config import config as _cfg
        if _cfg.db_url:
            return _cfg.db_url
    except Exception:
        pass
    return f"sqlite:///{_default_sqlite_path()}"


# ─────────────────────────────────────────────────────────────────────────────
# Engine singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _build_engine(url: str) -> Engine:
    """Create a SQLAlchemy engine appropriate for the given URL."""
    if url.startswith("sqlite"):
        engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,
            pool_pre_ping=True,
        )

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")   # 64 MB page cache

        log.info("Database backend: SQLite  →  %s", url)

    elif url.startswith("postgresql"):
        engine = create_engine(
            url,
            echo=False,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=1800,    # recycle idle connections every 30 min
        )
        log.info("Database backend: PostgreSQL  →  %s",
                 url.split("@")[-1])   # hide credentials in log

    else:
        raise ValueError(
            f"Unsupported database URL: {url!r}. "
            "Use sqlite:/// or postgresql+psycopg2://"
        )

    return engine


def _get_engine() -> Engine:
    global _engine
    if _engine is None:
        _engine = _build_engine(_resolve_db_url())
    return _engine


def _get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(
            bind=_get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db() -> str:
    """
    Create all tables if they do not yet exist.
    Safe to call multiple times (idempotent).
    Returns the active database URL (credentials redacted).
    """
    engine = _get_engine()
    Base.metadata.create_all(engine)
    url = _resolve_db_url()
    redacted = url.split("@")[-1] if "@" in url else url
    log.info("Database initialised: %s", redacted)
    return redacted


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional database session.
    Commits on clean exit, rolls back on any exception.

    Usage:
        with get_session() as db:
            db.add(some_model)
    """
    factory = _get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_stats() -> dict:
    """Return row counts for all tables — used by the health endpoint."""
    engine = _get_engine()
    tables = [
        "projects", "simulation_runs", "training_epochs",
        "hm_iterations", "well_observations", "model_versions", "audit_log",
    ]
    stats: dict = {}
    with engine.connect() as conn:
        for table in tables:
            try:
                stats[table] = conn.execute(
                    text(f"SELECT COUNT(*) FROM {table}")
                ).scalar()
            except Exception:
                stats[table] = 0
    return stats


def db_backend() -> str:
    """Return 'sqlite' or 'postgresql'."""
    return "postgresql" if _resolve_db_url().startswith("postgresql") else "sqlite"


def reset_engine() -> None:
    """Force-reset the engine singleton (used in tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
