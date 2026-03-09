"""
PhysicsFlow — Database engine and session management.

Single SQLite database file per installation (default: ~/.physicsflow/physicsflow.db).
Path is overridden by the PHYSICSFLOW_DB_PATH environment variable.

Usage:
    from physicsflow.db.database import get_session, init_db

    init_db()                       # creates tables if not exist
    with get_session() as session:  # context manager, auto-commit on exit
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
# Database path resolution
# ─────────────────────────────────────────────────────────────────────────────

def _default_db_path() -> Path:
    """Resolve database path: env var → user data dir → fallback."""
    env_path = os.environ.get("PHYSICSFLOW_DB_PATH")
    if env_path:
        return Path(env_path)
    # Windows: %APPDATA%\PhysicsFlow\physicsflow.db
    # Linux/Mac: ~/.physicsflow/physicsflow.db
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA", Path.home())) / "PhysicsFlow"
    else:
        base = Path.home() / ".physicsflow"
    base.mkdir(parents=True, exist_ok=True)
    return base / "physicsflow.db"


# ─────────────────────────────────────────────────────────────────────────────
# Engine singleton
# ─────────────────────────────────────────────────────────────────────────────

_engine: Engine | None = None
_SessionLocal: sessionmaker | None = None


def _get_engine(db_path: Path | None = None) -> Engine:
    global _engine
    if _engine is None:
        path = db_path or _default_db_path()
        url  = f"sqlite:///{path}"
        log.info("Database: %s", url)
        _engine = create_engine(
            url,
            connect_args={"check_same_thread": False},
            echo=False,          # set True for SQL query logging
            pool_pre_ping=True,
        )
        # Enable WAL mode for concurrent reads while writing
        @event.listens_for(_engine, "connect")
        def _set_sqlite_pragma(conn, _):
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=-64000")    # 64 MB page cache

    return _engine


def _get_session_factory(db_path: Path | None = None) -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        engine = _get_engine(db_path)
        _SessionLocal = sessionmaker(
            bind=engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )
    return _SessionLocal


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def init_db(db_path: Path | None = None) -> Path:
    """
    Create all tables if they do not yet exist.
    Safe to call multiple times (idempotent).
    Returns the database file path.
    """
    engine = _get_engine(db_path)
    Base.metadata.create_all(engine)
    resolved = db_path or _default_db_path()
    log.info("Database initialised at %s", resolved)
    return resolved


@contextmanager
def get_session(db_path: Path | None = None) -> Generator[Session, None, None]:
    """
    Provide a transactional database session.

    Commits on clean exit, rolls back on exception.

    Usage:
        with get_session() as db:
            db.add(some_model)
    """
    factory = _get_session_factory(db_path)
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_stats(db_path: Path | None = None) -> dict:
    """Return row counts for all tables — useful for the UI status panel."""
    engine = _get_engine(db_path)
    tables = [
        "projects", "simulation_runs", "training_epochs",
        "hm_iterations", "well_observations", "model_versions", "audit_log",
    ]
    stats = {}
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                stats[table] = result.scalar()
            except Exception:
                stats[table] = 0
    return stats


def reset_engine() -> None:
    """Force-reset the singleton (used in tests)."""
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
