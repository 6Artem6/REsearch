"""DB engine и session factory."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from knowledge_engine.config import DATABASE_URL
from knowledge_engine.db.base import Base
from knowledge_engine.models.article_diagrams import ArticleDiagram  # noqa: F401
from knowledge_engine.models.figure_registry import FigureRegistryRow  # noqa: F401


def _engine():
    return create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        future=True,
    )


_engine_instance = None
_SessionLocal: sessionmaker[Session] | None = None


def get_session_factory() -> sessionmaker[Session]:
    global _engine_instance, _SessionLocal
    if _SessionLocal is None:
        _engine_instance = _engine()
        _SessionLocal = sessionmaker(
            bind=_engine_instance,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )
    return _SessionLocal


def init_db() -> None:
    from knowledge_engine.config import PACKAGE_ROOT

    (PACKAGE_ROOT / ".runs").mkdir(parents=True, exist_ok=True)
    engine = _engine()
    Base.metadata.create_all(bind=engine)


@contextmanager
def db_session() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
