"""Database connection management."""

import asyncio
from contextlib import contextmanager
from typing import Generator

import structlog
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from recommendation_system.config import get_settings
from recommendation_system.database.models import Base

logger = structlog.get_logger()


class DatabaseManager:
    """Manages database connections and sessions."""

    def __init__(
        self,
        database_url: str | None = None,
        pool_size: int = 10,
        max_overflow: int = 20,
        pool_timeout: int = 30,
        pool_recycle: int = 1800,
        echo: bool = False,
    ):
        """Initialize database manager.

        Args:
            database_url: Database connection URL. Supports PostgreSQL, SQLite, MySQL.
            pool_size: Number of connections in the pool.
            max_overflow: Max connections above pool_size.
            pool_timeout: Timeout for getting connection from pool.
            pool_recycle: Recycle connections after this many seconds.
            echo: Log all SQL statements.
        """
        settings = get_settings()
        self.database_url = database_url or settings.database_url

        # Determine if SQLite (for local dev) or production DB
        self.is_sqlite = self.database_url.startswith("sqlite")

        engine_kwargs = {
            "echo": echo or settings.debug,
            "pool_pre_ping": True,  # Verify connections before use
        }

        if not self.is_sqlite:
            # Production pool settings for PostgreSQL/MySQL
            engine_kwargs.update({
                "poolclass": QueuePool,
                "pool_size": pool_size,
                "max_overflow": max_overflow,
                "pool_timeout": pool_timeout,
                "pool_recycle": pool_recycle,
            })
        else:
            # SQLite settings
            engine_kwargs["connect_args"] = {"check_same_thread": False}

        self.engine = create_engine(self.database_url, **engine_kwargs)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Set up SQLite optimizations
        if self.is_sqlite:
            self._setup_sqlite_optimizations()

        logger.info(
            "Database manager initialized",
            database_type="sqlite" if self.is_sqlite else "postgresql",
            pool_size=pool_size if not self.is_sqlite else "N/A",
        )

    def _setup_sqlite_optimizations(self):
        """Set up SQLite performance optimizations."""
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA cache_size=10000")
            cursor.execute("PRAGMA temp_store=MEMORY")
            cursor.close()

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all database tables. USE WITH CAUTION."""
        Base.metadata.drop_all(bind=self.engine)
        logger.warning("Database tables dropped")

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session with automatic cleanup."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_session_factory(self) -> sessionmaker:
        """Get the session factory for dependency injection."""
        return self.SessionLocal

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            with self.get_session() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error("Database health check failed", error=str(e))
            return False

    def get_stats(self) -> dict:
        """Get database connection pool statistics."""
        if self.is_sqlite:
            return {"type": "sqlite", "status": "connected"}

        pool = self.engine.pool
        return {
            "type": "postgresql",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalidatedcount() if hasattr(pool, "invalidatedcount") else 0,
        }


# Global database manager instance
_db_manager: DatabaseManager | None = None


def get_db_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def init_database(database_url: str | None = None, create_tables: bool = True) -> DatabaseManager:
    """Initialize the database manager and optionally create tables."""
    global _db_manager
    _db_manager = DatabaseManager(database_url=database_url)
    if create_tables:
        _db_manager.create_tables()
    return _db_manager


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions."""
    db_manager = get_db_manager()
    session = db_manager.SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def reset_database():
    """Reset the database manager (for testing)."""
    global _db_manager
    _db_manager = None
