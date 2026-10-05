from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

from app.config import get_settings


class Database:
    

    def __init__(
        self,
        database_url: str | None = None,
    ) -> None:
        settings = get_settings()

        self.database_url = database_url or settings.postgres_dsn

        self.engine: Engine = create_engine(
            self.database_url,
            pool_pre_ping=True,
            pool_recycle=1800,
        )

    @contextmanager
    def connection(self) -> Iterator[Connection]:
        """
        Provide a managed database connection.

        The connection is automatically closed after the operation.
        Transactions are handled explicitly by callers when required.
        """

        with self.engine.connect() as connection:
            yield connection

    def health_check(self) -> bool:
        """
        Verify that PostgreSQL is reachable.

        Returns
        -------
        bool
            True when the database responds successfully.
        """

        try:
            with self.connection() as connection:
                connection.execute(text("SELECT 1"))

            return True

        except Exception:
            return False

    def close(self) -> None:
        """Dispose of the SQLAlchemy connection pool."""

        self.engine.dispose()


_database: Database | None = None


def get_database() -> Database:
    """
    Return the application-wide database instance.

    Lazy initialization prevents database connections from being created
    merely by importing the module.
    """

    global _database

    if _database is None:
        _database = Database()

    return _database