"""Connection pool, migrations and per-source locking."""

import contextlib
import importlib.resources
import logging

import psycopg
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .config import settings

log = logging.getLogger(__name__)

# Arbitrary, fixed: serialises migrations across replicas and CronJobs.
_MIGRATION_LOCK = 0x5E3A_0001

_pool: ConnectionPool | None = None


def _configure(conn: psycopg.Connection) -> None:
    register_vector(conn)
    conn.row_factory = dict_row


def pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            settings().database_url,
            min_size=1,
            max_size=8,
            configure=_configure,
            open=True,
        )
    return _pool


def close() -> None:
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def connect() -> psycopg.Connection:
    """A dedicated connection, for work that holds a session-level lock."""
    conn = psycopg.connect(settings().database_url)
    _configure(conn)
    return conn


def migrate() -> None:
    """Apply numbered SQL files in migrations/ that have not run yet.

    Runs on a plain connection, before register_vector: the first migration
    is the one that creates the extension register_vector looks up.
    """
    files = sorted(
        f for f in importlib.resources.files(__package__).joinpath("migrations").iterdir()
        if f.name.endswith(".sql")
    )
    with psycopg.connect(settings().database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK,))
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                " name text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
            done = {r[0] for r in conn.execute("SELECT name FROM schema_migrations")}
            for f in files:
                if f.name in done:
                    continue
                log.info("applying migration %s", f.name)
                with conn.transaction():
                    conn.execute(f.read_text())
                    conn.execute("INSERT INTO schema_migrations (name) VALUES (%s)", (f.name,))
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK,))


@contextlib.contextmanager
def source_lock(source: str):
    """Hold the per-source reconcile lock for the duration of the block.

    Blocking on purpose. A webhook that arrives while the CronJob is mid-run
    must still get a run of its own afterwards -- the change it announces may
    have landed after the CronJob listed the source -- so it waits rather
    than skipping.
    """
    with connect() as conn:
        conn.autocommit = True
        conn.execute("SELECT pg_advisory_lock(hashtext('reconcile:' || %s))", (source,))
        try:
            yield
        finally:
            conn.execute("SELECT pg_advisory_unlock(hashtext('reconcile:' || %s))", (source,))
