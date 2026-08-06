"""
db/connection_manager.py
========================
Provides a context-managed PostgreSQL connection.

Public API (matches test_db.py):
  get_connection()         → context manager yielding a psycopg2 connection
  ConnectionError          → re-exported for import convenience

The connection is:
  - set to read-only via conn.set_session(readonly=True, autocommit=False)
  - rolled back and closed on exit (whether normal or exceptional)
"""

import os
import logging
import contextlib

import psycopg2
import psycopg2.extensions

logger = logging.getLogger(__name__)

# Re-export so callers can do:
#   from db.connection_manager import get_connection, ConnectionError as DBConnectionError
ConnectionError = ConnectionError   # built-in; re-exposed for explicit imports


def _get_dsn() -> dict:
    """
    Build connection kwargs from environment variables.

    Variables (all optional — psycopg2 falls back to libpq defaults):
      DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    """
    return dict(
        host    = os.environ.get("DB_HOST",     "localhost"),
        port    = int(os.environ.get("DB_PORT", "5432")),
        dbname  = os.environ.get("DB_NAME",     "optimizer"),
        user    = os.environ.get("DB_USER",     "postgres"),
        password= os.environ.get("DB_PASSWORD", ""),
    )


@contextlib.contextmanager
def get_connection():
    """
    Context manager that yields a psycopg2 connection configured for
    read-only access.

    Usage
    -----
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ...")
            rows = cur.fetchall()

    Guarantees
    ----------
    - Connection is set to read-only before the caller receives it.
    - conn.rollback() is called on exit to discard any implicit transaction.
    - conn.close()    is called on exit regardless of success or failure.

    Raises
    ------
    Any psycopg2 exception if the connection cannot be established.
    """
    conn = psycopg2.connect(**_get_dsn())
    try:
        # Enforce read-only mode
        conn.set_session(readonly=True, autocommit=False)
        logger.debug("DB connection opened (read-only)")
        yield conn
    finally:
        try:
            conn.rollback()
            logger.debug("DB connection rolled back")
        except Exception:
            pass
        try:
            conn.close()
            logger.debug("DB connection closed")
        except Exception:
            pass