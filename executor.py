"""
Query Executor
Executes validated SQL queries safely with transaction management.

Public API (matches test_executor.py):
  ExecutionResult   — dataclass: success, rows, columns, error, execution_time_ms
  QueryExecutor     — static methods: execute(), execute_explain(), execute_explain_analyze()
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from db.connection_manager import get_connection

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ExecutionResult:
    """
    Holds the outcome of a single query execution.

    Fields
    ------
    success          : bool             — True when query ran without error
    rows             : list             — Fetched rows (list of tuples)
    columns          : list             — Column names in result order
    error            : str | None       — Error message on failure, else None
    execution_time_ms: float | None     — Wall-clock time in milliseconds
    """
    success: bool
    rows: List[Any]
    columns: List[str]
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None


# ─────────────────────────────────────────────────────────────────────────────
# Executor
# ─────────────────────────────────────────────────────────────────────────────

class QueryExecutor:
    """
    Executes SQL queries through the shared connection manager.

    All methods are *static* — no instance required.
    Every method calls conn.rollback() after execution to enforce
    read-only semantics (even EXPLAIN ANALYZE can touch statistics).
    """

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    @staticmethod
    def _run(sql: str) -> ExecutionResult:
        """
        Core execution routine shared by execute(), execute_explain(),
        and execute_explain_analyze().

        Always rolls back after execution regardless of success/failure.
        """
        start = time.time()
        try:
            with get_connection() as conn:
                try:
                    with conn.cursor() as cursor:
                        cursor.execute(sql)

                        # Extract column names from cursor description
                        if cursor.description is not None:
                            columns = [col[0] for col in cursor.description]
                            rows = list(cursor.fetchall())
                        else:
                            columns = []
                            rows = []

                        elapsed_ms = (time.time() - start) * 1000
                        logger.debug("Query executed in %.2f ms: %s", elapsed_ms, sql[:120])
                        return ExecutionResult(
                            success=True,
                            rows=rows,
                            columns=columns,
                            execution_time_ms=elapsed_ms,
                        )

                except Exception as exc:
                    elapsed_ms = (time.time() - start) * 1000
                    logger.warning("Query execution error: %s", exc)
                    return ExecutionResult(
                        success=False,
                        rows=[],
                        columns=[],
                        error=str(exc),
                        execution_time_ms=elapsed_ms,
                    )
                finally:
                    # Always rollback — we are read-only
                    try:
                        conn.rollback()
                    except Exception:
                        pass

        except Exception as conn_exc:
            elapsed_ms = (time.time() - start) * 1000
            logger.error("Connection error: %s", conn_exc)
            return ExecutionResult(
                success=False,
                rows=[],
                columns=[],
                error=str(conn_exc),
                execution_time_ms=elapsed_ms,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def execute(sql: str) -> ExecutionResult:
        """
        Execute a SQL query and return results.

        Parameters
        ----------
        sql : str
            The SQL query to run.

        Returns
        -------
        ExecutionResult
            success=True with rows/columns on success;
            success=False with error message on failure.
        """
        logger.info("Executing query: %s", sql[:120])
        return QueryExecutor._run(sql)

    @staticmethod
    def execute_explain(sql: str) -> ExecutionResult:
        """
        Run EXPLAIN <sql> and return the plan rows.

        Parameters
        ----------
        sql : str
            Original SQL query (EXPLAIN prefix is added automatically).
        """
        explain_sql = f"EXPLAIN {sql}"
        logger.info("Running EXPLAIN: %s", sql[:120])
        return QueryExecutor._run(explain_sql)

    @staticmethod
    def execute_explain_analyze(sql: str) -> ExecutionResult:
        """
        Run EXPLAIN ANALYZE <sql> and return the plan rows.

        Note: EXPLAIN ANALYZE actually executes the query and can update
        planner statistics, so a rollback is still issued afterwards.

        Parameters
        ----------
        sql : str
            Original SQL query (EXPLAIN ANALYZE prefix is added automatically).
        """
        explain_sql = f"EXPLAIN ANALYZE {sql}"
        logger.info("Running EXPLAIN ANALYZE: %s", sql[:120])
        return QueryExecutor._run(explain_sql)