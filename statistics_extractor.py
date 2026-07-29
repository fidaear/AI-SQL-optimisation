"""
statistics_extractor.py
=======================
Extracts PostgreSQL table and index statistics.

Public API (matches test_plan_parser.py):

  StatisticsExtractor.get_table_stats(tables)           → dict
  StatisticsExtractor.get_index_stats(tables)           → dict
  StatisticsExtractor.get_comprehensive_statistics(tables_or_session, ...) → dict

All three are @staticmethod so tests can call them as
  StatisticsExtractor.get_table_stats([...])
without instantiating the class.

The old instance-based API (extract_reltuples / extract_n_distinct /
detect_existing_indexes) is preserved for backward-compatibility with the
orchestrator, which passes (session_id, table_name).
"""

import logging
from typing import Any, Dict, List, Optional, Union

from db.connection_manager import get_connection

logger = logging.getLogger(__name__)


class StatisticsExtractor:
    """
    Pulls statistics from PostgreSQL system catalogs.

    Static methods are the primary, test-friendly API.
    Instance methods delegate to them (backward-compatibility).
    """

    # ─────────────────────────────────────────────────────────────────────────
    # Static helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _rows_to_dicts(cursor) -> List[Dict[str, Any]]:
        """Convert cursor rows to list-of-dicts using column names from description."""
        if not cursor.description:
            return []
        cols = [col[0] for col in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC STATIC API  (called directly by tests)
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_table_stats(tables: List[str]) -> Dict[str, Any]:
        """
        Return row-count and page statistics for the given table list.

        Parameters
        ----------
        tables : list[str]
            Table names to query.  Pass [] to get an empty-but-valid result.

        Returns
        -------
        dict
            {
              "success": bool,
              "reltuples": {
                  "data": [{"table_name": ..., "reltuples": ..., ...}, ...]
              },
              "error": str   # only present on failure
            }
        """
        if not tables:
            return {
                "success": True,
                "reltuples": {"data": [], "summary": {"total_tables": 0, "total_rows": 0}},
            }

        placeholders = ", ".join(["%s"] * len(tables))
        query = f"""
            SELECT
                c.relname          AS table_name,
                n.nspname          AS schema,
                c.reltuples::BIGINT AS reltuples,
                c.relpages::BIGINT  AS relpages,
                COALESCE(s.n_dead_tup, 0) AS n_dead_tup
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            LEFT JOIN pg_stat_user_tables s ON s.relname = c.relname
            WHERE c.relname IN ({placeholders})
              AND c.relkind = 'r'
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY c.reltuples DESC;
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, tables)
                    data = StatisticsExtractor._rows_to_dicts(cursor)

            total_rows = sum(int(r.get("reltuples", 0)) for r in data)
            return {
                "success": True,
                "reltuples": {
                    "data": data,
                    "summary": {
                        "total_tables": len(data),
                        "total_rows": total_rows,
                    },
                },
            }

        except Exception as exc:
            logger.error("get_table_stats failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "reltuples": None,
            }

    @staticmethod
    def get_index_stats(tables: List[str]) -> Dict[str, Any]:
        """
        Return index information for the given table list.

        Parameters
        ----------
        tables : list[str]

        Returns
        -------
        dict
            {
              "success": bool,
              "indexes": {
                  "data": [{"table_name": ..., "index_name": ..., ...}, ...]
              }
            }
        """
        if not tables:
            return {
                "success": True,
                "indexes": {"data": [], "summary": {"total_indexes": 0}},
            }

        placeholders = ", ".join(["%s"] * len(tables))
        query = f"""
            SELECT
                i.tablename        AS table_name,
                i.indexname        AS index_name,
                a.attname          AS column_name,
                COALESCE(s.idx_scan, 0) AS idx_scan
            FROM pg_indexes i
            JOIN pg_class c  ON c.relname = i.indexname
            JOIN pg_index ix ON ix.indexrelid = c.oid
            JOIN pg_attribute a
                ON a.attrelid = ix.indrelid
               AND a.attnum   = ANY(ix.indkey)
            LEFT JOIN pg_stat_user_indexes s ON s.indexrelname = i.indexname
            WHERE i.tablename IN ({placeholders})
              AND i.schemaname NOT IN ('pg_catalog', 'information_schema')
            ORDER BY i.tablename, i.indexname;
        """

        try:
            with get_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, tables)
                    data = StatisticsExtractor._rows_to_dicts(cursor)

            return {
                "success": True,
                "indexes": {
                    "data": data,
                    "summary": {"total_indexes": len(data)},
                },
            }

        except Exception as exc:
            logger.error("get_index_stats failed: %s", exc)
            return {
                "success": False,
                "error": str(exc),
                "indexes": {"data": [], "summary": {"total_indexes": 0}},
            }

    @staticmethod
    def get_comprehensive_statistics(
        tables_or_session: Union[List[str], str],
        table_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Merge table stats and index stats into one dict.

        Accepts two call signatures:

        1. New API (used by tests):
               get_comprehensive_statistics(["orders", "customers"])

        2. Old API (used by orchestrator):
               get_comprehensive_statistics(session_id: str, table_name=None)
           → session_id is ignored; table_name used as a single-element list.

        Returns
        -------
        dict  with keys: success, reltuples, indexes
        """
        # ── Resolve table list from either call signature ─────────────────────
        if isinstance(tables_or_session, list):
            tables = tables_or_session
        else:
            # Old orchestrator signature: first arg is session_id (str)
            tables = [table_name] if table_name else []

        # ── Fetch both stat groups ────────────────────────────────────────────
        table_result = StatisticsExtractor.get_table_stats(tables)
        try:
            index_result = StatisticsExtractor.get_index_stats(tables)
        except Exception as exc:
            logger.warning("get_index_stats failed in comprehensive call: %s", exc)
            index_result = {
                "success": False,
                "error": str(exc),
                "indexes": {"data": [], "summary": {"total_indexes": 0}},
            }

        overall_success = table_result.get("success", False) or index_result.get("success", False)

        return {
            "success": overall_success,
            "reltuples": table_result.get("reltuples"),
            "indexes":   index_result.get("indexes"),
            # Preserve n_distinct key so orchestrator summary logging works
            "n_distinct": {"data": []},
        }

    # ─────────────────────────────────────────────────────────────────────────
    # BACKWARD-COMPATIBLE INSTANCE METHODS (used by old orchestrator)
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(self, session_manager=None, connection_timeout: int = 10):
        self.session_manager    = session_manager
        self.connection_timeout = connection_timeout

    def extract_reltuples(self, session_id: str, table_name: Optional[str] = None) -> Dict:
        tables = [table_name] if table_name else []
        return StatisticsExtractor.get_table_stats(tables)

    def extract_n_distinct(self, session_id: str, table_name: Optional[str] = None) -> Dict:
        return {"success": True, "data": [], "n_distinct": {"data": []}}

    def detect_existing_indexes(self, session_id: str, table_name: Optional[str] = None) -> Dict:
        tables = [table_name] if table_name else []
        return StatisticsExtractor.get_index_stats(tables)