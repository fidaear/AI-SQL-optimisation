"""
db/schema_service.py
====================
Reusable PostgreSQL schema metadata extractor.

This is the SINGLE SOURCE OF TRUTH for "what does the connected database look like".
It is used by:
  • orchestrator.py   — to build accurate LLM context before query rewriting
  • routers/schema_router.py — to serve GET /api/db/schema
  • (future) text-to-sql pipeline — to generate SQL from natural language

Design principles
-----------------
• Uses information_schema only — NO table row data is ever read.
• Works with any active X-Session-ID stored in SessionManager.
• Never logs or exposes credentials.
• Degrades gracefully: every sub-query (PKs, FKs, indexes, row counts) is
  wrapped in its own try/except so a missing privilege or non-standard engine
  feature never kills the entire response — you always get at least column
  names back.
• Returns a compact, JSON-serialisable dict (no Decimal, datetime, etc.).

Returned shape (per table)
--------------------------
{
  "table_name": {
    "row_count": int,              # pg_stat_user_tables estimate, 0 if unavailable
    "columns": [
      {
        "column_name": str,
        "data_type":   str,        # normalised to short aliases (varchar, int, …)
        "is_nullable": bool,
        "is_primary_key": bool,
        "foreign_key_ref": str     # "ref_table.ref_column" or ""
      },
      …
    ],
    "indexes": [
      {
        "index_name": str,
        "columns":    list[str],
        "unique":     bool
      },
      …
    ],
    "primary_key": list[str],      # convenience duplicate of is_primary_key flags
    "foreign_keys": [
      {"column": str, "ref_table": str, "ref_column": str}
    ]
  },
  …
}

Public API
----------
  SchemaService.get_schema(session_id, table_names=None) -> dict
      table_names=None  → discover and return ALL user tables in 'public' schema
      table_names=[…]   → return only the requested tables (same as before)

  SchemaService.get_table_names(session_id) -> list[str]
      Returns all user-visible table names in the 'public' schema.

  get_schema_service() -> SchemaService
      Module-level singleton accessor (mirrors db_service pattern).
"""

import logging
from typing import Optional

from db.db_service import SecureDatabaseService

logger = logging.getLogger(__name__)

# ── Type alias map (PostgreSQL verbose → compact) ─────────────────────────────
_TYPE_ALIASES: dict[str, str] = {
    "character varying":              "varchar",
    "character":                      "char",
    "timestamp without time zone":    "timestamp",
    "timestamp with time zone":       "timestamptz",
    "time without time zone":         "time",
    "time with time zone":            "timetz",
    "double precision":               "float8",
    "real":                           "float4",
    "integer":                        "int",
    "smallint":                       "int2",
    "bigint":                         "int8",
    "numeric":                        "numeric",
    "boolean":                        "bool",
    "bytea":                          "bytea",
}


def _shorten_type(pg_type: str) -> str:
    """Map verbose PostgreSQL type names to compact aliases."""
    return _TYPE_ALIASES.get(pg_type.lower(), pg_type)


class SchemaService:
    """
    Extracts PostgreSQL schema metadata for an active database session.

    All queries target information_schema (portable SQL standard) plus
    pg_stat_user_tables / pg_indexes for PostgreSQL-specific extras
    (row estimates, index details). System-schema tables are always excluded.
    """

    def __init__(self, db_service: Optional[SecureDatabaseService] = None):
        self._db = db_service or SecureDatabaseService()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_table_names(self, session_id: str) -> list[str]:
        """
        Return all user-visible table names in the 'public' schema.
        Returns [] if the session is invalid or the query fails.
        """
        if not session_id:
            return []
        try:
            sql = (
                "SELECT table_name "
                "FROM information_schema.tables "
                "WHERE table_schema = 'public' "
                "  AND table_type = 'BASE TABLE' "
                "ORDER BY table_name;"
            )
            result = self._query(session_id, sql, limit=500)
            rows = result.get("rows") or result.get("data") or []
            names = [r.get("table_name", "") for r in rows if r.get("table_name")]
            logger.info("  ✓ Discovered %d tables in session %s…", len(names), session_id[:8])
            return names
        except Exception as exc:
            logger.warning("  ⚠ get_table_names failed: %s", exc)
            return []

    def get_schema(
        self,
        session_id: str,
        table_names: Optional[list[str]] = None,
    ) -> dict:
        """
        Return full schema metadata for the requested tables.

        Parameters
        ----------
        session_id  : str  — active X-Session-ID from SessionManager
        table_names : list[str] | None
            None  → discover and return ALL user tables in 'public' schema
            [...]  → return only the given tables

        Returns
        -------
        dict  — keyed by table name (lowercase), values per the module docstring.
                Returns {} on session failure (never raises).
        """
        if not session_id:
            logger.warning("get_schema: empty session_id — returning {}")
            return {}

        # Discover tables when none are specified
        if not table_names:
            table_names = self.get_table_names(session_id)
            if not table_names:
                logger.warning("get_schema: no tables found in session %s…", session_id[:8])
                return {}

        lower_names = [t.lower() for t in table_names]

        # Initialise the result skeleton
        schema: dict = {
            t: {
                "row_count":    0,
                "columns":      [],
                "indexes":      [],
                "primary_key":  [],
                "foreign_keys": [],
            }
            for t in lower_names
        }

        try:
            self._fill_columns(session_id, lower_names, schema)
            self._fill_primary_keys(session_id, lower_names, schema)
            self._fill_foreign_keys(session_id, lower_names, schema)
            self._fill_indexes(session_id, lower_names, schema)
            self._fill_row_counts(session_id, lower_names, schema)

            # Annotate each column with its PK / FK flags so consumers have
            # everything they need in a single column dict
            self._annotate_columns(schema)

            logger.info(
                "  ✓ Schema extracted: %s",
                {t: f"{len(v['columns'])} cols" for t, v in schema.items()},
            )
        except Exception as exc:
            logger.error("  ✗ get_schema: unexpected error — %s", exc)

        return schema

    # ──────────────────────────────────────────────────────────────────────────
    # Private fill helpers — each is independently fault-tolerant
    # ──────────────────────────────────────────────────────────────────────────

    def _placeholders(self, names: list[str]) -> str:
        """Build a SQL IN list: 'users', 'orders', …"""
        return ", ".join(f"'{n}'" for n in names)

    def _rows(self, result: dict) -> list[dict]:
        """Normalise run_query_with_metrics result to a plain list of dicts."""
        return result.get("rows") or result.get("data") or []

    def _query(self, session_id: str, sql: str, limit: int) -> dict:
        """
        Thin wrapper around SecureDatabaseService.run_query_with_metrics().

        IMPORTANT: strips the trailing ';' before calling enforce_limit().
        QuerySanitizer.enforce_limit() appends 'LIMIT N' to whatever SQL
        string it receives — if the string already ends in ';' (every query
        in this file is written with one, for readability), the result is
        '...;  LIMIT 500' which is invalid PostgreSQL syntax. db_service.py's
        own methods already strip this internally for the *outer* query
        (user_query.strip().rstrip(';')), but that stripping happens on the
        caller's input there, not here — so we replicate it on our end
        rather than relying on every call site to remember to do it.
        """
        return self._db.run_query_with_metrics(session_id, sql.rstrip(";"), limit=limit)

    def _fill_columns(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """
        Populate schema[t]["columns"] with ordered column metadata.

        Filtered to table_schema = 'public' to prevent ghost columns from
        identically-named tables in other schemas.
        """
        sql = (
            "SELECT table_name, column_name, data_type, is_nullable, ordinal_position "
            "FROM information_schema.columns "
            f"WHERE table_name IN ({self._placeholders(table_names)}) "
            "  AND table_schema = 'public' "
            "ORDER BY table_name, ordinal_position;"
        )
        try:
            result = self._query(session_id, sql, limit=5000)
            for row in self._rows(result):
                tname = (row.get("table_name") or "").lower()
                cname = (row.get("column_name") or "").lower()
                if tname not in schema or not cname:
                    continue
                schema[tname]["columns"].append({
                    "column_name":    cname,
                    "data_type":      _shorten_type(row.get("data_type") or "text"),
                    "is_nullable":    (row.get("is_nullable") or "YES").upper() == "YES",
                    # filled in later by _annotate_columns
                    "is_primary_key": False,
                    "foreign_key_ref": "",
                })
        except Exception as exc:
            logger.warning("  ⚠ _fill_columns failed: %s", exc)

    def _fill_primary_keys(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """Populate schema[t]["primary_key"] list."""
        sql = (
            "SELECT tc.table_name, kcu.column_name "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema    = kcu.table_schema "
            "WHERE tc.constraint_type = 'PRIMARY KEY' "
            f"  AND tc.table_name IN ({self._placeholders(table_names)}) "
            "  AND tc.table_schema = 'public' "
            "ORDER BY kcu.ordinal_position;"
        )
        try:
            result = self._query(session_id, sql, limit=500)
            for row in self._rows(result):
                tname = (row.get("table_name") or "").lower()
                cname = (row.get("column_name") or "").lower()
                if tname in schema and cname:
                    schema[tname]["primary_key"].append(cname)
        except Exception as exc:
            logger.debug("  _fill_primary_keys skipped: %s", exc)

    def _fill_foreign_keys(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """
        Populate schema[t]["foreign_keys"].

        Uses the same FK ghost-column guard as _get_live_schema() in
        orchestrator.py: only records a FK if source_column is actually
        a real column of source_table (the ccu join can surface incoming FKs
        from child tables as if they belong to the parent).
        """
        sql = (
            "SELECT "
            "  tc.table_name  AS source_table, "
            "  kcu.column_name AS source_column, "
            "  ccu.table_name  AS ref_table, "
            "  ccu.column_name AS ref_column "
            "FROM information_schema.table_constraints tc "
            "JOIN information_schema.key_column_usage kcu "
            "  ON tc.constraint_name = kcu.constraint_name "
            " AND tc.table_schema    = kcu.table_schema "
            "JOIN information_schema.constraint_column_usage ccu "
            "  ON tc.constraint_name = ccu.constraint_name "
            " AND tc.table_schema    = ccu.table_schema "
            "WHERE tc.constraint_type = 'FOREIGN KEY' "
            f"  AND tc.table_name IN ({self._placeholders(table_names)}) "
            "  AND tc.table_schema = 'public';"
        )
        try:
            result = self._query(session_id, sql, limit=500)
            for row in self._rows(result):
                tname      = (row.get("source_table")  or "").lower()
                source_col = (row.get("source_column") or "").lower()
                ref_table  = (row.get("ref_table")     or "").lower()
                ref_col    = (row.get("ref_column")    or "").lower()
                if tname not in schema or not source_col:
                    continue
                # Ghost-column guard — only accept if the column exists on the table
                real_col_names = {c["column_name"] for c in schema[tname]["columns"]}
                if source_col not in real_col_names:
                    logger.debug(
                        "  _fill_foreign_keys: skipping ghost FK %s.%s → %s.%s",
                        tname, source_col, ref_table, ref_col,
                    )
                    continue
                schema[tname]["foreign_keys"].append({
                    "column":     source_col,
                    "ref_table":  ref_table,
                    "ref_column": ref_col,
                })
        except Exception as exc:
            logger.debug("  _fill_foreign_keys skipped: %s", exc)

    def _fill_indexes(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """
        Populate schema[t]["indexes"] using pg_indexes (PostgreSQL-specific).
        Falls back gracefully on non-PG engines.

        pg_indexes gives us the full CREATE INDEX DDL in the `indexdef` column
        which we parse for column names (handles multi-column indexes correctly).
        We also query pg_index + pg_attribute for structured column data which
        is more reliable than parsing DDL strings, using pg_indexes only as a
        fallback for the unique flag.
        """
        sql = (
            "SELECT "
            "  i.relname  AS index_name, "
            "  t.relname  AS table_name, "
            "  ix.indisunique AS is_unique, "
            "  array_agg(a.attname ORDER BY array_position(ix.indkey, a.attnum)) AS columns "
            "FROM pg_index ix "
            "JOIN pg_class  i ON i.oid = ix.indexrelid "
            "JOIN pg_class  t ON t.oid = ix.indrelid "
            "JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = ANY(ix.indkey) "
            "JOIN pg_namespace n ON n.oid = t.relnamespace "
            "WHERE n.nspname = 'public' "
            f"  AND t.relname IN ({self._placeholders(table_names)}) "
            "GROUP BY i.relname, t.relname, ix.indisunique;"
        )
        try:
            result = self._query(session_id, sql, limit=500)
            for row in self._rows(result):
                tname  = (row.get("table_name") or "").lower()
                iname  = (row.get("index_name") or "").lower()
                unique = bool(row.get("is_unique") or False)
                cols   = row.get("columns") or []
                # pg8000 returns arrays as Python lists already
                if isinstance(cols, str):
                    # Strip curly braces from PostgreSQL array literal e.g. "{id,name}"
                    cols = [c.strip() for c in cols.strip("{}").split(",") if c.strip()]
                if tname in schema and iname:
                    schema[tname]["indexes"].append({
                        "index_name": iname,
                        "columns":    cols,
                        "unique":     unique,
                    })
        except Exception as exc:
            logger.debug("  _fill_indexes (pg_index) failed: %s — trying pg_indexes fallback", exc)
            self._fill_indexes_fallback(session_id, table_names, schema)

    def _fill_indexes_fallback(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """
        Simpler fallback using pg_indexes view (still PostgreSQL-only).
        Parses column names from the indexdef DDL string.
        """
        sql = (
            "SELECT tablename, indexname, indexdef "
            "FROM pg_indexes "
            "WHERE schemaname = 'public' "
            f"AND tablename IN ({self._placeholders(table_names)});"
        )
        try:
            import re
            result = self._query(session_id, sql, limit=500)
            for row in self._rows(result):
                tname   = (row.get("tablename")  or "").lower()
                iname   = (row.get("indexname")  or "").lower()
                indexdef = row.get("indexdef") or ""
                if tname not in schema or not iname:
                    continue
                # Parse columns from e.g. "CREATE UNIQUE INDEX idx ON t (col1, col2)"
                col_match = re.search(r"\(([^)]+)\)", indexdef)
                cols = []
                if col_match:
                    cols = [c.strip().lower() for c in col_match.group(1).split(",")]
                unique = "UNIQUE" in indexdef.upper()
                schema[tname]["indexes"].append({
                    "index_name": iname,
                    "columns":    cols,
                    "unique":     unique,
                })
        except Exception as exc:
            logger.debug("  _fill_indexes_fallback also failed: %s", exc)

    def _fill_row_counts(
        self, session_id: str, table_names: list[str], schema: dict
    ) -> None:
        """
        Populate schema[t]["row_count"] from pg_stat_user_tables (fast estimate).
        Does NOT read any actual row data — this is a catalogue query only.
        Falls back to 0 if the view is unavailable.
        """
        sql = (
            "SELECT relname AS table_name, n_live_tup AS row_count "
            "FROM pg_stat_user_tables "
            f"WHERE relname IN ({self._placeholders(table_names)});"
        )
        try:
            result = self._query(session_id, sql, limit=500)
            for row in self._rows(result):
                tname = (row.get("table_name") or "").lower()
                if tname in schema:
                    try:
                        schema[tname]["row_count"] = int(row.get("row_count") or 0)
                    except (TypeError, ValueError):
                        schema[tname]["row_count"] = 0
        except Exception as exc:
            logger.debug("  _fill_row_counts skipped (pg_stat_user_tables unavailable): %s", exc)

    def _annotate_columns(self, schema: dict) -> None:
        """
        Cross-reference PK and FK data into each column dict so API consumers
        get a single, self-contained column object rather than having to join
        three separate lists.
        """
        for tname, tdata in schema.items():
            pk_set = set(tdata.get("primary_key") or [])
            fk_map = {
                fk["column"]: f"{fk['ref_table']}.{fk['ref_column']}"
                for fk in (tdata.get("foreign_keys") or [])
            }
            for col in tdata.get("columns") or []:
                cname = col["column_name"]
                col["is_primary_key"]  = cname in pk_set
                col["foreign_key_ref"] = fk_map.get(cname, "")


# ── Module-level singleton ─────────────────────────────────────────────────────

_singleton: Optional[SchemaService] = None


def get_schema_service() -> SchemaService:
    """Return the module-level SchemaService singleton."""
    global _singleton
    if _singleton is None:
        _singleton = SchemaService()
    return _singleton