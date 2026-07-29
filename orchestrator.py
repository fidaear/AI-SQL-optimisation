"""
orchestrator.py
---------------
Central pipeline coordinator for the AI-Assisted SQL Query Optimizer.
Connects: query_analyzer → statistics_extractor → index_recommender → rule_engine → ai_optimizer
Provides: full end-to-end flow with logging and structured error handling.

Added: run_pipeline_streaming() — sync generator that yields one SSE-ready
dict per stage so main.py can stream live progress to the Angular UI.
"""
import json
import re as _re
import re
import difflib
import traceback
from typing import Generator, Optional
from utils.logger import get_logger
from pathlib import Path
from dotenv import load_dotenv

# Explicit path — works regardless of cwd when uvicorn is launched
env_path = Path(__file__).resolve().parent.parent / ".env"   # backend/.env
load_dotenv(dotenv_path=env_path)

import sys
import os
# ... rest of your imports
# NEW: for JSON-safe conversion
import decimal
import datetime

from core.query_analyzer import QueryAnalyzer
from core.statistics_extractor import StatisticsExtractor
from core.index_recommender import generate_index_suggestions
from core.rule_engine import run_all_rules
from ai.ai_optimizer import AIOptimizer, ContextBuilder, SQLSafetyValidator
from ai.llm_client   import LLMCallType, LLMClientError
from ai.sql_extractor import extract_sql
from ai.table_selector import select_relevant_tables_basic
from ai.text_to_sql_builder import generate_sql_from_question
from db.db_service import SecureDatabaseService
from db.schema_service import get_schema_service
from ai.schema_formatter import format_schema_for_prompt

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# NEW: JSON-safety helper
# ---------------------------------------------------------------------------

def _make_json_safe(obj):
    """
    Recursively convert any non-JSON-serialisable objects (Decimal, datetime, …)
    into Python primitives (float, str, …).
    """
    if isinstance(obj, decimal.Decimal):
        # Convert to float (or int if exactly representable)
        return float(obj) if obj % 1 != 0 else int(obj)
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Helpers (existing)
# ---------------------------------------------------------------------------

def _clean_llm_text(text: str) -> str:
    """
    Strip markdown artifacts that local LLMs often inject:
      **bold**  ```sql ... ```  ``` ... ```  leading/trailing whitespace
    """
    if not text:
        return ""
    text = re.sub(r"```sql\s*", "", text)
    text = re.sub(r"```\s*",    "", text)
    text = re.sub(r"\*\*",      "", text)
    text = re.sub(r"\*",        "", text)
    text = text.strip()
    return text


def _is_valid_sql(sql: str) -> bool:
    """
    Minimal sanity check: must contain SELECT/WITH and FROM,
    and not be obviously truncated.
    """
    if not sql or len(sql.strip()) < 10:
        return False
    upper = sql.upper().strip()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False
    if "FROM" not in upper:
        return False
    return True


def _clean_optimized_sql(optimized_sql: str, original_sql: str) -> str:
    """
    Run the full sql_extractor pipeline on Mistral's raw output.
    Falls back to the original query if extraction yields nothing or
    produces an invalid SQL string.
    """
    if not optimized_sql or not optimized_sql.strip():
        logger.warning("  ⚠ optimized_sql is empty — falling back to original query")
        return original_sql.strip()

    cleaned = extract_sql(optimized_sql)

    if not cleaned or not cleaned.strip():
        logger.warning(
            "  ⚠ sql_extractor returned empty for input: %r — falling back to original",
            optimized_sql[:200],
        )
        return original_sql.strip()

    if not _is_valid_sql(cleaned):
        logger.warning(
            "  ⚠ sql_extractor returned invalid SQL: %r — falling back to original",
            cleaned[:200],
        )
        return original_sql.strip()

    logger.info("  ✓ sql_extractor cleaned SQL: %r", cleaned[:120])
    return cleaned


def _serialize_violations(violations: list) -> list:
    """
    Convert RuleResult dataclass instances → plain dicts so that
    FastAPI / json.dumps can serialise them without errors.
    """
    serialized = []
    for v in violations:
        if isinstance(v, dict):
            serialized.append(v)
        else:
            serialized.append({
                "rule":       v.rule,
                "severity":   v.severity,
                "message":    v.message,
                "suggestion": getattr(v, "suggestion", ""),
            })
    return serialized


def _unwrap_envelope(raw_stats: dict) -> dict:
    """
    Unwrap a top-level {"data": ..., "summary": ...} envelope, regardless of
    whether 'data' holds a single stats dict or a list of per-table dicts.

    Returns a dict suitable for the normal Shape A/B/C detection logic:
      • If 'data' is a dict   → return it directly (single-table or already
                                  nested-by-table-name payload).
      • If 'data' is a list   → re-key it into {table_name: table_dict, ...}
                                  using whatever name field each entry has
                                  (table_name / table / name), so downstream
                                  nested-shape detection works normally.
      • No envelope detected  → return raw_stats unchanged.
    """
    if not isinstance(raw_stats, dict) or "data" not in raw_stats:
        return raw_stats

    data = raw_stats["data"]

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        rekeyed: dict = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            tname = (
                entry.get("table_name")
                or entry.get("table")
                or entry.get("name")
            )
            if tname:
                rekeyed[str(tname).lower()] = entry
        if rekeyed:
            logger.debug(
                "_unwrap_envelope: re-keyed list envelope into tables: %s",
                list(rekeyed.keys()),
            )
            return rekeyed
        logger.warning(
            "_unwrap_envelope: 'data' list entries had no table_name/table/name "
            "field — falling back to raw envelope"
        )

    return raw_stats


def _normalize_index(raw_index) -> dict:
    """
    Convert whatever shape a single index entry arrives in into the dict
    that ContextBuilder.from_statistics_dict() expects:

        {"index_name": str, "columns": list[str], "unique": bool}

    Handles:
      • Already a correct dict           → pass through (fill missing keys)
      • A plain string (just the name)   → wrap it
      • A dict with 'indexname' key      → rename key (psycopg2 shape)
    """
    if isinstance(raw_index, str):
        # e.g. "idx_users_user_id"  or  "CREATE INDEX ..."
        # Try to parse a column name out of a CREATE INDEX statement
        columns: list[str] = []
        col_match = re.search(r"\(([^)]+)\)", raw_index)
        if col_match:
            columns = [c.strip() for c in col_match.group(1).split(",")]
        return {
            "index_name": raw_index,
            "columns":    columns,
            "unique":     "UNIQUE" in raw_index.upper(),
        }

    if isinstance(raw_index, dict):
        # Normalize key aliases that different DB drivers use
        name = (
            raw_index.get("index_name")
            or raw_index.get("indexname")   # psycopg2 shape
            or raw_index.get("name")
            or ""
        )
        columns = raw_index.get("columns") or raw_index.get("column_names") or []
        # Some drivers return columns as a comma-separated string
        if isinstance(columns, str):
            columns = [c.strip() for c in columns.split(",")]
        unique = bool(
            raw_index.get("unique")
            or raw_index.get("is_unique")
            or False
        )
        return {"index_name": name, "columns": columns, "unique": unique}

    if isinstance(raw_index, list):
        # An envelope or grouping artifact handed us a list where a single
        # index entry was expected (e.g. {"data": [...]} leaking through,
        # or the extractor grouping several indexes together). Recurse into
        # the first usable dict/string entry rather than discarding the
        # whole thing as a stub.
        for item in raw_index:
            if isinstance(item, (dict, str)):
                return _normalize_index(item)
        logger.warning("_normalize_index: list contained no usable entries — using stub")
        return {"index_name": "", "columns": [], "unique": False}

    # Fallback — unknown type, return empty stub so the loop doesn't crash
    logger.warning("_normalize_index: unexpected type %s — using stub", type(raw_index))
    return {"index_name": str(raw_index), "columns": [], "unique": False}


def _has_real_table_fields(d: dict) -> bool:
    """True if `d` already looks like a table-stats dict (not an envelope)."""
    return isinstance(d, dict) and any(
        k in d
        for k in ("row_count", "reltuples", "n_live_tup", "columns", "schema", "indexes", "index_list")
    )


def _deep_unwrap_table_entry(raw, table_name: str | None = None, _depth: int = 0):
    """
    Recursively unwrap {"data": ..., "summary": ...} envelopes around a
    SINGLE table's stats, however many levels deep they're nested.

    Some StatisticsExtractor versions wrap the whole multi-table response
    in one envelope AND wrap each individual table's payload in its own
    envelope, e.g.:
        {"data": [{"table_name": "products",
                   "data": {"row_count": 1200, "indexes": [...]},
                   "summary": "..."}], "summary": "..."}

    _unwrap_envelope() (used in _normalize_statistics) only peels the
    OUTER envelope. This peels any remaining inner envelope(s) so the
    dict that finally reaches row_count/columns/indexes extraction below
    is the real table payload, not another wrapper.

    Guards against unbounded recursion on malformed/cyclic input with a
    depth cap; falls back to an empty dict rather than raising.
    """
    if _depth > 5:
        logger.warning("_deep_unwrap_table_entry: max unwrap depth exceeded — using stub")
        return {}

    if not isinstance(raw, dict):
        return raw if isinstance(raw, dict) else {}

    if _has_real_table_fields(raw):
        return raw

    if "data" not in raw:
        # No real fields AND no envelope marker — nothing more to unwrap.
        return raw

    data = raw["data"]

    if isinstance(data, dict):
        return _deep_unwrap_table_entry(data, table_name, _depth + 1)

    if isinstance(data, list):
        # Prefer the entry matching table_name when we know it; otherwise
        # fall back to the first dict entry.
        chosen = None
        if table_name:
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                tname = entry.get("table_name") or entry.get("table") or entry.get("name")
                if tname and str(tname).lower() == table_name.lower():
                    chosen = entry
                    break
        if chosen is None:
            chosen = next((e for e in data if isinstance(e, dict)), None)
        if chosen is None:
            logger.warning(
                "_deep_unwrap_table_entry: 'data' list had no usable dict entries — using stub"
            )
            return {}
        return _deep_unwrap_table_entry(chosen, table_name, _depth + 1)

    return {}


def _normalize_table_entry(raw: dict, table_name: str | None = None) -> dict:
    """
    Ensure a single table's stats dict has the exact keys ContextBuilder expects:
        row_count : int
        columns   : list[dict]   (each dict has at least 'column_name', 'data_type')
        indexes   : list[dict]   (each dict has 'index_name', 'columns', 'unique')

    Handles field-name aliases from different DB drivers / extractor versions,
    and recursively unwraps any nested {"data": ..., "summary": ...} envelopes
    (single or double-wrapped) before reading fields. Without this, values
    silently fall back to empty/zero because the real keys live one or more
    levels deeper than expected.
    """
    raw = _deep_unwrap_table_entry(raw, table_name)

    row_count_raw = (
        raw.get("row_count")
        or raw.get("reltuples")
        or raw.get("n_live_tup")
        or 0
    )
    row_count = _coerce_row_count(row_count_raw)

    # ── columns ──────────────────────────────────────────────────────────────
    raw_columns = raw.get("columns") or raw.get("schema") or []
    columns: list[dict] = []
    for col in raw_columns:
        if isinstance(col, dict):
            columns.append(col)
        elif isinstance(col, str):
            columns.append({"column_name": col, "data_type": "text"})

    # ── indexes ──────────────────────────────────────────────────────────────
    raw_indexes = raw.get("indexes") or raw.get("index_list") or []
    if isinstance(raw_indexes, dict):
        # Some extractors key indexes by index name: {"idx_x": {...}, ...}
        # but guard against this actually being an envelope itself.
        if "data" in raw_indexes and not _has_real_table_fields(raw_indexes):
            raw_indexes = raw_indexes.get("data") or []
        else:
            raw_indexes = list(raw_indexes.values())
    if not isinstance(raw_indexes, list):
        raw_indexes = []
    indexes = [_normalize_index(idx) for idx in raw_indexes]

    return {
        "row_count": row_count,
        "columns":   columns,
        "indexes":   indexes,
    }


def _coerce_row_count(value) -> int:
    """
    Safely coerce a row-count-like value into an int, regardless of which
    shape the upstream extractor handed us. Handles:
      • int / float            → int(value)
      • numeric string         → int(float(value))
      • dict with a nested estimate, e.g. {"estimate": 1200}, {"value": 1200},
        {"reltuples": 1200}, or anything else with a usable numeric field
      • None / "" / garbage     → 0
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value.strip())) if value.strip() else 0
        except (ValueError, TypeError):
            return 0
    if isinstance(value, dict):
        for key in ("row_count", "reltuples", "n_live_tup", "estimate", "value", "count"):
            if key in value:
                coerced = _coerce_row_count(value[key])
                if coerced:
                    return coerced
        for v in value.values():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return int(v)
        logger.warning(
            "_coerce_row_count: dict had no usable numeric field (keys: %s) — defaulting to 0",
            list(value.keys()),
        )
        return 0
    if isinstance(value, (list, tuple)) and value:
        return _coerce_row_count(value[0])
    logger.warning(
        "_coerce_row_count: unexpected type %s for row_count — defaulting to 0",
        type(value),
    )
    return 0


def _is_extractor_envelope(raw_stats: dict) -> bool:
    """
    True if raw_stats matches the real StatisticsExtractor.get_comprehensive_statistics()
    contract:

        {
            "success": bool,
            "reltuples": {"data": [{"table_name": ..., "reltuples": ..., ...}, ...], "summary": {...}},
            "indexes":   {"data": [{"table_name": ..., "index_name": ..., "column_name": ..., ...}, ...], "summary": {...}},
            "n_distinct": {"data": [...]},
        }

    This is NOT nested by table name at the top level — "reltuples" and
    "indexes" are fixed keys holding flat lists of per-row records (one row
    per table for reltuples; one row per table+index+column for indexes,
    since a multi-column index produces multiple rows). Distinguishing this
    from Shape A/B/C matters: treating the whole blob as one flat table's
    stats (the old Shape-C fallback) is what produced the
    "_coerce_row_count: dict had no usable numeric field (keys: ['data', 'summary'])"
    and "_normalize_index: unexpected type list" warnings — raw.get("indexes")
    resolved to {"data": [...], "summary": {...}}, a dict, not a list, and
    raw.get("row_count") was never present at all.
    """
    if not isinstance(raw_stats, dict):
        return False
    has_reltuples = isinstance(raw_stats.get("reltuples"), dict) and "data" in raw_stats["reltuples"]
    has_indexes   = isinstance(raw_stats.get("indexes"), dict) and "data" in raw_stats["indexes"]
    return has_reltuples or has_indexes


def _normalize_extractor_statistics(raw_stats: dict, table_names: list[str]) -> dict:
    """
    Normalize the real StatisticsExtractor envelope shape into the per-table
    nested dict ContextBuilder expects:

        {"table_name": {"row_count": int, "columns": [...], "indexes": [...]}, ...}

    Steps:
      1. Read reltuples["data"] — one row per table — and pick out row_count
         (and relpages/n_dead_tup for context, dropped here since
         ContextBuilder doesn't use them) per requested table name.
      2. Read indexes["data"] — one row per (table, index, column) — and
         GROUP rows sharing the same (table_name, index_name) into a single
         index dict with a multi-column "columns" list. Without this
         grouping, a 2-column index would silently appear as two separate
         single-column index entries.
      3. Match table names case-insensitively, since Postgres relnames are
         lowercase by convention but the SQL the user typed may not be.
    """
    reltuples_rows = ((raw_stats.get("reltuples") or {}).get("data")) or []
    index_rows     = ((raw_stats.get("indexes")   or {}).get("data")) or []

    # ── row_count per table, case-insensitive lookup ───────────────────────
    row_count_by_table: dict = {}
    for row in reltuples_rows:
        if not isinstance(row, dict):
            continue
        tname = row.get("table_name")
        if not tname:
            continue
        row_count_by_table[str(tname).lower()] = _coerce_row_count(
            row.get("reltuples", row.get("row_count", 0))
        )

    # ── indexes per table, grouped by (table, index_name) → multi-column ───
    # raw rows look like: {"table_name": "reviews", "index_name": "idx_x",
    #                       "column_name": "user_id", "idx_scan": 12}
    indexes_by_table: dict = {}
    for row in index_rows:
        if not isinstance(row, dict):
            continue
        tname = row.get("table_name")
        iname = row.get("index_name")
        if not tname or not iname:
            continue
        tkey = str(tname).lower()
        bucket = indexes_by_table.setdefault(tkey, {})
        entry = bucket.setdefault(iname, {"index_name": iname, "columns": [], "unique": False})
        col = row.get("column_name")
        if col and col not in entry["columns"]:
            entry["columns"].append(col)

    normalized: dict = {}
    for tname in table_names:
        tkey = tname.lower()
        normalized[tname] = {
            "row_count": row_count_by_table.get(tkey, 0),
            "columns":   [],  # StatisticsExtractor doesn't return column/type info today
            "indexes":   list(indexes_by_table.get(tkey, {}).values()),
        }

    logger.debug(
        "_normalize_extractor_statistics: row_counts=%s, index_counts=%s",
        {t: normalized[t]["row_count"] for t in normalized},
        {t: len(normalized[t]["indexes"]) for t in normalized},
    )
    return normalized


def _normalize_statistics(raw_stats: dict, table_names: list[str]) -> dict:
    """
    Normalize whatever shape StatisticsExtractor returns into the nested
    dict that ContextBuilder.from_statistics_dict() expects:

        {
            "table_name": {
                "row_count": int,
                "columns":   [{"column_name": str, "data_type": str, ...}],
                "indexes":   [{"index_name": str, "columns": [...], "unique": bool}],
            },
            ...
        }

    Shape detection order matters — checked from most-specific to most-generic:

      Shape D (checked FIRST) — the real StatisticsExtractor envelope:
        {"success": ..., "reltuples": {"data": [...]}, "indexes": {"data": [...]}, "n_distinct": {...}}
        → _normalize_extractor_statistics() (flat lists, grouped by table)

      Shape A — already nested by table name, values are well-formed dicts:
        {"users": {"row_count": 100, "columns": [...], "indexes": [...]}, ...}
        → normalize each table entry's internals (index strings → dicts)

      Shape B — nested by table name but values may have wrong field names
        or indexes stored as strings / aliased keys:
        {"users": {"reltuples": 100, "indexes": ["idx_users_pk", ...]}, ...}
        → normalize each table entry via _normalize_table_entry()

      Shape C — flat dict (e.g. a single table's stats passed directly via
        db_stats= in the API, bypassing StatisticsExtractor entirely):
        {"reltuples": 100, "indexes": [...]}
        → wrap under every requested table name
    """
    if not raw_stats:
        return {}

    # Shape D — check BEFORE the generic envelope unwrap, since
    # _unwrap_envelope() would otherwise never even look at "reltuples"/
    # "indexes" (it only inspects a top-level "data" key, which this shape
    # doesn't have at the outer level).
    if _is_extractor_envelope(raw_stats):
        return _normalize_extractor_statistics(raw_stats, table_names)

    raw_stats = _unwrap_envelope(raw_stats)
    if not raw_stats:
        return {}

    # Detect Shape A / B: at least one queried table name is a top-level key
    # AND its value is a dict (not a scalar like row counts stored at top level)
    raw_keys_lower = {str(k).lower(): k for k in raw_stats.keys()}
    is_nested = any(
        t.lower() in raw_keys_lower and isinstance(raw_stats[raw_keys_lower[t.lower()]], dict)
        for t in table_names
    )

    if is_nested:
        # Shape A / B — normalize each table's internals
        normalized: dict = {}
        for tname in table_names:
            actual_key = raw_keys_lower.get(tname.lower())
            raw_table = raw_stats.get(actual_key, {}) if actual_key else {}
            if isinstance(raw_table, dict) and raw_table:
                normalized[tname] = _normalize_table_entry(raw_table, table_name=tname)
            else:
                logger.warning(
                    "_normalize_statistics: table '%s' value is %s, not dict — using stub",
                    tname, type(raw_table),
                )
                normalized[tname] = {"row_count": 0, "columns": [], "indexes": []}
        return normalized

    # Shape C — flat dict: wrap it under every requested table name
    logger.debug(
        "_normalize_statistics: flat dict detected (keys: %s) — wrapping for tables: %s",
        list(raw_stats.keys())[:8], table_names,
    )
    normalized = {}
    for tname in table_names:
        normalized[tname] = _normalize_table_entry(raw_stats, table_name=tname)
    return normalized


def _extract_table_names(sql: str) -> list[str]:
    """
    Quick regex extraction of table names referenced in the SQL.
    Used to filter the statistics dict before building the DB context,
    so we only pass the LLM the tables it actually needs.
    """
    pattern = r"(?:FROM|JOIN)\s+([`\"'\[]?[\w.]+[`\"'\]]?)"
    names = {
        m.group(1).strip('`"\'[]').lower()
        for m in _re.finditer(pattern, sql, _re.IGNORECASE)
    }
    return list(names)


def _get_table_columns(session_id: str, table_names: list[str]) -> dict[str, set[str]]:
    """
    Query the database's information_schema to get the real column names for
    every table referenced in the SQL. Returns a dict like:
        {"products": {"id", "name", "price"}, "users": {"user_id", "email"}}

    Used by _validate_optimized_columns() to catch LLM hallucinated column
    names before we send the rewritten query to PostgreSQL.

    Returns an empty dict if the session is missing or the query fails —
    callers must handle this gracefully (skip validation, not crash).

    NOTE: kept as a thin wrapper around _get_live_schema() for backward
    compatibility with existing callers that only need the column-name set
    (not full type/PK/FK detail).
    """
    schema = _get_live_schema(session_id, table_names)
    return {t: set(info["columns"].keys()) for t, info in schema.items()}


def _get_live_schema(session_id: str, table_names: list[str]) -> dict[str, dict]:
    """
    Single source of truth for "what does this database actually look like".

    REFACTORED: this now delegates to db.schema_service.SchemaService instead
    of running its own information_schema queries. SchemaService is the
    canonical schema extractor shared with GET /api/db/schema and the
    upcoming /api/text-to-sql endpoint (Text-to-SQL Task 1). Keeping two
    independent implementations of the same FK ghost-column guard in sync
    was the original risk this refactor removes — there is now exactly one
    place that decides "is this FK row real or a leaked incoming reference".

    The return shape below is preserved EXACTLY as before (columns as a
    dict[name, type], not SchemaService's richer list[dict] form) so every
    existing caller in this file — _get_table_columns(), the Stage 5
    db_context injection block, and _validate_optimized_columns() via
    real_columns — keeps working with zero changes.

    Returns:
        {
          "users": {
            "columns": {"id": "integer", "email": "varchar", ...},
            "primary_key": ["id"],
            "foreign_keys": [
                {"column": "role_id", "ref_table": "roles", "ref_column": "id"}
            ],
          },
          ...
        }

    Returns {} if session_id/table_names are missing or introspection fails
    — callers must treat that as "no schema available", not crash.

    NOTE: this is now a thin wrapper around _get_live_schema_full(), which
    also exposes SchemaService's raw (unreshaped) output — needed by
    ai.schema_formatter.format_schema_for_prompt() to render the compact
    "Table: X / Columns: ..." text block for the LLM prompt. Kept as a
    separate function so existing call sites that only want the reshaped
    dict don't have to change.
    """
    reshaped, _raw = _get_live_schema_full(session_id, table_names)
    return reshaped


def _get_live_schema_full(
    session_id: str, table_names: list[str]
) -> tuple[dict[str, dict], dict]:
    """
    Fetch live schema from SchemaService ONCE and return both:
      1. reshaped  — the compact {col_name: data_type} shape this module's
                      existing callers expect (see _get_live_schema() above).
      2. raw_schema — SchemaService's native shape (list[dict] per column,
                      with is_primary_key / foreign_key_ref already annotated),
                      which is exactly what ai.schema_formatter expects.
                      Keyed by lowercase table name, same as `reshaped`.

    Doing both from a single SchemaService.get_schema() call avoids hitting
    the database twice for the same tables in the same pipeline run.

    Returns ({}, {}) if session_id/table_names are missing or introspection
    fails — callers must treat that as "no schema available", not crash.
    """
    if not session_id or not table_names:
        return {}, {}

    lower_names = [t.lower() for t in table_names]

    try:
        schema_service = get_schema_service()
        full_schema = schema_service.get_schema(session_id, table_names=lower_names)

        if not full_schema:
            logger.warning(
                "  _get_live_schema_full: SchemaService returned no data for "
                "tables %s (session %s…)", lower_names, session_id[:8]
            )
            return {}, {}

        # ── Reshape SchemaService's per-column dict list into the compact
        #    {col_name: data_type} form this module's callers expect ────────
        reshaped: dict[str, dict] = {}
        for tname in lower_names:
            tdata = full_schema.get(tname)
            if not tdata:
                # Table wasn't found / introspection failed for it specifically
                reshaped[tname] = {"columns": {}, "primary_key": [], "foreign_keys": []}
                continue

            columns = {
                col["column_name"]: col["data_type"]
                for col in tdata.get("columns", [])
            }
            reshaped[tname] = {
                "columns":      columns,
                "primary_key":  tdata.get("primary_key", []),
                "foreign_keys": tdata.get("foreign_keys", []),
            }

        logger.info(
            "  ✓ Live schema fetched via SchemaService: %s",
            {t: f"{len(v['columns'])} cols, {len(v['foreign_keys'])} FKs"
             for t, v in reshaped.items()},
        )
        return reshaped, full_schema

    except Exception as exc:
        logger.warning("  ⚠ _get_live_schema_full failed (non-fatal): %s", exc)
        return {}, {}


def _build_schema_text(raw_schema: dict, table_names: list[str]) -> str:
    """
    Render the compact "Table: X / Columns: ..." text block for the LLM
    prompt, using ai.schema_formatter.SchemaFormatter on SchemaService's
    raw (unreshaped) output.

    Non-fatal: returns "" (never raises) if raw_schema is empty or
    formatting fails for any reason, so a formatting hiccup never breaks
    the pipeline — the LLM just gets an emptier prompt that run.
    """
    if not raw_schema:
        return ""
    try:
        return format_schema_for_prompt(raw_schema, tables=table_names)
    except Exception as exc:
        logger.warning("  ⚠ _build_schema_text failed (non-fatal): %s", exc)
        return ""


def _find_from_index(upper_sql: str) -> int:
    """
    Return the index of the first ' FROM' token that is at parenthesis
    depth 0 (i.e. not inside a subquery or function call).
    Returns -1 if not found.

    Using upper.find(" FROM ") is broken for two reasons:
      1. It requires a trailing space, so 'FROM orders;' (no trailing space
         before FROM) never matches when the table name directly follows.
      2. It matches ' FROM' inside subqueries, which breaks col extraction
         for queries like: SELECT (SELECT MAX(id) FROM sub) AS x FROM main
    This depth-aware scan fixes both issues.
    """
    depth = 0
    n = len(upper_sql)
    i = 0
    while i < n:
        ch = upper_sql[i]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif depth == 0 and upper_sql[i:i+5] == ' FROM':
            # Accept if followed by a space/tab/newline/end-of-string/semicolon
            after = upper_sql[i+5:i+6]
            if after in ('', ' ', '\t', '\n', '\r', ';'):
                return i
        i += 1
    return -1


def _extract_selected_columns(sql: str) -> list[str]:
    """
    Return the bare, lowercase column names from a flat SELECT list.

    Returns [] when:
      - The query is not a SELECT
      - The SELECT list is just '*' (wildcard)
      - The query cannot be parsed (subquery in SELECT, CTE, etc.)

    Handles:
      - table.column prefix   ('o.status'  ->  'status')
      - col AS alias          ('status AS s'  ->  'status')
      - aggregate functions   ('COUNT(*)'  ->  skipped, contains '(')
      - leading/trailing whitespace in each token
    """
    try:
        stripped = sql.strip()
        upper = stripped.upper()
        if not upper.startswith("SELECT"):
            return []

        from_idx = _find_from_index(upper)
        if from_idx == -1:
            return []

        col_part = stripped[len("SELECT"):from_idx].strip()
        if col_part == "*" or col_part == "":
            return []

        cols: list[str] = []
        for raw_col in col_part.split(","):
            token = raw_col.strip()
            if not token:
                continue
            # Strip table alias prefix: 'o.status' -> 'status'
            if "." in token:
                token = token.split(".")[-1].strip()
            # Strip column alias: 'status AS s' or 'status s' -> 'status'
            alias_match = _re.split(r"\s+AS\s+|\s+", token, maxsplit=1, flags=_re.IGNORECASE)
            base = alias_match[0].strip().strip('`"\'[]').lower()
            # Skip expressions/functions (contain parentheses)
            if base and "(" not in base:
                cols.append(base)
        return cols
    except Exception:
        return []


def _strip_unknown_columns(sql: str, unknown: set[str]) -> Optional[str]:
    """
    Remove every column whose bare name is in *unknown* from the SELECT
    list of a flat 'SELECT col1, col2, ... FROM ...' query.

    Returns the repaired SQL string, or None when:
      - The query is not a simple top-level SELECT (CTE/subquery — don't touch)
      - Stripping would leave an empty SELECT list
      - Any parse error occurs

    Preserves original casing and spacing of kept columns.
    Handles table-aliased columns ('o.status') and AS aliases.
    *unknown* should be a set for O(1) lookup.
    """
    if not unknown:
        return sql
    try:
        stripped = sql.strip()
        upper = stripped.upper()
        if not upper.startswith("SELECT"):
            return None

        from_idx = _find_from_index(upper)
        if from_idx == -1:
            return None

        col_part = stripped[len("SELECT"):from_idx]
        rest = stripped[from_idx:]      # ' FROM ...' onwards, kept verbatim

        kept: list[str] = []
        for raw_col in col_part.split(","):
            token = raw_col.strip()
            if not token:
                continue
            # Extract bare name (same logic as _extract_selected_columns)
            bare = token
            if "." in bare:
                bare = bare.split(".")[-1].strip()
            alias_match = _re.split(r"\s+AS\s+|\s+", bare, maxsplit=1, flags=_re.IGNORECASE)
            base = alias_match[0].strip().strip('`"\'[]').lower()
            if base in unknown:
                continue        # drop hallucinated column
            kept.append(token)  # preserve original token verbatim

        if not kept:
            return None         # would produce 'SELECT FROM ...' — not repairable

        return "SELECT " + ", ".join(kept) + rest
    except Exception:
        return None


def _validate_optimized_columns(
    optimized_sql: str,
    original_sql: str,
    columns_by_table: dict[str, set[str]],
    table_names: list[str] | None = None,
) -> tuple[str, bool]:
    """
    Ensure every bare column name in the SELECT list of *optimized_sql*
    actually exists in the database schema.

    Pipeline (each step only runs if the previous one left unknowns):
      1. No schema info  ->  skip validation, trust SQL.
      2. Parse SELECT list.  SELECT * / unparseable  ->  skip validation.
      3. Build union of all known columns across referenced tables.
      4. Detect unknown (hallucinated) columns.
      5. No unknowns found  ->  SQL is clean, return unchanged.
      6. Try to strip the unknown columns from the SELECT list while
         keeping all other parts of the rewrite (JOINs, WHERE, ORDER BY).
      7. Re-verify repaired SQL contains no unknown columns.
      8. Return (repaired_sql, True) on success or
         (original_sql, False) if repair was not possible.

    Returns:
        (sql_to_execute: str, col_valid: bool)

    NOTE: This deliberately does NOT attempt fuzzy-correction of column
    names (the old _autocorrect_unknown_columns step). Fuzzy correction
    was silently re-introducing hallucinated names as 'corrections' and
    masking the real problem. If typo correction is needed in the future,
    add it as a separate, clearly-logged step AFTER this guard.
    """
    # Step 1 -- no schema, cannot validate
    if not columns_by_table:
        logger.debug("  [GUARD] No schema -- skipping column validation")
        return optimized_sql, True

    # Step 2 -- parse SELECT list
    selected_cols = _extract_selected_columns(optimized_sql)
    if not selected_cols:
        logger.debug("  [GUARD] SELECT * or unparseable -- skipping column validation")
        return optimized_sql, True

    # Step 3 -- union of all real columns across all referenced tables
    all_known: set[str] = set()
    for col_set in columns_by_table.values():
        all_known.update(c for c in col_set if isinstance(c, str))

    # Step 4 -- detect unknowns
    unknown_list = [c for c in selected_cols if c not in all_known]

    # Step 5 -- SQL is clean
    if not unknown_list:
        logger.debug(
            "  [GUARD] All %d selected column(s) verified in schema -- SQL is clean",
            len(selected_cols),
        )
        return optimized_sql, True

    unknown_set = set(unknown_list)
    logger.warning(
        "  [GUARD] Hallucinated column(s) detected: %s  |  known cols: %s",
        sorted(unknown_set), sorted(all_known),
    )

    # Step 6 -- strip unknowns from the SELECT list
    repaired = _strip_unknown_columns(optimized_sql, unknown_set)

    if repaired is None:
        logger.warning(
            "  [GUARD] Could not strip unknown columns (complex SQL) "
            "-- falling back to original query"
        )
        return original_sql, False

    # Step 7 -- re-verify: confirm no unknown col survived the strip
    repaired_cols = _extract_selected_columns(repaired)
    still_bad = [c for c in repaired_cols if c in unknown_set]

    if still_bad:
        logger.warning(
            "  [GUARD] Strip incomplete -- %s still present "
            "-- falling back to original query", still_bad,
        )
        return original_sql, False

    if not repaired_cols:
        logger.warning(
            "  [GUARD] Strip left empty SELECT list "
            "-- falling back to original query"
        )
        return original_sql, False

    logger.warning(
        "  [GUARD] Found unknown columns: %s, stripped. col_valid=True\n"
        "  [GUARD] SQL after guard: %s",
        sorted(unknown_set), repaired,
    )
    return repaired, True



def _execute_query_for_ab_test(
    session_id: str,
    sql_query: str,
    limit: int = 50
) -> dict:
    """
    Execute a query safely and return timing + rows/columns for A/B testing.

    Uses EXPLAIN ANALYZE to get real PostgreSQL execution time, then runs
    the actual query (with LIMIT) to fetch the result rows.

    Always returns a dict with these guaranteed keys:
        success           : bool
        execution_time_ms : float   (0.0 if unavailable)
        columns           : list
        rows              : list[dict]   ← frontend uses this key
        error             : str | None
    """
    if not session_id:
        return {
            "success":           False,
            "execution_time_ms": 0.0,
            "columns":           [],
            "rows":              [],
            "error":             "No session_id provided",
        }

    db_service = SecureDatabaseService()

    # ── Step 1: EXPLAIN ANALYZE → real execution time ─────────────────────────
    execution_time_ms = 0.0
    try:
        explain_result = db_service.run_secure_explain(session_id, sql_query, analyze=True)
        if explain_result.get("success"):
            plan = explain_result.get("explain_plan", [])
            if isinstance(plan, list) and plan:
                execution_time_ms = float(plan[0].get("Execution Time", 0) or 0)
    except Exception as exc:
        logger.warning("  ⚠ EXPLAIN ANALYZE failed (timing unavailable): %s", exc)

    # ── Step 2: actual SELECT → rows + columns ────────────────────────────────
    try:
        data_result = db_service.run_query_with_metrics(session_id, sql_query, limit=limit)

        # run_query_with_metrics may return 'rows' or 'data' depending on version
        rows    = data_result.get("rows") or data_result.get("data") or []
        columns = data_result.get("columns") or []

        # If EXPLAIN timing was 0, fall back to wall-clock time from run_query
        if execution_time_ms == 0.0:
            execution_time_ms = float(data_result.get("execution_time_ms", 0) or 0)

        logger.info(
            "  ✓ Executed query: %d rows, %.2f ms",
            len(rows), execution_time_ms
        )
        return {
            "success":           True,
            "execution_time_ms": execution_time_ms,
            "columns":           columns,
            "rows":              rows,
            "error":             None,
        }

    except Exception as exc:
        logger.warning("  ⚠ Query execution failed: %s", exc)
        return {
            "success":           False,
            "execution_time_ms": execution_time_ms,
            "columns":           [],
            "rows":              [],
            "error":             str(exc),
        }


# ---------------------------------------------------------------------------
# EXPLAIN plan helper
# ---------------------------------------------------------------------------

def _get_explain_plan(session_id: str, sql_query: str) -> str:
    """
    Run EXPLAIN (FORMAT JSON, VERBOSE) on the query and return the raw EXPLAIN
    plan as a JSON string, suitable for AIOptimizer.optimize(explain_json=...)
    and ContextBuilder, both of which parse this value with json.loads().

    IMPORTANT: this must stay valid JSON. Do NOT return the human-readable
    "=== QUERY PLAN ===" summary here — that caused
    "ContextBuilder: could not parse EXPLAIN JSON: Expecting value: line 1
    column 1 (char 0)" because the consumer tried to json.loads() plain text.
    Use _format_explain_summary() below if a human-readable version is needed
    (e.g. for logs or a UI panel) — call it separately, don't substitute it
    here.

    Returns "" on failure so the pipeline degrades gracefully if EXPLAIN is
    unavailable (never raises).
    """
    if not session_id:
        return ""
    try:
        db_service = SecureDatabaseService()
        explain_result = db_service.run_secure_explain(
            session_id, sql_query, analyze=False   # FORMAT JSON, no actual execution
        )
        if not explain_result.get("success"):
            logger.warning("  ⚠ EXPLAIN failed: %s", explain_result.get("error"))
            return ""

        plan = explain_result.get("explain_plan")
        if not plan:
            return ""

        # plan is typically a list of dicts from psycopg2:
        #   [{"Plan": {...}, "Planning Time": ..., "Execution Time": ...}]
        # Normalize to the root dict, then serialize back to a JSON string
        # since that's the contract AIOptimizer/ContextBuilder expect.
        if isinstance(plan, list) and plan:
            root = plan[0]
        elif isinstance(plan, dict):
            root = plan
        elif isinstance(plan, str):
            # Some drivers already hand back a JSON string — validate it
            # parses, then pass it straight through.
            try:
                json.loads(plan)
                logger.info("  ✓ EXPLAIN plan captured (raw JSON string, %d chars)", len(plan))
                return plan
            except (json.JSONDecodeError, ValueError):
                logger.warning("  ⚠ EXPLAIN plan string was not valid JSON — dropping")
                return ""
        else:
            return ""

        total_cost = root.get("Plan", {}).get("Total Cost", "?")
        explain_json_str = json.dumps(root)
        logger.info(
            "  ✓ EXPLAIN plan captured (%d chars, total_cost=%s)",
            len(explain_json_str), total_cost,
        )
        return explain_json_str

    except Exception as exc:
        logger.warning("  ⚠ _get_explain_plan exception (non-fatal): %s", exc)
        return ""


def _format_explain_summary(explain_json_str: str) -> str:
    """
    Convert a raw EXPLAIN JSON string into the compact, human-readable
    multi-line summary previously produced inline by _get_explain_plan().
    Use this for logs / UI display — NOT for the AIOptimizer/ContextBuilder
    explain_json= contract, which needs real JSON (see _get_explain_plan).

    Returns "" on any parse failure (never raises).
    """
    if not explain_json_str:
        return ""
    try:
        root = json.loads(explain_json_str)
    except (json.JSONDecodeError, ValueError):
        logger.warning("_format_explain_summary: could not parse EXPLAIN JSON")
        return ""

    lines = ["=== QUERY PLAN (EXPLAIN JSON) ==="]
    total_cost   = root.get("Plan", {}).get("Total Cost", "?")
    startup_cost = root.get("Plan", {}).get("Startup Cost", "?")
    lines.append(f"Total cost estimate : {total_cost}")
    lines.append(f"Startup cost        : {startup_cost}")

    def _walk(node: dict, depth: int = 0) -> None:
        indent    = "  " * depth
        node_type = node.get("Node Type", "?")
        rel       = node.get("Relation Name", "")
        alias     = node.get("Alias", "")
        node_cost = node.get("Total Cost", "?")
        rows      = node.get("Plan Rows", "?")
        join_type = node.get("Join Type", "")

        rel_str  = f" on {rel}" + (f" ({alias})" if alias and alias != rel else "") if rel else ""
        join_str = f" [{join_type} JOIN]" if join_type else ""
        lines.append(f"{indent}{node_type}{rel_str}{join_str}  cost={node_cost}  rows≈{rows}")

        for child in node.get("Plans", []):
            _walk(child, depth + 1)

    _walk(root.get("Plan", {}))
    lines.append("=== END QUERY PLAN ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Streaming entry point (used by main.py SSE endpoint)
# ---------------------------------------------------------------------------

def run_pipeline_streaming(
    sql_query: str,
    db_stats: dict | None = None,
    session_id: str = "",
) -> Generator[dict, None, None]:
    """
    Sync generator — runs the 5-stage pipeline and yields one event dict per
    stage so the caller can stream SSE chunks to the Angular frontend.

    Each yielded dict:
        { "stage": int, "label": str, "status": str, "data": dict }

    stage=0 status="complete"    → final payload, always emitted last
    stage=0 status="fatal_error" → pipeline aborted at stage 1
    """

    STAGES = [
        (1, "Query Analysis"),
        (2, "Statistics Extraction"),
        (3, "Index Recommendations"),
        (4, "Rule-Based Analysis"),
        (5, "AI Optimization"),
    ]

    result: dict = {
        "success":            False,
        "query":              sql_query.strip(),
        "analysis":           {},
        "statistics":         {},
        "index_suggestions":  [],
        "rule_violations":    [],
        "optimized_sql":      "",
        "ai_explanation":     "",
        # Compact "Table: X / Columns: ..." text block (ai.schema_formatter)
        # that was actually shown to the LLM for this run — surfaced for
        # UI/debugging so it's easy to see exactly what schema context the
        # optimizer had available.
        "schema_context":     "",
        "errors":             [],
        # Execution A/B data — always present in final payload
        "original_execution":  None,
        "optimized_execution": None,
        "original_time_ms":    None,
        "optimized_time_ms":   None,
        "speedup_pct":         0.0,
    }

    logger.info("=" * 60)
    logger.info("Streaming pipeline started")

    # ── Stage 1 — Query Analysis ──────────────────────────────────────────────
    stage_num, stage_label = STAGES[0]
    yield {"stage": stage_num, "label": stage_label, "status": "running", "data": {}}
    try:
        analysis = QueryAnalyzer.analyze_full(sql_query)
        result["analysis"] = analysis
        logger.info("  ✓ Stage 1 complete")
        yield {"stage": stage_num, "label": stage_label, "status": "done",
               "data": {"analysis": analysis}}
    except Exception as exc:
        msg = f"Query analysis failed: {exc}"
        logger.error("  ✗ %s", msg)
        result["errors"].append(msg)
        yield {"stage": stage_num, "label": stage_label, "status": "error",
               "data": {"error": msg}}
        yield {"stage": 0, "label": "Pipeline", "status": "fatal_error",
               "data": {**result, "errors": result["errors"]}}
        return

    # ── Stage 2 — Statistics Extraction ──────────────────────────────────────
    stage_num, stage_label = STAGES[1]
    yield {"stage": stage_num, "label": stage_label, "status": "running", "data": {}}
    statistics = {}
    try:
        if db_stats is not None:
            statistics = db_stats
        else:
            extractor = StatisticsExtractor()
            statistics = extractor.get_comprehensive_statistics(
                session_id, table_name=None
            )
        result["statistics"] = statistics
        logger.info("  ✓ Stage 2 complete (%d tables)", len(statistics))
        yield {"stage": stage_num, "label": stage_label, "status": "done",
               "data": {"statistics": statistics}}
    except Exception as exc:
        msg = f"Statistics extraction failed: {exc}"
        logger.warning("  ⚠ %s — continuing with empty stats", msg)
        result["errors"].append(msg)
        yield {"stage": stage_num, "label": stage_label, "status": "error",
               "data": {"error": msg}}

    # ── Stage 3 — Index Recommendations ──────────────────────────────────────
    stage_num, stage_label = STAGES[2]
    yield {"stage": stage_num, "label": stage_label, "status": "running", "data": {}}
    try:
        suggestions = generate_index_suggestions(sql_query)
        result["index_suggestions"] = suggestions
        logger.info("  ✓ Stage 3 complete (%d suggestions)", len(suggestions))
        yield {"stage": stage_num, "label": stage_label, "status": "done",
               "data": {"index_suggestions": suggestions}}
    except Exception as exc:
        msg = f"Index recommendation failed: {exc}"
        logger.warning("  ⚠ %s", msg)
        result["errors"].append(msg)
        yield {"stage": stage_num, "label": stage_label, "status": "error",
               "data": {"error": msg}}

    # ── Stage 4 — Rule-Based Optimization ────────────────────────────────────
    stage_num, stage_label = STAGES[3]
    yield {"stage": stage_num, "label": stage_label, "status": "running", "data": {}}
    try:
        violations = run_all_rules(sql_query, statistics)
        serialized = _serialize_violations(violations)
        result["rule_violations"] = serialized
        logger.info("  ✓ Stage 4 complete (%d violations)", len(serialized))
        yield {"stage": stage_num, "label": stage_label, "status": "done",
               "data": {"rule_violations": serialized}}
    except Exception as exc:
        msg = f"Rule engine failed: {exc}"
        logger.warning("  ⚠ %s", msg)
        result["errors"].append(msg)
        yield {"stage": stage_num, "label": stage_label, "status": "error",
               "data": {"error": msg}}

    # ── Stage 4.5 — Execute Original Query (baseline A/B) ────────────────────
    logger.info("  ⚡ Executing original query for baseline metrics…")
    original_exec = _execute_query_for_ab_test(session_id, sql_query, limit=50)
    result["original_execution"] = original_exec
    if original_exec["success"]:
        logger.info("  ✓ Original: %d rows in %.2f ms",
                    len(original_exec["rows"]), original_exec["execution_time_ms"])
    else:
        logger.warning("  ⚠ Original query execution failed: %s", original_exec.get("error"))

    # ── Stage 4.6 — EXPLAIN plan for LLM context ─────────────────────────────
    # Run EXPLAIN (FORMAT JSON) BEFORE the LLM so Mistral can see exactly
    # which nodes are expensive (Seq Scan, Nested Loop, high cost) and make
    # targeted rewrites rather than guessing.
    logger.info("  ⚡ Running EXPLAIN to build LLM query-plan context…")
    explain_plan_text = _get_explain_plan(session_id, sql_query)

    # ── Stage 5 — LLM AI Optimization ────────────────────────────────────────
    stage_num, stage_label = STAGES[4]
    yield {"stage": stage_num, "label": stage_label, "status": "running", "data": {}}
    try:
        # Build schema context from Stage 2 statistics
        table_names = _extract_table_names(sql_query)

        # Normalize stats shape before passing to ContextBuilder.
        # This converts any index strings → dicts and aliases reltuples → row_count
        # so ContextBuilder.from_statistics_dict() never receives raw strings.
        normalized_stats = _normalize_statistics(
            result.get("statistics", {}), table_names
        )
        logger.debug("  Normalized stats keys: %s", list(normalized_stats.keys()))

        db_context = ContextBuilder.from_statistics_dict(
            normalized_stats, table_names
        ) if table_names else None

        # ── PRE-LLM: fetch live schema and inject it into db_context ─────────
        # This is the key fix: _get_live_schema() was previously only called
        # AFTER the LLM rewrite (for validation). That meant the LLM's prompt
        # always had an empty "Columns:" section, forcing Mistral to guess
        # column names from training data (causing hallucinations like
        # "SELECT orders FROM orders"). By running this BEFORE the LLM call
        # and injecting the real columns into db_context, the LLM gets the
        # full accurate schema and can safely expand SELECT * or reference
        # any column. The same live_schema dict is then reused for validation
        # — zero extra DB round-trips.
        live_schema: dict = {}
        raw_full_schema: dict = {}
        if session_id and table_names:
            live_schema, raw_full_schema = _get_live_schema_full(session_id, table_names)
            if live_schema and db_context:
                # Inject real columns (+ PK/FK metadata) into every TableContext
                # that ContextBuilder already built from statistics.
                from ai.ai_optimizer import ColumnInfo
                for tbl in db_context.tables:
                    tkey = tbl.name.lower()
                    if tkey not in live_schema:
                        continue
                    schema_info = live_schema[tkey]
                    pk_set = set(schema_info.get("primary_key", []))
                    fk_map = {
                        fk["column"]: f"{fk['ref_table']}.{fk['ref_column']}"
                        for fk in schema_info.get("foreign_keys", [])
                    }
                    tbl.columns = [
                        ColumnInfo(
                            name=col_name,
                            data_type=col_type,
                            nullable=True,        # nullability not in _get_live_schema yet
                            is_pk=(col_name in pk_set),
                            is_fk=(col_name in fk_map),
                            fk_ref=fk_map.get(col_name, ""),
                        )
                        for col_name, col_type in schema_info["columns"].items()
                    ]
                    logger.debug(
                        "  Schema injected for '%s': %d columns, %d PKs, %d FKs",
                        tbl.name, len(tbl.columns), len(pk_set), len(fk_map),
                    )
            elif live_schema and not db_context:
                # No stats were available but we do have a live schema —
                # build a minimal DbContext from the schema alone so the LLM
                # still gets column info.
                from ai.ai_optimizer import DbContext, TableContext, ColumnInfo
                tables = []
                for tname in table_names:
                    tkey = tname.lower()
                    schema_info = live_schema.get(tkey, {})
                    pk_set = set(schema_info.get("primary_key", []))
                    fk_map = {
                        fk["column"]: f"{fk['ref_table']}.{fk['ref_column']}"
                        for fk in schema_info.get("foreign_keys", [])
                    }
                    columns = [
                        ColumnInfo(
                            name=col_name,
                            data_type=col_type,
                            nullable=True,
                            is_pk=(col_name in pk_set),
                            is_fk=(col_name in fk_map),
                            fk_ref=fk_map.get(col_name, ""),
                        )
                        for col_name, col_type in schema_info.get("columns", {}).items()
                    ]
                    tables.append(TableContext(name=tname, columns=columns))
                db_context = DbContext(tables=tables)

        # Build real_columns for the post-rewrite validator from the live schema
        # we already fetched above — no second DB query needed.
        real_columns: dict[str, set[str]] = {
            t: set(info["columns"].keys()) for t, info in live_schema.items()
        } if live_schema else {}

        # Compact schema text (ai.schema_formatter) built from the SAME
        # SchemaService call as live_schema above — zero extra DB round-trips.
        # This is the human-readable block a text-to-sql prompt would use;
        # here it's surfaced on the result mainly for UI/debugging visibility
        # into exactly what schema context this pipeline run had available.
        schema_text = _build_schema_text(raw_full_schema, table_names)
        if schema_text:
            result["schema_context"] = schema_text
            logger.debug("  Schema context for LLM prompt:\n%s", schema_text)

        # Pass the EXPLAIN plan to the optimizer so the LLM sees real costs
        ai_result = AIOptimizer().optimize(
            sql_query,
            db_context=db_context,
            explain_json=explain_plan_text,
        )
        if ai_result.success:
            # ── Clean the LLM output and validate it ──────────────────────
            optimized_sql_candidate = _clean_optimized_sql(
                ai_result.optimized_sql, sql_query
            )

            # ── Stage 5.5 — Execute Optimized Query ──────────────────────────
            logger.info("  ⚡ Executing optimized query for performance comparison…")

            # Additional validation (the cleaned SQL should already be valid)
            if not _is_valid_sql(optimized_sql_candidate):
                logger.warning(
                    "  ⚠ Optimized SQL failed final validation — skipping DB execution. "
                    "Value was: %r", optimized_sql_candidate[:200]
                )
                result["optimized_execution"] = {
                    "success":           False,
                    "execution_time_ms": 0.0,
                    "columns":           [],
                    "rows":              [],
                    "error":             "Extracted SQL did not pass SELECT/WITH sanity check",
                }
                result["optimized_sql"]  = sql_query
                result["ai_explanation"] = _clean_llm_text(ai_result.explanation)
            else:
                # ── Column-existence guard ────────────────────────────────────
                # Reuse real_columns from the pre-LLM live schema fetch above.
                # If live schema wasn't available (empty dict), _validate_optimized_columns
                # skips validation and trusts the LLM output — same safe fallback as before.
                optimized_sql_candidate, col_valid = _validate_optimized_columns(
                    optimized_sql_candidate, sql_query, real_columns, table_names
                )

                if not col_valid:
                    # Rewrite references non-existent columns — skip DB execution
                    note = (
                        "LLM rewrote the query with column names that do not exist "
                        "in this database — returning the original query unchanged."
                    )
                    result["optimized_sql"]  = sql_query
                    result["ai_explanation"] = (
                        _clean_llm_text(ai_result.explanation) + "\n\n" + f"Note: {note}"
                    )
                    result["optimized_execution"] = {
                        "success":           False,
                        "execution_time_ms": 0.0,
                        "columns":           [],
                        "rows":              [],
                        "error":             note,
                    }
                    logger.info("  ℹ %s", note)
                else:
                    optimized_exec = _execute_query_for_ab_test(
                        session_id, optimized_sql_candidate, limit=50
                    )
                    result["optimized_execution"] = optimized_exec

                    orig_ms = (result["original_execution"] or {}).get("execution_time_ms") or 0.0
                    opt_ms  = optimized_exec.get("execution_time_ms") or 0.0

                    if optimized_exec["success"]:
                        logger.info(
                            "  ✓ Optimized: %d rows in %.2f ms",
                            len(optimized_exec["rows"]), opt_ms,
                        )
                    else:
                        logger.warning(
                            "  ⚠ Optimized query execution failed: %s",
                            optimized_exec.get("error"),
                        )

                    # ── Return the FASTER query ───────────────────────────────
                    # If the LLM rewrite is slower or failed, serve the original.
                    if optimized_exec["success"] and opt_ms < orig_ms:
                        result["optimized_sql"]  = optimized_sql_candidate
                        result["ai_explanation"] = _clean_llm_text(ai_result.explanation)
                        logger.info(
                            "  ✓ LLM rewrite is faster — serving optimized query "
                            "(%.2f ms → %.2f ms, %.1f%% speedup)",
                            orig_ms, opt_ms, ((orig_ms - opt_ms) / orig_ms) * 100,
                        )
                    else:
                        result["optimized_sql"] = sql_query
                        reason = (
                            f"LLM rewrite ({opt_ms:.1f} ms) was not faster than original "
                            f"({orig_ms:.1f} ms) — returning original query."
                            if optimized_exec["success"]
                            else "LLM rewrite execution failed — returning original query."
                        )
                        result["ai_explanation"] = (
                            _clean_llm_text(ai_result.explanation) + "\n\n" +
                            f"Note: {reason}"
                        )
                        logger.info("  ℹ %s", reason)

            logger.info("  ✓ Stage 5 complete")
            yield {"stage": stage_num, "label": stage_label, "status": "done",
                   "data": {
                       "optimized_sql":  result["optimized_sql"],
                       "ai_explanation": result["ai_explanation"],
                   }}
        else:
            msg = f"LLM optimization did not succeed: {ai_result.error}"
            logger.warning("  ⚠ %s", msg)
            result["errors"].append(msg)
            yield {"stage": stage_num, "label": stage_label, "status": "error",
                   "data": {"error": msg}}

    except Exception as exc:
        msg = f"LLM optimization failed: {exc}"
        logger.warning("  ⚠ %s\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)
        yield {"stage": stage_num, "label": stage_label, "status": "error",
               "data": {"error": msg}}

    # ── Calculate speedup ─────────────────────────────────────────────────────
    orig    = result["original_execution"]
    opt     = result["optimized_execution"]
    orig_ms = orig["execution_time_ms"] if orig else None
    opt_ms  = opt["execution_time_ms"]  if opt  else None

    result["original_time_ms"]  = orig_ms
    result["optimized_time_ms"] = opt_ms

    if orig_ms and opt_ms and orig_ms > 0:
        result["speedup_pct"] = ((orig_ms - opt_ms) / orig_ms) * 100
        logger.info("  📊 Speedup: %.1f%% (%.2f ms → %.2f ms)",
                    result["speedup_pct"], orig_ms, opt_ms)

    # ── Final complete event ──────────────────────────────────────────────────
    result["success"] = True
    logger.info("Streaming pipeline finished")
    logger.info("=" * 60)

    # --- FIX: convert any Decimal / non-serialisable objects before yielding ---
    safe_result = _make_json_safe(result)

    try:
        # Verify serialisability
        json.dumps(safe_result)
        yield {"stage": 0, "label": "Complete", "status": "complete", "data": safe_result}
    except (TypeError, ValueError) as e:
        logger.error(f"Final payload is not JSON-serialisable: {e}")
        yield {"stage": 0, "label": "Complete", "status": "error",
               "data": {"message": "Internal serialisation error"}}


# ---------------------------------------------------------------------------
# Original blocking entry point (kept for backward compat / tests)
# ---------------------------------------------------------------------------

def run_pipeline(sql_query: str, db_stats: dict | None = None, session_id: str = "") -> dict:
    """
    Blocking version of the pipeline (no streaming).
    Returns the full result dict when everything is done.
    """
    result = {
        "success":           False,
        "query":             sql_query.strip(),
        "analysis":          {},
        "statistics":        {},
        "index_suggestions": [],
        "rule_violations":   [],
        "optimized_sql":     "",
        "ai_explanation":    "",
        "schema_context":    "",
        "errors":            [],
    }

    logger.info("=" * 60)
    logger.info("Pipeline started (blocking)")

    # Stage 1 — Query Analysis
    logger.info("[Stage 1/5] Query analysis")
    try:
        result["analysis"] = QueryAnalyzer.analyze_full(sql_query)
        logger.info("  ✓ Complete")
    except Exception as exc:
        msg = f"Query analysis failed: {exc}"
        logger.error("  ✗ %s\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)
        return result

    # Stage 2 — Statistics Extraction
    logger.info("[Stage 2/5] Statistics extraction")
    statistics = {}
    try:
        if db_stats is not None:
            statistics = db_stats
        else:
            statistics = StatisticsExtractor().get_comprehensive_statistics(
                session_id, table_name=None
            )
        result["statistics"] = statistics
        logger.info("  ✓ Complete (%d tables)", len(statistics))
    except Exception as exc:
        msg = f"Statistics extraction failed: {exc}"
        logger.warning("  ⚠ %s — continuing with empty stats\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)

    # Stage 3 — Index Recommendations
    logger.info("[Stage 3/5] Index recommendation")
    try:
        result["index_suggestions"] = generate_index_suggestions(sql_query)
        logger.info("  ✓ %d suggestion(s)", len(result["index_suggestions"]))
    except Exception as exc:
        msg = f"Index recommendation failed: {exc}"
        logger.warning("  ⚠ %s\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)

    # Stage 4 — Rule-Based Optimization
    logger.info("[Stage 4/5] Rule-based analysis")
    try:
        result["rule_violations"] = _serialize_violations(
            run_all_rules(sql_query, statistics)
        )
        logger.info("  ✓ %d violation(s)", len(result["rule_violations"]))
    except Exception as exc:
        msg = f"Rule engine failed: {exc}"
        logger.warning("  ⚠ %s\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)

    # Stage 5 — LLM AI Optimization
    logger.info("[Stage 5/5] LLM AI optimization")
    try:
        table_names = _extract_table_names(sql_query)
        normalized_stats = _normalize_statistics(
            result.get("statistics", {}), table_names
        )
        db_context = ContextBuilder.from_statistics_dict(
            normalized_stats, table_names
        ) if table_names else None

        ai_result = AIOptimizer().optimize(sql_query, db_context=db_context)
        if ai_result.success:
            # Use the same guard as streaming version
            candidate = _clean_optimized_sql(ai_result.optimized_sql, sql_query)
            result["optimized_sql"]  = candidate if _is_valid_sql(candidate) else sql_query
            result["ai_explanation"] = _clean_llm_text(ai_result.explanation)
            logger.info("  ✓ Complete")
        else:
            msg = f"LLM optimization did not succeed: {ai_result.error}"
            logger.warning("  ⚠ %s", msg)
            result["errors"].append(msg)
    except Exception as exc:
        msg = f"LLM optimization failed: {exc}"
        logger.warning("  ⚠ %s\n%s", msg, traceback.format_exc())
        result["errors"].append(msg)

    result["success"] = True
    _log_summary(result)
    logger.info("Pipeline finished — OK")
    logger.info("=" * 60)
    return result


def _log_summary(result: dict) -> None:
    logger.info("--- Pipeline Summary ---")
    logger.info("  Index suggestions : %d", len(result["index_suggestions"]))
    logger.info("  Rule violations   : %d", len(result["rule_violations"]))
    logger.info("  LLM optimized     : %s", "yes" if result["optimized_sql"] else "no")
    if result["errors"]:
        logger.info("  Non-fatal errors  : %d", len(result["errors"]))
        for e in result["errors"]:
            logger.info("    • %s", e)
    logger.info("  Overall status    : %s", "OK" if result["success"] else "FAILED")

# ---------------------------------------------------------------------------
# Text-to-SQL pipeline (Phase 3 — Basic Relevant Table Selector, no ChromaDB)
# ---------------------------------------------------------------------------

def _get_full_schema_context(session_id: str) -> dict:
    """
    Fetch the full live schema for a session (ALL tables, not scoped to any
    particular SQL query) in the shape ai/table_selector.py and
    ai/text_to_sql_builder.py already understand:

        {table_name: {"columns": [...], "foreign_keys": [...], ...}}

    Unlike _get_live_schema() (used by the optimizer pipeline), this does
    NOT take a table_names filter — Text-to-SQL doesn't know which tables
    are relevant yet; that's exactly what select_relevant_tables_basic()
    figures out from this full schema.

    OPEN ITEM (unresolved): SchemaService.get_schema() currently returns an
    empty dict for BOTH "session has no tables" and "session_id is
    invalid/expired" — schema_router.py has the same 200 {} ambiguity.
    Until that's disambiguated upstream, this helper can only report "no
    schema available", not distinguish the two cases.

    Returns {} on any failure — callers must treat that as "cannot proceed",
    never crash.
    """
    if not session_id:
        logger.warning("_get_full_schema_context: no session_id provided")
        return {}

    try:
        schema_service = get_schema_service()
        full_schema = schema_service.get_schema(session_id, table_names=None)
        if not full_schema:
            logger.warning(
                "_get_full_schema_context: SchemaService returned no data for "
                "session %s… (empty schema OR invalid/expired session — "
                "see open item in docstring)", session_id[:8],
            )
            return {}
        return full_schema
    except Exception as exc:
        logger.warning("_get_full_schema_context: schema fetch failed (non-fatal): %s", exc)
        return {}


def run_text_to_sql_pipeline(
    question: str,
    session_id: str = "",
    top_k: int = 5,
) -> dict:
    """
    End-to-end Text-to-SQL pipeline (Phase 3):

      1. Fetch full live schema for the session.
      2. select_relevant_tables_basic() — keyword/fuzzy table selection
         (Phase 1 selector; swap-in point for ChromaDB semantic search
         later without changing this function's contract).
      3. generate_sql_from_question() (ai/text_to_sql_builder.py) — builds
         the schema-scoped prompt and calls the LLM.
      4. Safety-gate the generated SQL with the SAME SQLSafetyValidator
         the optimizer pipeline uses — text-to-sql output is not exempt.

    Returns a dict matching routers/text_to_sql_router.py's
    TextToSqlResponse contract exactly:

        {
          "success":          bool,
          "question":         str,
          "generated_sql":    str | None,
          "detected_tables":  list[str],
          "retrieval_method": str,   # "keyword_match" | "no_schema" | "error"
          "validation":       {"is_valid": bool, "message": str | None},
          "warnings":         list[str],
          "error":            str | None,
        }

    Never raises — every failure path returns a well-formed dict so the
    router can serialise it directly into TextToSqlResponse.
    """
    warnings: list[str] = []
    question = (question or "").strip()

    base_response = {
        "success":          False,
        "question":         question,
        "generated_sql":    None,
        "detected_tables":  [],
        "retrieval_method": "error",
        "validation":       {"is_valid": False, "message": None},
        "warnings":         warnings,
        "error":            None,
    }

    # ── Guard 1: empty question ─────────────────────────────────────────────
    if not question:
        base_response["validation"]["message"] = "Question cannot be empty"
        base_response["error"] = "Empty query question provided."
        logger.warning("run_text_to_sql_pipeline: empty question")
        return base_response

    # ── Guard 2: no session ─────────────────────────────────────────────────
    if not session_id:
        base_response["validation"]["message"] = "No active database session"
        base_response["error"] = "X-Session-ID is required for text-to-SQL."
        logger.warning("run_text_to_sql_pipeline: missing session_id")
        return base_response

    # ── Step 1: full schema ──────────────────────────────────────────────────
    schema_context = _get_full_schema_context(session_id)
    if not schema_context:
        base_response["retrieval_method"] = "no_schema"
        base_response["validation"]["message"] = (
            "No schema available for this session (empty database, or the "
            "session is invalid/expired)."
        )
        base_response["error"] = "Could not load database schema."
        logger.warning(
            "run_text_to_sql_pipeline: no schema for session %s…", session_id[:8]
        )
        return base_response

    # ── Step 2: relevant table selection (Phase 1 — keyword/fuzzy) ──────────
    relevant_tables = select_relevant_tables_basic(
        question=question,
        schema_context=schema_context,
        limit=max(1, top_k),
    )
    base_response["retrieval_method"] = "keyword_match"

    if not relevant_tables:
        base_response["validation"]["message"] = (
            "Could not identify any relevant tables for this question. "
            "Try mentioning a table or column name explicitly."
        )
        base_response["error"] = "No relevant tables found."
        logger.warning(
            "run_text_to_sql_pipeline: table selector returned [] for "
            "question=%r (session %s…)", question, session_id[:8],
        )
        return base_response

    base_response["detected_tables"] = relevant_tables
    logger.info(
        "run_text_to_sql_pipeline: selected tables %s for question=%r",
        relevant_tables, question,
    )

    # ── Step 3: LLM SQL generation (ai/text_to_sql_builder.py) ──────────────
    # ASSUMPTION: generate_sql_from_question(question, session_id, table_names)
    # — adjust this call if Phase 2's actual signature differs.
    try:
        generated_sql = generate_sql_from_question(
            question=question,
            session_id=session_id,
            table_names=relevant_tables,
        )
    except LLMClientError as exc:
        base_response["validation"]["message"] = "SQL generation failed (LLM error)"
        base_response["error"] = f"LLM call failed: {exc}"
        logger.error("run_text_to_sql_pipeline: LLM call failed: %s", exc)
        return base_response
    except Exception as exc:
        base_response["validation"]["message"] = "SQL generation failed"
        base_response["error"] = str(exc)
        logger.error(
            "run_text_to_sql_pipeline: generate_sql_from_question failed: %s\n%s",
            exc, traceback.format_exc(),
        )
        return base_response

    generated_sql = (generated_sql or "").strip()
    if not generated_sql:
        base_response["validation"]["message"] = "LLM returned an empty SQL response"
        base_response["error"] = "Empty SQL generated."
        logger.warning("run_text_to_sql_pipeline: empty SQL from generate_sql_from_question")
        return base_response

    # ── Step 4: safety gate (same validator the optimizer pipeline uses) ────
    safe, violations = SQLSafetyValidator.is_safe(generated_sql)
    if not safe:
        base_response["validation"]["message"] = "Generated SQL failed safety check"
        base_response["error"] = f"Unsafe SQL rejected: {violations}"
        logger.error(
            "run_text_to_sql_pipeline: unsafe SQL generated for question=%r: %s",
            question, violations,
        )
        return base_response

    if not SQLSafetyValidator.is_select_only(generated_sql):
        base_response["validation"]["message"] = "Generated SQL is not a SELECT statement"
        base_response["error"] = "Only SELECT queries are supported by text-to-SQL."
        logger.error(
            "run_text_to_sql_pipeline: non-SELECT SQL generated for question=%r", question,
        )
        return base_response

    if not _is_valid_sql(generated_sql):
        warnings.append(
            "Generated SQL passed safety checks but failed a basic structural "
            "sanity check (missing FROM or too short) — review before running."
        )
        logger.warning(
            "run_text_to_sql_pipeline: generated SQL failed _is_valid_sql "
            "sanity check but was allowed through: %r", generated_sql[:200],
        )

    if len(relevant_tables) >= top_k:
        warnings.append(
            f"Table selection hit the top_k={top_k} limit — there may be "
            "additional relevant tables not included in this query."
        )

    # ── Success ────────────────────────────────────────────────────────────
    base_response["success"]          = True
    base_response["generated_sql"]    = generated_sql
    base_response["validation"]       = {"is_valid": True, "message": None}
    base_response["error"]            = None

    logger.info(
        "run_text_to_sql_pipeline: success — tables=%s, sql_len=%d",
        relevant_tables, len(generated_sql),
    )
    return base_response