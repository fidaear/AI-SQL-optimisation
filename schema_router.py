"""
routers/schema_router.py
========================
FastAPI router — exposes PostgreSQL schema metadata to the frontend
and to the upcoming Text-to-SQL pipeline.

Endpoints
---------
GET  /api/db/schema
    Returns full schema (all user tables) for the active session.
    Query param ?tables=orders,users  → restrict to those tables only.

GET  /api/db/schema/tables
    Returns only the list of table names (cheap discovery call).

Both endpoints:
  • Require X-Session-ID header (active database session).
  • Never expose credentials or read row data.
  • Return 400 when the session is missing/invalid.
  • Return 200 with an empty dict/list when the DB has no user tables yet.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Query

from db.schema_service import get_schema_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db", tags=["schema"])


# ── GET /api/db/schema/tables ─────────────────────────────────────────────────

@router.get("/schema/tables")
def list_tables(
    x_session_id: str = Header(default="", alias="X-Session-ID"),
):
    """
    Return all user-visible table names in the connected database.

    Response
    --------
    {"tables": ["orders", "products", "users", …]}
    """
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Session-ID header is required. Connect to a database first.",
        )

    service = get_schema_service()
    try:
        tables = service.get_table_names(x_session_id)
    except Exception as exc:
        logger.error("list_tables: unexpected error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Schema extraction failed: {exc}")

    return {"tables": tables}


# ── GET /api/db/schema ────────────────────────────────────────────────────────

@router.get("/schema")
def get_schema(
    x_session_id: str = Header(default="", alias="X-Session-ID"),
    tables: Optional[str] = Query(
        default=None,
        description="Comma-separated table names to filter. Omit to return all tables.",
    ),
):
    """
    Return full PostgreSQL schema metadata for the connected session.

    Query parameters
    ----------------
    tables : str (optional)
        Comma-separated table names, e.g. ``?tables=orders,users``.
        When omitted, all user tables in the 'public' schema are returned.

    Response shape
    --------------
    {
      "schema": {
        "orders": {
          "row_count": 12345,
          "columns": [
            {
              "column_name":    "id",
              "data_type":      "int",
              "is_nullable":    false,
              "is_primary_key": true,
              "foreign_key_ref": ""
            },
            …
          ],
          "indexes": [
            {"index_name": "orders_pkey", "columns": ["id"], "unique": true},
            …
          ],
          "primary_key":  ["id"],
          "foreign_keys": [
            {"column": "user_id", "ref_table": "users", "ref_column": "id"}
          ]
        },
        …
      },
      "table_count": 3,
      "session_id_prefix": "a1b2c3…"
    }
    """
    if not x_session_id:
        raise HTTPException(
            status_code=400,
            detail="X-Session-ID header is required. Connect to a database first.",
        )

    # Parse optional table filter
    requested_tables: Optional[list[str]] = None
    if tables:
        requested_tables = [t.strip().lower() for t in tables.split(",") if t.strip()]

    service = get_schema_service()
    try:
        schema = service.get_schema(x_session_id, table_names=requested_tables)
    except Exception as exc:
        logger.error("get_schema: unexpected error — %s", exc)
        raise HTTPException(status_code=500, detail=f"Schema extraction failed: {exc}")

    return {
        "schema":            schema,
        "table_count":       len(schema),
        # Never expose the full session ID — prefix only for debugging
        "session_id_prefix": x_session_id[:8] + "…" if x_session_id else "",
    }