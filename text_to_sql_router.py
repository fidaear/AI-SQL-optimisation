from fastapi import APIRouter, Header
from pydantic import BaseModel
from typing import List, Optional
import logging

from ai.text_to_sql_service import generate_sql, TextToSQLServiceError
from db.schema_service import get_schema_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["text-to-sql"])


class TextToSqlValidation(BaseModel):
    is_valid: bool
    message: Optional[str] = None


class TextToSqlRequest(BaseModel):
    question: str
    top_k: int = 5


class TextToSqlResponse(BaseModel):
    success: bool
    question: str
    generated_sql: Optional[str] = None
    detected_tables: List[str]
    retrieval_method: str
    validation: TextToSqlValidation
    warnings: List[str]
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers — table detection + schema-prompt formatting.
# NOTE: there is no dedicated table_selector / text_to_sql_builder module in
# this project yet. This is a minimal keyword-match implementation, kept
# local to this router until/if that logic gets its own module. Swap it out
# without touching the rest of the router if a smarter version is built.
# ---------------------------------------------------------------------------

def _detect_tables(question: str, all_table_names: list[str], top_k: int) -> list[str]:
    """
    Very simple keyword matching: a table is 'detected' if its name (singular
    or plural-stripped) appears as a substring of the question.
    Falls back to returning all known tables (capped at top_k) if nothing
    matches, so the LLM still has schema context to work with.
    """
    q = question.lower()
    matched = [t for t in all_table_names if t.lower().rstrip("s") in q or t.lower() in q]
    if not matched:
        matched = all_table_names[:top_k]
    return matched[:top_k] if top_k else matched


def _format_schema_prompt(schema: dict) -> str:
    """
    Render the RAW schema_service.get_schema() output into a plain-text block
    the LLM can use as DATABASE SCHEMA context.

    IMPORTANT: schema_service returns "columns" as a list[dict], each with
    at least "column_name" / "data_type" keys — e.g.:
        {"users": {"columns": [{"column_name": "id", "data_type": "integer"}, ...],
                    "primary_key": [...], "foreign_keys": [...]}}
    This is NOT already a {name: type} dict — that reshape only happens
    inside orchestrator.py's _get_live_schema() for its own internal use.
    Do not assume schema_service's raw output is pre-reshaped anywhere else.
    """
    lines = ["=== DATABASE SCHEMA ==="]
    for table_name, info in schema.items():
        lines.append(f"\nTABLE {table_name}")
        pk = set(info.get("primary_key", []))
        fk_map = {fk["column"]: f"{fk['ref_table']}.{fk['ref_column']}"
                  for fk in info.get("foreign_keys", [])}
        raw_columns = info.get("columns", [])
        for col in raw_columns:
            if isinstance(col, dict):
                col_name = col.get("column_name", "?")
                col_type = col.get("data_type", "text")
            else:
                # Defensive fallback in case a different shape shows up
                col_name, col_type = str(col), "text"
            flags = []
            if col_name in pk:
                flags.append("PK")
            if col_name in fk_map:
                flags.append(f"FK -> {fk_map[col_name]}")
            flag_str = f"  [{', '.join(flags)}]" if flags else ""
            lines.append(f"  {col_name}  {col_type}{flag_str}")
    lines.append("\n=== END DATABASE SCHEMA ===")
    return "\n".join(lines)


@router.post("/text-to-sql", response_model=TextToSqlResponse)
def text_to_sql(
    payload: TextToSqlRequest,
    x_session_id: str = Header(default="", alias="X-Session-ID"),
):
    """
    Translates a natural-language question into SQL using live DB schema
    context (via db.schema_service) and ai.text_to_sql_service.generate_sql().

    Requires X-Session-ID header — same session used by /api/optimize and
    /api/db/* — since schema introspection needs an active DB connection.
    """
    question = payload.question.strip()

    if not question:
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=[],
            retrieval_method="error",
            validation=TextToSqlValidation(is_valid=False, message="Question cannot be empty"),
            warnings=[],
            error="Empty query question provided."
        )

    if not x_session_id:
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=[],
            retrieval_method="error",
            validation=TextToSqlValidation(is_valid=False, message="Missing X-Session-ID header"),
            warnings=[],
            error="No active database session. Connect via /api/db/connect first."
        )

    warnings: list[str] = []

    # ── Step 1: get the full live schema for this session ────────────────
    try:
        schema_service = get_schema_service()
        full_schema = schema_service.get_schema(x_session_id)   # ASSUMPTION: no table_names -> all tables
    except Exception as exc:
        logger.error("Schema retrieval failed: %s", exc)
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=[],
            retrieval_method="failed_retrieval",
            validation=TextToSqlValidation(is_valid=False, message="Could not load database schema"),
            warnings=[],
            error=f"Schema retrieval error: {exc}"
        )

    if not full_schema:
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=[],
            retrieval_method="failed_retrieval",
            validation=TextToSqlValidation(is_valid=False, message="No tables found for this session"),
            warnings=[],
            error="Invalid/expired session or empty schema."
        )

    # ── Step 2: keyword-match table detection ────────────────────────────
    all_table_names = list(full_schema.keys())
    detected_tables = _detect_tables(question, all_table_names, payload.top_k)

    if not detected_tables:
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=[],
            retrieval_method="keyword_match",
            validation=TextToSqlValidation(is_valid=False, message="No relevant tables found for this question"),
            warnings=[],
            error="No matching tables in schema."
        )

    # ── Step 3: build schema prompt from the subset of tables detected ───
    relevant_schema = {t: full_schema[t] for t in detected_tables if t in full_schema}
    schema_prompt = _format_schema_prompt(relevant_schema)

    # ── Step 4: real LLM call ─────────────────────────────────────────────
    try:
        generated_sql = generate_sql(question=question, schema_prompt=schema_prompt)
    except TextToSQLServiceError as exc:
        logger.error("Text-to-SQL generation failed: %s", exc)
        return TextToSqlResponse(
            success=False,
            question=payload.question,
            generated_sql=None,
            detected_tables=detected_tables,
            retrieval_method="keyword_match",
            validation=TextToSqlValidation(is_valid=False, message="LLM generation failed"),
            warnings=[],
            error=str(exc)
        )

    # ── Step 5: lightweight validation ────────────────────────────────────
    is_valid = bool(generated_sql) and generated_sql.strip().upper().startswith("SELECT")
    if not is_valid:
        warnings.append("Generated output did not look like a valid SELECT statement.")

    return TextToSqlResponse(
        success=True,
        question=payload.question,
        generated_sql=generated_sql,
        detected_tables=detected_tables,
        retrieval_method="keyword_match",
        validation=TextToSqlValidation(
            is_valid=is_valid,
            message=None if is_valid else "Generated SQL failed basic sanity check"
        ),
        warnings=warnings,
        error=None
    )