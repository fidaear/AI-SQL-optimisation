"""
ai/schema_formatter.py
=======================
Compact Schema Formatter for LLM Prompts.

WHY THIS MODULE EXISTS
-----------------------
`SchemaService.get_schema()` (db/schema_service.py) returns a rich, fully
detailed dict per table — row counts, indexes, PK/FK flags on every column,
etc. That shape is great for an API response, but it is far too verbose
(and far too "JSON-shaped") to paste directly into an LLM prompt:

  * It wastes tokens (cost + latency) on fields the LLM never needs to
    write a query (row_count, index_name, is_nullable, …).
  * LLMs write better SQL from a short, DDL-like text block than from a
    dumped JSON structure.

This module takes the SchemaService output and renders ONLY what a
text-to-sql model actually needs: table names, column names, short types,
primary keys, and foreign key relationships — in the compact format:

    Table: customers
    Columns:
    * id integer primary key
    * name varchar
    * region varchar

    Table: orders
    Columns:
    * id integer primary key
    * customer_id integer references customers.id
    * total_amount numeric
    * order_date date

HOW TO USE
----------
    from db.schema_service import get_schema_service
    from ai.schema_formatter import SchemaFormatter

    schema_service = get_schema_service()
    raw_schema = schema_service.get_schema(session_id, table_names=["customers", "orders"])

    formatter = SchemaFormatter()
    schema_text = formatter.format(raw_schema, tables=["customers", "orders"])

    # schema_text is now ready to drop straight into your LLM prompt.

Notes
-----
* This module has NO database dependency — it is a pure function of the
  dict shape produced by SchemaService. This keeps it trivially unit
  testable (pass in a hand-built dict, assert the string output).
* Table/column ordering is deterministic (input dict order, or the order
  of the `tables` filter list if provided) so prompts are stable across
  runs — this matters for prompt-caching and for reproducible debugging.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SchemaFormatter:
    """
    Renders SchemaService.get_schema() output as a compact text block
    suitable for inclusion in an LLM prompt.
    """

    def __init__(self, max_columns_per_table: Optional[int] = None):
        """
        Args:
            max_columns_per_table:
                Optional safety cap. If a table has more columns than this,
                only the first N are rendered and a note is appended
                ("... (12 more columns omitted)"). Use this for very wide
                tables (50+ columns) so one table can't blow the prompt
                budget. Default None = render all columns.
        """
        self.max_columns_per_table = max_columns_per_table

    def format(
        self,
        schema: dict,
        tables: Optional[list[str]] = None,
    ) -> str:
        """
        Build the compact schema text block.

        Args:
            schema:
                The dict returned by SchemaService.get_schema(). Keyed by
                lowercase table name; each value has at least "columns"
                (list of column dicts with column_name / data_type /
                is_primary_key / foreign_key_ref) — see db/schema_service.py
                docstring for the exact shape.
            tables:
                Optional list of table names to include. Only these tables
                are rendered, in the order given — this is how callers keep
                the prompt scoped to the tables actually relevant to the
                user's question, instead of dumping the whole database.
                Names are matched case-insensitively. Unknown / missing
                table names are skipped with a debug log (never raise —
                a formatting helper should never crash the request).
                If None, every table present in `schema` is rendered
                (in dict order).

        Returns:
            A compact, human-readable text block. Returns "" if there is
            nothing to render (empty schema, or none of the requested
            tables exist).
        """
        if not schema:
            logger.debug("SchemaFormatter.format: empty schema — nothing to render")
            return ""

        table_names = self._resolve_table_names(schema, tables)
        if not table_names:
            logger.warning(
                "SchemaFormatter.format: none of the requested tables %s "
                "were found in schema (available: %s)",
                tables, list(schema.keys()),
            )
            return ""

        blocks = [
            self._format_table(tname, schema[tname])
            for tname in table_names
        ]
        return "\n\n".join(blocks)

    # ──────────────────────────────────────────────────────────────────
    # Internals
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_table_names(schema: dict, tables: Optional[list[str]]) -> list[str]:
        """
        Resolve the ordered, de-duplicated list of table names to render.
        Matches case-insensitively against schema's (lowercase) keys.
        """
        if not tables:
            return list(schema.keys())

        resolved: list[str] = []
        for t in tables:
            key = (t or "").strip().lower()
            if key in schema and key not in resolved:
                resolved.append(key)
            else:
                logger.debug(
                    "SchemaFormatter: requested table '%s' not found in schema — skipping",
                    t,
                )
        return resolved

    def _format_table(self, table_name: str, table_data: dict) -> str:
        """Render a single table's "Table: X / Columns: ..." block."""
        columns = table_data.get("columns") or []
        lines = [f"Table: {table_name}", "Columns:"]

        rendered = columns
        omitted = 0
        if self.max_columns_per_table is not None and len(columns) > self.max_columns_per_table:
            rendered = columns[: self.max_columns_per_table]
            omitted = len(columns) - self.max_columns_per_table

        for col in rendered:
            lines.append(f"* {self._format_column(col)}")

        if omitted:
            lines.append(f"* ... ({omitted} more columns omitted)")

        return "\n".join(lines)

    @staticmethod
    def _format_column(col: dict) -> str:
        """
        Render a single column as: "<name> <type> [primary key] [references ref_table.ref_column]"
        Matches the shape already annotated by SchemaService._annotate_columns().
        """
        name = col.get("column_name", "")
        data_type = col.get("data_type", "")
        parts = [name, data_type]

        if col.get("is_primary_key"):
            parts.append("primary key")

        fk_ref = col.get("foreign_key_ref")
        if fk_ref:
            parts.append(f"references {fk_ref}")

        return " ".join(p for p in parts if p)


# ── Module-level convenience function ─────────────────────────────────────

_default_formatter = SchemaFormatter()


def format_schema_for_prompt(
    schema: dict,
    tables: Optional[list[str]] = None,
) -> str:
    """
    Convenience wrapper around SchemaFormatter().format() for callers that
    don't need a custom max_columns_per_table setting.
    """
    return _default_formatter.format(schema, tables=tables)