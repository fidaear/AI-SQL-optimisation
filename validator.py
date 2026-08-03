"""
SQL Query Validator
Validates SQL queries against security rules and syntax constraints.

Public API (matches test_validator.py):
  ValidatorResult   — dataclass: is_valid, statement_type, error_message, blocked_reason
  SQLValidator      — static methods: validate(), is_read_only(), get_statement_type()
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ValidatorResult:
    """
    Outcome of a single validation run.

    Fields
    ------
    is_valid         : bool         — True only for safe SELECT queries
    statement_type   : str          — Detected statement keyword (SELECT, DROP, …)
    error_message    : str | None   — Short error description
    blocked_reason   : str | None   — Human-readable reason the query was blocked
    """
    is_valid: bool
    statement_type: str
    error_message: Optional[str] = None
    blocked_reason: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Validator
# ─────────────────────────────────────────────────────────────────────────────

class SQLValidator:
    """
    Validates SQL queries for safety before execution.

    Rules
    -----
    1. Query must be non-empty and non-whitespace.
    2. No semicolon-stacking (multiple statements).
    3. Statement must start with SELECT or WITH (CTEs).
    4. No DDL keywords  : DROP, CREATE, ALTER, TRUNCATE, RENAME.
    5. No DML keywords  : INSERT, UPDATE, DELETE, MERGE, UPSERT, REPLACE.
    6. No privilege/exec: GRANT, REVOKE, EXEC, EXECUTE, COPY, CALL.
    7. No comment-hidden injection: SQL with '; …' patterns outside strings.
    """

    # ── Keyword sets ──────────────────────────────────────────────────────────

    _DDL_KEYWORDS = {
        "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
        "COMMENT", "FLASHBACK", "PURGE",
    }

    _DML_KEYWORDS = {
        "INSERT", "UPDATE", "DELETE", "MERGE", "UPSERT", "REPLACE",
    }

    _DANGEROUS_KEYWORDS = {
        "GRANT", "REVOKE", "EXEC", "EXECUTE", "COPY", "CALL",
        "DECLARE", "DO",
    }

    _BLOCKED_KEYWORDS = _DDL_KEYWORDS | _DML_KEYWORDS | _DANGEROUS_KEYWORDS

    # ── Regex helpers ─────────────────────────────────────────────────────────

    # Strip single-line comments before type detection
    _SINGLE_LINE_COMMENT_RE = re.compile(r'--[^\n]*')

    # Detect semicolons that are NOT inside a string literal.
    # We can't perfectly parse SQL in regex, but we can catch the common
    # stacking pattern: any ';' that is followed by non-whitespace content.
    _SEMICOLON_RE = re.compile(r';')

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def get_statement_type(sql: str) -> str:
        """
        Return the first SQL keyword of a query in uppercase.

        Parameters
        ----------
        sql : str

        Returns
        -------
        str
            E.g. "SELECT", "DROP", "INSERT", "UNKNOWN"
        """
        # Strip single-line comments and leading whitespace
        cleaned = SQLValidator._SINGLE_LINE_COMMENT_RE.sub(' ', sql).strip()
        # Handle CTE: WITH … SELECT
        first_word = cleaned.split()[0].upper() if cleaned.split() else "UNKNOWN"
        return first_word

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def is_read_only(sql: str) -> bool:
        """
        Quick boolean check — True only if the statement is a SELECT / WITH.

        Parameters
        ----------
        sql : str

        Returns
        -------
        bool
        """
        stmt_type = SQLValidator.get_statement_type(sql)
        return stmt_type in ("SELECT", "WITH")

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def validate(sql: str) -> ValidatorResult:
        """
        Full validation pipeline.

        Parameters
        ----------
        sql : str
            Raw SQL string supplied by the caller.

        Returns
        -------
        ValidatorResult
            is_valid=True  → safe to execute.
            is_valid=False → blocked; see blocked_reason.
        """
        # ── Guard: empty / whitespace ─────────────────────────────────────────
        if not sql or not sql.strip():
            return ValidatorResult(
                is_valid=False,
                statement_type="UNKNOWN",
                error_message="Empty query",
                blocked_reason="Query is empty or contains only whitespace.",
            )

        # ── Guard: lone semicolons ────────────────────────────────────────────
        if sql.strip() in (";", ";;"):
            return ValidatorResult(
                is_valid=False,
                statement_type="UNKNOWN",
                error_message="Empty statement",
                blocked_reason="Query contains only a semicolon.",
            )

        # ── Detect statement type ─────────────────────────────────────────────
        stmt_type = SQLValidator.get_statement_type(sql)

        # ── Guard: blocked statement types ────────────────────────────────────
        if stmt_type in SQLValidator._BLOCKED_KEYWORDS:
            reason = (
                f"'{stmt_type}' statements are not allowed. "
                "Only read-only SELECT queries are permitted."
            )
            return ValidatorResult(
                is_valid=False,
                statement_type=stmt_type,
                error_message=f"Blocked statement type: {stmt_type}",
                blocked_reason=reason,
            )

        # ── Guard: must start with SELECT or WITH ─────────────────────────────
        if stmt_type not in ("SELECT", "WITH"):
            reason = (
                f"Statement type '{stmt_type}' is not allowed. "
                "Only SELECT queries (including CTEs with WITH) are permitted."
            )
            return ValidatorResult(
                is_valid=False,
                statement_type=stmt_type,
                error_message=f"Non-SELECT statement: {stmt_type}",
                blocked_reason=reason,
            )

        # ── Guard: dangerous keywords hidden inside comments ──────────────────
        # Check single-line comments (--) for blocked keywords BEFORE stripping
        # e.g. "SELECT 1; --DROP TABLE users" should be blocked
        single_line_comments = re.findall(r'--[^\n]*', sql)
        for comment in single_line_comments:
            comment_tokens = re.findall(r'\b([A-Z_]+)\b', comment.upper())
            for token in comment_tokens:
                if token in SQLValidator._BLOCKED_KEYWORDS:
                    return ValidatorResult(
                        is_valid=False,
                        statement_type=stmt_type,
                        error_message=f"Blocked keyword '{token}' found in comment",
                        blocked_reason=(
                            f"Blocked keyword '{token}' detected inside a comment. "
                            "Comment-hidden injection is not allowed."
                        ),
                    )

        # ── Guard: semicolon stacking (multi-statement injection) ─────────────
        # Remove content inside single-quoted strings before checking
        # (simplified: strip string literals and check for remaining semicolons
        #  that are followed by non-whitespace, indicating a second statement)
        stripped = re.sub(r"'[^']*'", "''", sql)  # collapse string literals
        stripped = SQLValidator._SINGLE_LINE_COMMENT_RE.sub(' ', stripped)
        # Remove block comments
        stripped = re.sub(r'/\*.*?\*/', ' ', stripped, flags=re.DOTALL)

        semicolons = list(SQLValidator._SEMICOLON_RE.finditer(stripped))
        for m in semicolons:
            after = stripped[m.end():].strip()
            if after:   # there is real content after the semicolon
                return ValidatorResult(
                    is_valid=False,
                    statement_type=stmt_type,
                    error_message="Multiple statements detected",
                    blocked_reason=(
                        "Semicolon-separated statement injection is not allowed. "
                        "Submit one query at a time."
                    ),
                )

        # ── Guard: scan for blocked keywords anywhere in the statement ─────────
        # Tokenise the normalised SQL and look for any blocked keyword
        # (guards against e.g. UNION SELECT … ; DROP)
        tokens = re.findall(r'\b([A-Z_]+)\b', stripped.upper())
        for token in tokens:
            if token in SQLValidator._BLOCKED_KEYWORDS:
                reason = (
                    f"Blocked keyword '{token}' detected in query. "
                    "Only read-only operations are permitted."
                )
                return ValidatorResult(
                    is_valid=False,
                    statement_type=stmt_type,
                    error_message=f"Blocked keyword: {token}",
                    blocked_reason=reason,
                )

        # ── All checks passed ─────────────────────────────────────────────────
        logger.debug("Query validated OK: %s", sql[:80])
        return ValidatorResult(
            is_valid=True,
            statement_type=stmt_type,
        )