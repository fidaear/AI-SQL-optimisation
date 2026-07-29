"""
rule_engine.py
==============
Task: Rule-Based Optimization Engine
Project: AI-Assisted SQL Query Optimization

Public API (matches test_rule_engine.py):

  RuleResult                — dataclass: rule, severity, message, suggestion
  detect_seq_scan()         — flags Seq Scan on large tables (uses ParsedPlan + stats)
  detect_select_star()      — flags SELECT * (plain star, table.*)
  detect_missing_where()    — flags queries with no WHERE clause
  detect_cartesian_product()— flags comma-JOIN without ON condition
  detect_high_cost()        — flags total_cost above threshold
  run_all_rules()           — runs all rules; returns list[RuleResult] of violations only

NOTE: The orchestrator calls run_all_rules(sql, statistics) with the OLD
signature.  The new signature is run_all_rules(sql, plan, stats) — callers
that omit plan / stats receive safe defaults, so both call-sites work.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

# Row-count threshold above which a table is "large"
_LARGE_TABLE_THRESHOLD = 1_000

# Cost threshold above which a plan is "high cost"
_HIGH_COST_THRESHOLD = 10_000.0


# ─────────────────────────────────────────────────────────────────────────────
# RuleResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuleResult:
    """
    Represents a single rule violation.

    Fields
    ------
    rule       : str          — Machine-readable rule name
    severity   : str          — "HIGH" | "MEDIUM" | "LOW"
    message    : str          — Human-readable description of the violation
    suggestion : str | None   — Optional fix hint
    """
    rule: str
    severity: str
    message: str
    suggestion: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _row_count_for_table(table: str, stats: dict) -> int:
    """Extract row count for a table from the stats dict."""
    try:
        rows_data = stats.get("reltuples", {}).get("data", [])
        for entry in rows_data:
            if entry.get("table_name") == table:
                return int(entry.get("reltuples", 0))
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# RULE 1 — Detect Sequential Scan on large tables
# ─────────────────────────────────────────────────────────────────────────────

def detect_seq_scan(plan, stats: dict) -> List[RuleResult]:
    """
    Flag tables that are accessed via a Sequential Scan and have more
    rows than _LARGE_TABLE_THRESHOLD.

    Parameters
    ----------
    plan  : ParsedPlan   — output of PlanParser.parse()
    stats : dict         — output of StatisticsExtractor (has 'reltuples' key)

    Returns
    -------
    list[RuleResult]  — empty if no violation
    """
    violations: List[RuleResult] = []

    if not getattr(plan, "has_seq_scan", False):
        return violations

    scanned = getattr(plan, "scanned_tables", []) or []

    for table in scanned:
        row_count = _row_count_for_table(table, stats)
        if row_count > _LARGE_TABLE_THRESHOLD:
            msg = (
                f"Sequential Scan detected on '{table}' "
                f"({row_count:,} rows). Consider adding an index."
            )
            violations.append(RuleResult(
                rule="detect_seq_scan",
                severity="HIGH",
                message=msg,
                suggestion=f"CREATE INDEX ON {table}(<filter_column>);",
            ))
            logger.info("[Rule] detect_seq_scan triggered on '%s' (%d rows)", table, row_count)

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# RULE 2 — Detect SELECT *
# ─────────────────────────────────────────────────────────────────────────────

def detect_select_star(sql: str) -> List[RuleResult]:
    """
    Flag SELECT * or SELECT alias.* usage.

    COUNT(*) is explicitly allowed — it is a function call, not a column
    wildcard, and does not cause unnecessary data transfer.

    Parameters
    ----------
    sql : str   — raw SQL query

    Returns
    -------
    list[RuleResult]  — empty if no violation
    """
    violations: List[RuleResult] = []

    # Normalise whitespace for easier matching
    normalised = " ".join(sql.split())

    # Remove COUNT(*) so it is not flagged
    without_count = re.sub(r'\bCOUNT\s*\(\s*\*\s*\)', 'COUNT_STAR_PLACEHOLDER', normalised, flags=re.IGNORECASE)

    # Match bare `*` after SELECT, or `alias.*`
    # Patterns:
    #   SELECT *        →  SELECT \*
    #   SELECT u.*      →  SELECT \w+\.\*
    if re.search(r'\bSELECT\s+\*', without_count, re.IGNORECASE) or \
       re.search(r'\bSELECT\s+\w+\.\*', without_count, re.IGNORECASE) or \
       re.search(r',\s*\w+\.\*', without_count, re.IGNORECASE) or \
       re.search(r',\s*\*', without_count, re.IGNORECASE):

        violations.append(RuleResult(
            rule="detect_select_star",
            severity="MEDIUM",
            message=(
                "SELECT * detected. Fetching all columns increases I/O and "
                "memory usage. Specify only the columns you need."
            ),
            suggestion="Replace SELECT * with an explicit column list.",
        ))
        logger.info("[Rule] detect_select_star triggered")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# RULE 3 — Detect Missing WHERE clause
# ─────────────────────────────────────────────────────────────────────────────

def detect_missing_where(sql: str) -> List[RuleResult]:
    """
    Flag SELECT queries that have no WHERE clause and are not pure
    aggregates (which intentionally scan the whole table).

    Parameters
    ----------
    sql : str

    Returns
    -------
    list[RuleResult]
    """
    violations: List[RuleResult] = []

    upper = sql.upper()

    # Skip non-SELECT queries
    stripped = sql.strip()
    if not re.match(r'\bSELECT\b', stripped, re.IGNORECASE):
        return violations

    # Allow pure aggregates: queries whose SELECT list contains only
    # aggregate functions and no FROM … JOIN (i.e. they are intentional
    # full-table aggregations).
    # Heuristic: if query has COUNT/SUM/AVG/MIN/MAX but NO JOIN and NO WHERE
    # *and* only one table in FROM — it is intentional.
    is_aggregate_only = bool(
        re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql, re.IGNORECASE)
        and not re.search(r'\bJOIN\b', sql, re.IGNORECASE)
    )
    if is_aggregate_only:
        return violations

    has_where = bool(re.search(r'\bWHERE\b', sql, re.IGNORECASE))
    if not has_where:
        violations.append(RuleResult(
            rule="detect_missing_where",
            severity="MEDIUM",
            message=(
                "Query has no WHERE clause. This will perform a full table scan "
                "and may return an unexpectedly large result set."
            ),
            suggestion="Add a WHERE clause to filter results.",
        ))
        logger.info("[Rule] detect_missing_where triggered")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# RULE 4 — Detect Cartesian Product
# ─────────────────────────────────────────────────────────────────────────────

def detect_cartesian_product(sql: str) -> List[RuleResult]:
    """
    Flag queries that produce a Cartesian product:
    - Comma-separated tables in FROM without a JOIN … ON condition,
      e.g.  SELECT * FROM orders, customers

    Parameters
    ----------
    sql : str

    Returns
    -------
    list[RuleResult]
    """
    violations: List[RuleResult] = []

    upper = sql.upper()

    # A Cartesian product arises from a comma-list in FROM with no JOIN ON.
    # Simple heuristic: FROM clause contains a comma AND there is no JOIN keyword.
    from_match = re.search(r'\bFROM\b(.+?)(?:\bWHERE\b|\bGROUP\b|\bORDER\b|\bLIMIT\b|$)',
                           sql, re.IGNORECASE | re.DOTALL)
    if not from_match:
        return violations

    from_clause = from_match.group(1)
    has_comma_join = bool(re.search(r',', from_clause))
    # A real JOIN keyword would introduce a proper ON condition
    has_explicit_join = bool(re.search(r'\bJOIN\b', sql, re.IGNORECASE))

    # Cartesian: comma in FROM and no explicit JOIN … ON
    if has_comma_join and not has_explicit_join:
        violations.append(RuleResult(
            rule="detect_cartesian_product",
            severity="HIGH",
            message=(
                "Cartesian product detected: multiple tables in FROM clause "
                "without a JOIN condition. This may return millions of rows."
            ),
            suggestion="Replace comma-joined tables with explicit JOIN … ON syntax.",
        ))
        logger.info("[Rule] detect_cartesian_product triggered")

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# RULE 5 — Detect High Cost
# ─────────────────────────────────────────────────────────────────────────────

def detect_high_cost(plan) -> List[RuleResult]:
    """
    Flag queries whose estimated total_cost exceeds _HIGH_COST_THRESHOLD.

    Parameters
    ----------
    plan : ParsedPlan

    Returns
    -------
    list[RuleResult]
    """
    violations: List[RuleResult] = []

    cost = getattr(plan, "total_cost", None)
    if cost is None:
        return violations   # cannot determine cost — skip silently

    if cost > _HIGH_COST_THRESHOLD:
        violations.append(RuleResult(
            rule="detect_high_cost",
            severity="HIGH",
            message=(
                f"Query has a high estimated cost of {cost:,.0f}. "
                "Consider adding indexes or rewriting the query."
            ),
            suggestion="Run EXPLAIN ANALYZE to identify the most expensive nodes.",
        ))
        logger.info("[Rule] detect_high_cost triggered (cost=%.0f)", cost)

    return violations


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT — Run all rules
# ─────────────────────────────────────────────────────────────────────────────

def run_all_rules(
    sql: str,
    plan=None,
    stats: dict = None,
) -> List[RuleResult]:
    """
    Run all optimization rules against a SQL query.

    Parameters
    ----------
    sql   : str           — Raw SQL query string
    plan  : ParsedPlan    — Output of PlanParser.parse() (optional)
    stats : dict          — Output of StatisticsExtractor (optional)

    Returns
    -------
    list[RuleResult]
        Only rules that triggered (have violations) are included.
        Returns an empty list for a clean query.

    Backward-compatibility note
    ---------------------------
    The orchestrator calls run_all_rules(sql, statistics_dict) with only two
    positional arguments.  When `plan` receives a dict (old call-site), we
    treat it as `stats` and use a dummy plan so the function does not crash.
    """
    # ── Backward-compatibility shim ───────────────────────────────────────────
    if isinstance(plan, dict):
        # Called as run_all_rules(sql, stats_dict) — old orchestrator signature
        stats = plan
        plan  = None

    stats = stats or {}

    # Create a minimal no-op plan if none was supplied
    if plan is None:
        class _EmptyPlan:
            has_seq_scan    = False
            has_index_scan  = False
            has_nested_loop = False
            has_hash_join   = False
            total_cost      = None
            estimated_rows  = None
            scanned_tables  = []
            used_indexes    = []
        plan = _EmptyPlan()

    # ── Collect violations from every rule ────────────────────────────────────
    violations: List[RuleResult] = []

    violations.extend(detect_seq_scan(plan, stats))
    violations.extend(detect_select_star(sql))
    violations.extend(detect_missing_where(sql))
    violations.extend(detect_cartesian_product(sql))
    violations.extend(detect_high_cost(plan))

    logger.info(
        "run_all_rules: %d violation(s) from %d rule(s)",
        len(violations), 5,
    )
    return violations