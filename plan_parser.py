"""
EXPLAIN Plan Parser
Parses and analyzes PostgreSQL EXPLAIN / EXPLAIN ANALYZE output.

Public API (matches test_plan_parser.py):
  ParsedPlan   — dataclass holding all extracted fields
  PlanParser   — static method: parse(plan) → ParsedPlan
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ParsedPlan:
    """
    Structured representation of a PostgreSQL EXPLAIN output.

    Fields
    ------
    has_seq_scan      : bool            — Any "Seq Scan" node present
    has_index_scan    : bool            — Any "Index Scan" node present
    has_nested_loop   : bool            — Any "Nested Loop" node present
    has_hash_join     : bool            — Any "Hash Join" node present
    total_cost        : float | None    — Outermost cost=X..Y → Y value
    estimated_rows    : int   | None    — rows=N from outermost node
    actual_rows       : int   | None    — actual rows=N (EXPLAIN ANALYZE)
    execution_time_ms : float | None    — "Execution Time: X ms"
    planning_time_ms  : float | None    — "Planning Time: X ms"
    scanned_tables    : list[str]       — All table names after "on <table>"
    used_indexes      : list[str]       — All index names after "using <index>"
    """
    has_seq_scan: bool = False
    has_index_scan: bool = False
    has_nested_loop: bool = False
    has_hash_join: bool = False
    total_cost: Optional[float] = None
    estimated_rows: Optional[int] = None
    actual_rows: Optional[int] = None
    execution_time_ms: Optional[float] = None
    planning_time_ms: Optional[float] = None
    scanned_tables: List[str] = field(default_factory=list)
    used_indexes: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Parser
# ─────────────────────────────────────────────────────────────────────────────

class PlanParser:
    """
    Parses raw PostgreSQL EXPLAIN / EXPLAIN ANALYZE output into a ParsedPlan.

    Accepts:
      - A list of strings (one line per element, as returned by psycopg2
        when EXPLAIN output rows are fetched as single-column tuples or strings)
      - A single multi-line string
      - None / empty input → safe defaults
    """

    # -- Compiled regex patterns ------------------------------------------------

    # cost=0.00..4821.00 rows=210000 width=97
    _COST_RE = re.compile(r'cost=[\d.]+\.\.([\d.]+)\s+rows=(\d+)', re.IGNORECASE)

    # (actual time=0.012..0.025 rows=10 loops=1)
    _ACTUAL_RE = re.compile(r'actual time=[\d.]+\.\.[\d.]+\s+rows=(\d+)', re.IGNORECASE)

    # Execution Time: 0.5 ms
    _EXEC_TIME_RE = re.compile(r'Execution Time:\s*([\d.]+)\s*ms', re.IGNORECASE)

    # Planning Time: 0.2 ms
    _PLAN_TIME_RE = re.compile(r'Planning Time:\s*([\d.]+)\s*ms', re.IGNORECASE)

    # Seq Scan on <table>
    _SEQ_SCAN_RE = re.compile(r'Seq Scan on (\w+)', re.IGNORECASE)

    # Index Scan using <index> on <table>   OR   Index Scan on <table>
    _IDX_SCAN_RE = re.compile(
        r'Index(?:\s+Only)?\s+Scan(?:\s+using\s+(\w+))?\s+on\s+(\w+)',
        re.IGNORECASE,
    )

    # Nested Loop
    _NESTED_LOOP_RE = re.compile(r'Nested Loop', re.IGNORECASE)

    # Hash Join
    _HASH_JOIN_RE = re.compile(r'Hash Join', re.IGNORECASE)

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def parse(plan: Union[list, str, None]) -> ParsedPlan:
        """
        Parse EXPLAIN output and return a ParsedPlan.

        Parameters
        ----------
        plan : list[str] | str | None
            Raw EXPLAIN output.  Each element may be a plain string or a
            single-element tuple as psycopg2 returns (the parser handles both).

        Returns
        -------
        ParsedPlan
            Always returns a valid object — never raises.
        """
        result = ParsedPlan()

        if not plan:
            return result

        # ── Normalise input to a flat list of strings ─────────────────────────
        if isinstance(plan, str):
            lines = plan.splitlines()
        else:
            # Each element may be a tuple like ("Seq Scan on t ...",) or a str
            lines = []
            for item in plan:
                if isinstance(item, tuple):
                    lines.append(str(item[0]))
                else:
                    lines.append(str(item))

        # ── Join everything for global regex passes ───────────────────────────
        full_text = "\n".join(lines)

        # ── Scan-type detection ───────────────────────────────────────────────
        for line in lines:
            if PlanParser._SEQ_SCAN_RE.search(line):
                result.has_seq_scan = True
                m = PlanParser._SEQ_SCAN_RE.search(line)
                table = m.group(1)
                if table not in result.scanned_tables:
                    result.scanned_tables.append(table)

            idx_m = PlanParser._IDX_SCAN_RE.search(line)
            if idx_m:
                result.has_index_scan = True
                index_name = idx_m.group(1)   # may be None (no "using")
                table_name = idx_m.group(2)
                if index_name and index_name not in result.used_indexes:
                    result.used_indexes.append(index_name)
                if table_name not in result.scanned_tables:
                    result.scanned_tables.append(table_name)

            if PlanParser._NESTED_LOOP_RE.search(line):
                result.has_nested_loop = True

            if PlanParser._HASH_JOIN_RE.search(line):
                result.has_hash_join = True

        # ── Cost / rows from outermost (first) node ───────────────────────────
        for line in lines:
            cost_m = PlanParser._COST_RE.search(line)
            if cost_m and result.total_cost is None:
                result.total_cost = float(cost_m.group(1))
                result.estimated_rows = int(cost_m.group(2))
                break   # only the outermost node

        # ── EXPLAIN ANALYZE extras ────────────────────────────────────────────
        # actual rows — from the first "actual time=..." occurrence
        actual_m = PlanParser._ACTUAL_RE.search(full_text)
        if actual_m:
            result.actual_rows = int(actual_m.group(1))

        exec_m = PlanParser._EXEC_TIME_RE.search(full_text)
        if exec_m:
            result.execution_time_ms = float(exec_m.group(1))

        plan_m = PlanParser._PLAN_TIME_RE.search(full_text)
        if plan_m:
            result.planning_time_ms = float(plan_m.group(1))

        logger.debug(
            "ParsedPlan: seq=%s idx=%s cost=%s rows=%s tables=%s indexes=%s",
            result.has_seq_scan, result.has_index_scan,
            result.total_cost, result.estimated_rows,
            result.scanned_tables, result.used_indexes,
        )
        return result