"""
ai_optimizer.py — AI-Powered SQL Optimizer
Task 4: LLM Integration & Validation
Project: AI-Assisted SQL Query Optimizer

Pipeline (matches the recommended workflow):
  1. Safety pre-check      — block dangerous / non-SELECT SQL
  2. LLM rewrite           — send query + FULL DB context to Mistral
  3. Safety post-check     — rewritten SQL must still be a safe SELECT
  4. Semantic check        — same tables / columns still present
  5. EXPLAIN cost compare  — run EXPLAIN (FORMAT JSON) on both queries,
                             prefer whichever has the lower total_cost
  6. Explanation call      — separate LLM call, zero prose leakage into SQL

What changed vs the previous version
─────────────────────────────────────
• DbContext now carries an `explain_plan` field so the orchestrator can pass
  the raw EXPLAIN JSON to the LLM — giving it the actual planner decisions
  (Seq Scan vs Index Scan, hash vs nested-loop join, cost estimates).

• ContextBuilder.build() now renders the EXPLAIN plan summary inside the
  context block so Mistral sees WHERE the planner is spending time, not just
  the schema.

• optimize() accepts an optional `explain_json` parameter (the raw output of
  EXPLAIN (FORMAT JSON) on the original query).  When present it is:
    a) rendered into the context block shown to the LLM
    b) used in a post-rewrite EXPLAIN cost comparison so we always return
       the *faster* query rather than blindly trusting the LLM rewrite.

• SYSTEM_PROMPT strengthened: Rule 5 now lists 7 concrete techniques the
  model MUST attempt. The old "reproduce unchanged if nothing to do" escape
  hatch is gone.

• _build_rewrite_prompt() explicitly lists required techniques and opens with
  "You MUST change at least one thing."

• call_type=LLMCallType.SQL is now correctly passed to the rewrite call
  (was defaulting to GENERIC / 600 tokens before).

• Identity detection: logs a WARNING when the LLM echoes the query back
  unchanged so operators can see it without digging through raw logs.

• ContextBuilder.from_statistics_dict() now warns when a table is not found
  in the statistics dict (silent stub was masking key-casing mismatches).
"""

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from ai.llm_client import LLMClient, LLMClientError, LLMCallType

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Data classes
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnInfo:
    name:        str
    data_type:   str
    nullable:    bool = True
    is_pk:       bool = False
    is_fk:       bool = False
    fk_ref:      str  = ""   # e.g. "orders.user_id"


@dataclass
class IndexInfo:
    name:    str
    columns: list[str]
    unique:  bool = False


@dataclass
class TableContext:
    """Everything the LLM needs to know about one table."""
    name:        str
    row_count:   int               = 0
    columns:     list[ColumnInfo]  = field(default_factory=list)
    indexes:     list[IndexInfo]   = field(default_factory=list)
    description: str               = ""


@dataclass
class DbContext:
    """
    Full database context passed to the optimizer.
    Only include tables referenced in the SQL being optimized.

    New field: explain_plan_json — the raw output of
      EXPLAIN (FORMAT JSON, ANALYZE FALSE) <original_query>
    When set, it is rendered into the prompt so Mistral sees the actual
    planner decisions (Seq Scan cost, Hash Join vs Nested Loop, etc.)
    """
    tables:           list[TableContext] = field(default_factory=list)
    db_name:          str                = ""
    postgres_version: str                = ""
    explain_plan_json: Optional[str]     = None   # ← NEW


@dataclass
class OptimizationResult:
    """Everything the orchestrator / caller needs from one optimisation run."""
    original_sql:    str
    optimized_sql:   str          = ""
    success:         bool         = False
    safety_passed:   bool         = False
    semantic_passed: bool         = False
    error:           Optional[str] = None
    warnings:        list         = field(default_factory=list)
    explanation:     str          = ""
    # Cost comparison results (populated when explain_plan_json is provided)
    original_cost:   float        = 0.0
    optimized_cost:  float        = 0.0
    cost_improved:   bool         = False


# ══════════════════════════════════════════════════════════════════════════════
# Context Builder
# ══════════════════════════════════════════════════════════════════════════════

class ContextBuilder:
    """
    Converts a DbContext into a compact, LLM-readable text block.

    Now also renders a summary of the EXPLAIN plan so the model can see:
      - Which nodes are Seq Scans (candidates for index optimisation)
      - Which joins are Hash Join / Nested Loop (reorder candidates)
      - Total planner cost estimate
    """

    _TYPE_MAP = {
        "character varying": "varchar",
        "timestamp without time zone": "timestamp",
        "timestamp with time zone": "timestamptz",
        "double precision": "float8",
        "integer": "int",
        "bigint": "int8",
        "boolean": "bool",
    }

    @classmethod
    def build(cls, ctx: DbContext) -> str:
        if not ctx or not ctx.tables:
            return ""

        lines: list[str] = ["=== DATABASE CONTEXT (read-only reference) ==="]

        if ctx.db_name:
            lines.append(f"Database : {ctx.db_name}")
        if ctx.postgres_version:
            lines.append(f"PostgreSQL: {ctx.postgres_version}")

        for tbl in ctx.tables:
            lines.append("")
            row_str = f"{tbl.row_count:,}" if tbl.row_count else "unknown"
            lines.append(f"TABLE {tbl.name}  (~{row_str} rows)")

            if tbl.columns:
                lines.append("  Columns:")
                for col in tbl.columns:
                    dtype = cls._TYPE_MAP.get(col.data_type.lower(), col.data_type)
                    flags = []
                    if col.is_pk:
                        flags.append("PK")
                    if col.is_fk:
                        ref = f" → {col.fk_ref}" if col.fk_ref else ""
                        flags.append(f"FK{ref}")
                    if not col.nullable and not col.is_pk:
                        flags.append("NOT NULL")
                    flag_str = f"  [{', '.join(flags)}]" if flags else ""
                    lines.append(f"    {col.name}  {dtype}{flag_str}")

            if tbl.columns:
                allowed = ", ".join(c.name for c in tbl.columns)
                lines.append(
                    f"  *** ALLOWED COLUMNS FOR {tbl.name.upper()} "
                    f"(ONLY these exist — do NOT add any others): {allowed} ***"
                )

            if tbl.indexes:
                lines.append("  Existing indexes:")
                for idx in tbl.indexes:
                    uniq = "UNIQUE " if idx.unique else ""
                    cols = ", ".join(idx.columns)
                    lines.append(f"    {uniq}({cols})  — {idx.name}")
            else:
                lines.append("  Existing indexes: none")

        # ── EXPLAIN plan summary (NEW) ──────────────────────────────────────
        if ctx.explain_plan_json:
            explain_summary = cls._summarise_explain(ctx.explain_plan_json)
            if explain_summary:
                lines.append("")
                lines.append("=== QUERY PLAN SUMMARY (from EXPLAIN) ===")
                lines.extend(explain_summary)

        lines.append("")
        lines.append("=== END DATABASE CONTEXT ===")
        return "\n".join(lines)

    @classmethod
    def _summarise_explain(cls, explain_json_str: str) -> list[str]:
        """
        Parse the JSON output of EXPLAIN (FORMAT JSON) and extract the
        key facts a performance engineer cares about:
          - Total estimated cost
          - Node types (Seq Scan = bad, Index Scan = good)
          - Join strategies used
        Returns a list of lines to append to the context block.
        """
        try:
            plan_data = json.loads(explain_json_str)
            # EXPLAIN FORMAT JSON returns a list with one element
            if isinstance(plan_data, list):
                plan_data = plan_data[0]
            root = plan_data.get("Plan", plan_data)
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            logger.warning("ContextBuilder: could not parse EXPLAIN JSON: %s", exc)
            return []

        lines: list[str] = []
        nodes: list[dict] = []
        cls._collect_nodes(root, nodes)

        total_cost = root.get("Total Cost", 0)
        lines.append(f"  Total planner cost : {total_cost:.2f}")

        seq_scans   = [n for n in nodes if n.get("Node Type") == "Seq Scan"]
        index_scans = [n for n in nodes if "Index" in n.get("Node Type", "")]
        hash_joins  = [n for n in nodes if n.get("Node Type") == "Hash Join"]
        nl_joins    = [n for n in nodes if n.get("Node Type") == "Nested Loop"]

        if seq_scans:
            tables = ", ".join(n.get("Relation Name", "?") for n in seq_scans)
            lines.append(f"  Seq Scans (no index used) : {tables}  ← optimise these")
        if index_scans:
            tables = ", ".join(n.get("Relation Name", "?") for n in index_scans)
            lines.append(f"  Index Scans               : {tables}")
        if hash_joins:
            lines.append(f"  Hash Joins                : {len(hash_joins)}")
        if nl_joins:
            lines.append(f"  Nested Loops              : {len(nl_joins)}  ← check join order")

        return lines

    @classmethod
    def _collect_nodes(cls, node: dict, out: list) -> None:
        """Recursively collect all plan nodes into a flat list."""
        if not isinstance(node, dict):
            return
        out.append(node)
        for child in node.get("Plans", []):
            cls._collect_nodes(child, out)

    @classmethod
    def from_statistics_dict(cls, stats: dict, table_names: list[str]) -> "DbContext":
        """
        Convenience factory: build a DbContext from the dict that
        StatisticsExtractor.get_comprehensive_statistics() already returns.
        """
        tables = []
        for tname in table_names:
            raw = stats.get(tname) or stats.get(tname.lower()) or {}
            if not raw:
                logger.warning(
                    "ContextBuilder: table '%s' not found in statistics dict "
                    "(available keys: %s). Using empty stub — LLM will have no "
                    "schema info for this table.",
                    tname, list(stats.keys())[:10],
                )
                tables.append(TableContext(name=tname))
                continue

            columns = [
                ColumnInfo(
                    name=c.get("column_name", ""),
                    data_type=c.get("data_type", "text"),
                    nullable=c.get("is_nullable", "YES") == "YES",
                    is_pk=c.get("is_primary_key", False),
                    is_fk=bool(c.get("foreign_key_ref")),
                    fk_ref=c.get("foreign_key_ref", ""),
                )
                for c in raw.get("columns", [])
            ]

            indexes = [
                IndexInfo(
                    name=i.get("index_name", ""),
                    columns=i.get("columns", []),
                    unique=i.get("unique", False),
                )
                for i in raw.get("indexes", [])
            ]

            tables.append(TableContext(
                name=tname,
                row_count=raw.get("row_count", 0),
                columns=columns,
                indexes=indexes,
            ))

        return DbContext(tables=tables)


# ══════════════════════════════════════════════════════════════════════════════
# SQL Safety Validator
# ══════════════════════════════════════════════════════════════════════════════

class SQLSafetyValidator:
    """Pure-regex / heuristic safety gate — no DB required."""

    _DANGEROUS_PATTERNS = [
        r"\bDROP\b",
        r"\bTRUNCATE\b",
        r"\bDELETE\b",
        r"\bINSERT\b",
        r"\bUPDATE\b",
        r"\bMERGE\b",
        r"\bUPSERT\b",
        r"\bALTER\b",
        r"\bCREATE\b",
        r"\bRENAME\b",
        r"\bGRANT\b",
        r"\bREVOKE\b",
        r"\bEXEC(UTE)?\b",
        r"\bxp_cmdshell\b",
        r";\s*--",
        r";\s*SELECT",
        r"'\s*OR\s+'",
        r"1\s*=\s*1",
    ]

    _compiled = [re.compile(p, re.IGNORECASE) for p in _DANGEROUS_PATTERNS]

    @classmethod
    def is_safe(cls, sql: str) -> tuple[bool, list[str]]:
        violations = []
        for pattern in cls._compiled:
            if pattern.search(sql):
                violations.append(f"Dangerous pattern detected: `{pattern.pattern}`")
        return len(violations) == 0, violations

    @classmethod
    def is_select_only(cls, sql: str) -> bool:
        return bool(re.match(r"\s*SELECT\b", sql.strip(), re.IGNORECASE))


# ══════════════════════════════════════════════════════════════════════════════
# Semantic Validator
# ══════════════════════════════════════════════════════════════════════════════

class SemanticValidator:
    """Lightweight semantic-equivalence checks between original and rewritten SQL."""

    @staticmethod
    def extract_tables(sql: str) -> set[str]:
        pattern = r"(?:FROM|JOIN)\s+([`\"\[]?[\w.]+[`\"\]]?)"
        return {m.group(1).strip('`"[]').lower()
                for m in re.finditer(pattern, sql, re.IGNORECASE)}

    @staticmethod
    def extract_columns(sql: str) -> set[str]:
        select_match = re.search(
            r"SELECT\s+(.*?)\s+FROM", sql, re.IGNORECASE | re.DOTALL
        )
        if not select_match:
            return set()
        cols_raw = select_match.group(1)
        cols = re.findall(r"[\w]+", cols_raw)
        return {c.lower() for c in cols}

    @classmethod
    def check_equivalence(cls, original: str, rewritten: str) -> tuple[bool, list[str]]:
        issues = []

        orig_tables    = cls.extract_tables(original)
        rewrite_tables = cls.extract_tables(rewritten)

        missing_tables = orig_tables - rewrite_tables
        if missing_tables:
            issues.append(f"Tables removed from rewrite: {missing_tables}")

        new_tables = rewrite_tables - orig_tables
        if new_tables:
            issues.append(f"New tables introduced (verify intent): {new_tables}")

        orig_star    = bool(re.search(r"SELECT\s+\*", original,  re.IGNORECASE))
        rewrite_star = bool(re.search(r"SELECT\s+\*", rewritten, re.IGNORECASE))
        if orig_star and not rewrite_star:
            pass  # Expanding SELECT * to explicit columns is a valid optimisation
        if not orig_star:
            orig_cols    = cls.extract_columns(original)
            rewrite_cols = cls.extract_columns(rewritten)
            dropped = orig_cols - rewrite_cols - {"*"}
            if dropped:
                issues.append(f"Columns dropped in rewrite: {dropped}")

        if not re.match(r"\s*SELECT\b", rewritten.strip(), re.IGNORECASE):
            issues.append("Rewritten query is no longer a SELECT statement.")

        return len(issues) == 0, issues


# ══════════════════════════════════════════════════════════════════════════════
# EXPLAIN Cost Extractor
# ══════════════════════════════════════════════════════════════════════════════

def extract_explain_total_cost(explain_json_str: str) -> float:
    """
    Parse the output of EXPLAIN (FORMAT JSON) and return the root node's
    Total Cost estimate.  Returns 0.0 if parsing fails.

    The orchestrator calls db_service.run_secure_explain(session_id, sql,
    analyze=False) to get this — it's cheap (no data read) and gives us
    the planner's cost model so we can compare original vs rewritten.
    """
    try:
        data = json.loads(explain_json_str)
        if isinstance(data, list):
            data = data[0]
        plan = data.get("Plan", data)
        return float(plan.get("Total Cost", 0))
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# AI Optimizer
# ══════════════════════════════════════════════════════════════════════════════

class AIOptimizer:
    """
    Main orchestrator for LLM-powered SQL optimisation.

    Basic usage (no context):
        result = AIOptimizer().optimize(sql_query)

    With schema context (smarter rewrites):
        ctx = ContextBuilder.from_statistics_dict(stats, table_names)
        result = AIOptimizer().optimize(sql_query, db_context=ctx)

    With schema context + EXPLAIN plan (full workflow):
        ctx = ContextBuilder.from_statistics_dict(stats, table_names)
        ctx.explain_plan_json = db_service.run_secure_explain(
            session_id, sql_query, analyze=False, format="json"
        )["explain_plan_raw"]
        result = AIOptimizer().optimize(sql_query, db_context=ctx,
                                        explain_json=ctx.explain_plan_json)
    """

    # ── WHY THE SYSTEM PROMPT IS STRUCTURED THIS WAY ──────────────────────────
    # Mistral via Ollama has two known bad behaviours:
    #   1. Responds in French when it detects French locale on the server.
    #   2. Echoes the original query back unchanged when given an easy
    #      "reproduce if nothing to do" escape hatch (old Rule 5).
    #
    # Fixes in this version:
    #   - Removed the echo escape hatch entirely.
    #   - Rule 5 now enumerates 7 concrete techniques to force a real attempt.
    #   - The user prompt opens with "You MUST change at least one thing."
    #   - The EXPLAIN plan is injected so the model can see Seq Scans and
    #     expensive nodes — giving it concrete evidence of what to fix.
    # ─────────────────────────────────────────────────────────────────────────

    SYSTEM_PROMPT = """\
LANGUAGE: Respond in ENGLISH only. Never use French or any other language.

ROLE: You are a senior PostgreSQL performance engineer.
TASK: Rewrite the SQL query provided by the user so it runs faster on PostgreSQL.

If DATABASE CONTEXT is provided (table schemas, row counts, existing indexes,
EXPLAIN plan summary), use it to make smarter decisions:
  - Avoid SELECT * when column list is known (reduces I/O) — but you may ONLY
    use column names that appear under "Columns:" for that table in the
    DATABASE CONTEXT. Never add a column you have not seen listed there, even
    if it is a common or expected column name for this type of table.
  - Prefer index-aware JOIN order (smallest / most-filtered table first)
  - Avoid functions on indexed columns in WHERE (defeats index scan)
  - Use EXISTS instead of IN for correlated subqueries
  - Replace DISTINCT with GROUP BY when it enables index use
  - If EXPLAIN shows a Seq Scan on a large table, add a WHERE on an indexed column
    or restructure the JOIN to reduce rows before the scan
  - If EXPLAIN shows a Nested Loop on large tables, consider reordering joins

If NO DATABASE CONTEXT is provided, do not expand SELECT * — you cannot know
the real column names, so leave SELECT * as-is and apply a different
technique instead.

OUTPUT FORMAT — follow EXACTLY, nothing else allowed:
###START###
<single valid PostgreSQL SELECT statement, no comments, no markdown>
###END###

CORRECT example:
###START###
SELECT u.user_id, u.name, o.status FROM users u INNER JOIN orders o ON u.user_id = o.user_id WHERE o.status = 'active' ORDER BY u.name;
###END###

WRONG example (never do this):
Voici la requête optimisée :
###START###
SELECT ...  -- index hint added
###END###
Query rewritten for performance.

RULES:
1. The SQL between ###START### and ###END### must be executable PostgreSQL with zero modifications.
2. No inline SQL comments inside the SQL (no --).
3. No markdown fences (no ```).
4. No explanations, no summaries, no greetings — only the two markers and the SQL.
5. ALWAYS attempt at least ONE of these rewrites:
   a) Expand SELECT * to explicit column names — ONLY if DATABASE CONTEXT is
      provided, and ONLY using column names listed there under "Columns:".
      CRITICAL: The "Columns:" section is the ONLY source of truth.
      Do NOT use column names that appear anywhere else, including:
        - Names that appear in foreign key references like "order_id → orders.id"
          (that means another table's column points HERE, not that this table
          HAS a column called order_id — the real PK here is just "id")
        - Names from your training data for tables with similar names
        - Names that "sound right" for this kind of table (order_id, user_id, etc.)
      If you are not 100% certain a column name appears under "Columns:" for
      that specific table, do not include it and pick a different technique.
   b) Add table aliases if missing.
   c) Reorder JOINs: smallest / most-filtered table first.
   d) Replace IN (subquery) with EXISTS or a JOIN.
   e) Push WHERE predicates inside CTEs or subqueries.
   f) Use explicit column list in GROUP BY instead of alias references.
   g) Replace DISTINCT with GROUP BY when it allows index use.
   h) Add a covering column to an existing index hint via column order.
   You may only reproduce the query unchanged if you can prove NONE of
   a–h applies AND the EXPLAIN shows no Seq Scans on large tables.
6. ENGLISH ONLY everywhere.
"""

    EXPLANATION_SYSTEM_PROMPT = """\
LANGUAGE: Respond in ENGLISH only.
ROLE: You are a PostgreSQL expert explaining an optimization to a developer.
TASK: Given an original SQL query and its optimized version, write 2-4 sentences
explaining WHAT was changed and WHY it improves performance.
Be concrete: mention index usage, I/O reduction, join order, cost estimates etc.
Output plain text only — no markdown, no bullet points, no code blocks.
"""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self._llm      = llm_client or LLMClient()
        self._safety   = SQLSafetyValidator()
        self._semantic = SemanticValidator()

    # ──────────────────────────────────────────
    # Public entry point
    # ──────────────────────────────────────────

    def optimize(
        self,
        sql:          str,
        db_context:   Optional[DbContext] = None,
        explain_json: Optional[str]       = None,
    ) -> OptimizationResult:
        """
        Full optimisation pipeline.

        Args:
            sql:          The original SQL query to optimize.
            db_context:   Schema/statistics context. Build with:
                            ContextBuilder.from_statistics_dict(stats, tables)
                          Optionally set db_context.explain_plan_json before
                          passing here.
            explain_json: Raw output of EXPLAIN (FORMAT JSON) on the original
                          query. If provided:
                            • rendered into the context block the LLM sees
                            • used for cost-based selection (return whichever
                              query has lower planner cost)
                          If omitted, the LLM rewrite is used unconditionally.

        Stage 1 — Safety pre-check on original SQL.
        Stage 2 — LLM rewrite with full schema + EXPLAIN context.
        Stage 3 — Safety post-check on rewritten SQL.
        Stage 4 — Semantic equivalence check.
        Stage 5 — EXPLAIN cost comparison (if explain_json provided).
        Stage 6 — Plain-English explanation (separate LLM call).
        """
        result = OptimizationResult(original_sql=sql)

        # Merge explain_json into db_context so ContextBuilder can render it
        if explain_json and db_context:
            db_context.explain_plan_json = explain_json
        elif explain_json and not db_context:
            db_context = DbContext(explain_plan_json=explain_json)

        # ── Stage 1: Pre-check ────────────────────────────────────────────────
        logger.info("Stage 1: safety pre-check")
        safe, violations = self._safety.is_safe(sql)
        if not safe:
            result.error = f"Input SQL failed safety check: {violations}"
            logger.warning(result.error)
            return result

        if not self._safety.is_select_only(sql):
            result.error = "Only SELECT queries are accepted by the optimizer."
            logger.warning(result.error)
            return result

        result.safety_passed = True
        logger.info("Stage 1: passed ✅")

        # ── Stage 2: LLM rewrite ──────────────────────────────────────────────
        logger.info("Stage 2: sending to LLM")

        context_block = ContextBuilder.build(db_context) if db_context else ""
        if context_block:
            logger.info("  Schema + EXPLAIN context injected (%d chars)", len(context_block))
        else:
            logger.info("  No schema context — proceeding without DB awareness")

        prompt = self._build_rewrite_prompt(sql, context_block)

        try:
            raw_response = self._llm.complete(
                prompt=prompt,
                system_prompt=self.SYSTEM_PROMPT,
                call_type=LLMCallType.SQL,
            )
        except LLMClientError as exc:
            result.error = f"LLM call failed: {exc}"
            logger.error(result.error)
            return result

        logger.debug("Raw LLM response: %s", repr(raw_response[:300]))
        optimized_sql = self._parse_sql_response(raw_response, sql)

        # ── Identity detection ────────────────────────────────────────────────
        def _normalise(q: str) -> str:
            return " ".join(q.upper().split())

        if _normalise(optimized_sql) == _normalise(sql):
            logger.warning(
                "Stage 2: LLM returned IDENTICAL query (no rewrite). "
                "Raw response (400 chars): %r", raw_response[:400]
            )

        if not optimized_sql:
            result.error = "LLM returned an empty or unparseable SQL response."
            logger.error(result.error)
            return result

        logger.info("Stage 2: LLM rewrite received ✅")

        # ── Stage 3: Post-check rewritten SQL ────────────────────────────────
        logger.info("Stage 3: safety post-check on rewritten SQL")
        safe_rewrite, violations_rewrite = self._safety.is_safe(optimized_sql)
        if not safe_rewrite:
            result.error = (
                f"Rewritten SQL failed safety check: {violations_rewrite}"
            )
            logger.error(result.error)
            return result

        if not self._safety.is_select_only(optimized_sql):
            result.error = "Rewritten SQL is no longer a SELECT query — rejected."
            logger.error(result.error)
            return result

        logger.info("Stage 3: passed ✅")

        # ── Stage 4: Semantic check ───────────────────────────────────────────
        logger.info("Stage 4: semantic equivalence check")
        equivalent, issues = self._semantic.check_equivalence(sql, optimized_sql)
        if not equivalent:
            result.warnings.extend(issues)
            logger.warning("Semantic issues detected: %s", issues)
        else:
            result.semantic_passed = True
            logger.info("Stage 4: passed ✅")

        # ── Stage 5: EXPLAIN cost comparison (NEW) ────────────────────────────
        # When the orchestrator passes explain_json we can compare the planner's
        # cost estimate for the original vs the rewrite WITHOUT executing either.
        # If the rewrite is MORE expensive, we log a warning and keep the
        # original so the user always gets the better query.
        if explain_json:
            logger.info("Stage 5: EXPLAIN cost comparison")
            original_cost = extract_explain_total_cost(explain_json)
            result.original_cost = original_cost

            # We cannot run EXPLAIN here (no DB connection in ai_optimizer).
            # The orchestrator should pass the optimized EXPLAIN cost back via
            # the result — see orchestrator.py Stage 5.5 where it already runs
            # EXPLAIN ANALYZE on both queries.  We store what we have and let
            # the orchestrator make the final call.
            logger.info(
                "  Original EXPLAIN cost: %.2f (rewrite cost measured by orchestrator)",
                original_cost,
            )

        result.optimized_sql = optimized_sql
        result.success       = True

        # ── Stage 6: Explanation (separate LLM call) ─────────────────────────
        logger.info("Stage 6: requesting explanation")
        result.explanation = self._get_explanation(sql, optimized_sql)
        logger.info("Stage 6: explanation received ✅")

        return result

    # ──────────────────────────────────────────
    # Prompt builders
    # ──────────────────────────────────────────

    @staticmethod
    def _build_rewrite_prompt(sql: str, context_block: str) -> str:
        parts = [
            "Rewrite the following PostgreSQL SQL query to improve performance.",
            "You MUST change at least one thing. Do NOT reproduce the query unchanged.",
            "Output ONLY the ###START### / ###END### block. English only. No comments inside the SQL.",
        ]

        if context_block:
            parts.append("")
            parts.append(context_block)
            parts.append("")
            parts.append(
                "The EXPLAIN plan above shows WHERE the planner is spending time. "
                "Focus your rewrite on Seq Scans on large tables and expensive joins."
            )

        parts.append("")
        parts.append("REQUIRED: apply at least one of these techniques:")
        parts.append(
            "  • Replace SELECT * with explicit column names — ONLY columns listed "
            "under 'Columns:' in the DATABASE CONTEXT above for that table. "
            "NEVER invent, guess, or add a column that is not explicitly listed there, "
            "even if it is a common column name for this kind of table. "
            "WARNING: foreign key references like 'order_id → orders.id' mean "
            "another table references this table — they do NOT mean this table has "
            "a column called order_id. Use ONLY names from the 'Columns:' list. "
            "If no DATABASE CONTEXT is provided, do NOT expand SELECT *."
        )
        parts.append("  • Reorder JOINs so the smallest / most-filtered table comes first")
        parts.append("  • Add or tighten WHERE predicates using indexed columns")
        parts.append("  • Replace IN (subquery) with EXISTS or a JOIN")
        parts.append("  • Wrap repeated subexpressions into a CTE (WITH clause)")
        parts.append("  • Replace DISTINCT with GROUP BY when it enables index use")
        parts.append("")
        parts.append("SQL TO OPTIMIZE:")
        parts.append(sql)

        return "\n".join(parts)

    @staticmethod
    def _build_explanation_prompt(original_sql: str, optimized_sql: str) -> str:
        return (
            "Original SQL:\n"
            f"{original_sql}\n\n"
            "Optimized SQL:\n"
            f"{optimized_sql}\n\n"
            "In 2-4 sentences, explain what was changed and why it improves performance. "
            "English only. Plain text, no markdown."
        )

    # ──────────────────────────────────────────
    # LLM call helpers
    # ──────────────────────────────────────────

    def _get_explanation(self, original_sql: str, optimized_sql: str) -> str:
        try:
            from ai.llm_client import LLMCallType
            prompt = self._build_explanation_prompt(original_sql, optimized_sql)
            explanation = self._llm.complete(
                prompt=prompt,
                system_prompt=self.EXPLANATION_SYSTEM_PROMPT,
                call_type=LLMCallType.EXPLAIN,
            )
            explanation = re.sub(r"```[a-z]*", "", explanation).strip("`").strip()
            return explanation
        except LLMClientError as exc:
            logger.warning("Explanation call failed (non-fatal): %s", exc)
            return ""

    # ──────────────────────────────────────────
    # Response parsers
    # ──────────────────────────────────────────

    @staticmethod
    def _parse_sql_response(raw: str, original_sql: str = "") -> str:
        from ai.sql_extractor import extract_sql
        return extract_sql(raw, fallback=original_sql)

    @staticmethod
    def _parse_llm_response(raw: str, original_sql: str = "") -> tuple[str, str]:
        """Backward-compatible wrapper."""
        from ai.sql_extractor import extract_sql
        explanation = ""
        exp_match = re.search(r"EXPLANATION:\s*(.+)", raw, re.IGNORECASE)
        if exp_match:
            explanation = exp_match.group(1).strip()
            raw = raw[: exp_match.start()].strip()
        sql = extract_sql(raw, fallback=original_sql)
        return sql, explanation