import re
from utils.logger import get_logger
 
logger = get_logger(__name__)
 
LARGE_TABLES = {
    "orders", "order_items", "customers",
    "products", "employees", "transactions", "logs"
}
 
# ── Helpers ───────────────────────────────────────────────────
 
def _get_tables(sql: str) -> set:
    return {m.group(1).lower()
            for m in re.finditer(r'(?:FROM|JOIN)\s+(\w+)', sql, re.IGNORECASE)}
 
def _has_where(sql: str) -> bool:
    return bool(re.search(r'\bWHERE\b', sql, re.IGNORECASE))
 
def _count_joins(sql: str) -> int:
    return len(re.findall(r'\bJOIN\b', sql, re.IGNORECASE))
 
def _get_join_columns(sql: str) -> list:
    return re.findall(
        r'ON\s+\w+\.(\w+)\s*=\s*\w+\.(\w+)', sql, re.IGNORECASE)
 
def _get_where_body(sql: str) -> str:
    m = re.search(
        r'WHERE(.+?)(?:ORDER\s+BY|GROUP\s+BY|LIMIT|HAVING|$)',
        sql, re.IGNORECASE | re.DOTALL)
    return m.group(1) if m else ""
 
# ── Rule 1 — SELECT * ─────────────────────────────────────────
 
def rule_select_star(sql: str):
    if re.search(r'SELECT\s+\*', sql, re.IGNORECASE):
        tables = _get_tables(sql)
        ex = f"SELECT id, name, status FROM {next(iter(tables), 'table')}"
        return {
            "rule"        : "SELECT * detected",
            "severity"    : "MEDIUM",
            "category"    : "I/O Optimization",
            "advice"      : ("SELECT * fetches every column including unused ones. "
                             "Specify only the columns you need."),
            "code_example": f"-- Instead of: SELECT *\n-- Use:        {ex}",
        }
 
# ── Rule 2 — Missing WHERE / Seq Scan ─────────────────────────
 
def rule_missing_where(sql: str):
    tables = _get_tables(sql)
    large  = tables & LARGE_TABLES
    if not _has_where(sql) and large:
        return {
            "rule"        : "Seq Scan — missing WHERE on large table",
            "severity"    : "HIGH",
            "category"    : "Sequential Scan",
            "advice"      : (f"Query runs on large table(s) [{', '.join(large)}] "
                             "without any WHERE filter — full Sequential Scan guaranteed."),
            "code_example": ("WHERE created_at >= NOW() - INTERVAL '30 days'"),
        }
 
# ── Rule 3 — Leading wildcard LIKE ────────────────────────────
 
def rule_leading_wildcard(sql: str):
    m = re.search(r"LIKE\s+'(%[^']*)'", sql, re.IGNORECASE)
    if m:
        return {
            "rule"        : f"Leading wildcard LIKE '{m.group(1)}'",
            "severity"    : "HIGH",
            "category"    : "Selectivity / Index Usage",
            "advice"      : ("LIKE '%value' disables B-tree index usage — full table scan."),
            "code_example": ("-- Use trailing wildcard:\nWHERE name LIKE 'Smith%'"),
        }
 
# ── Rule 4 — OR in WHERE ──────────────────────────────────────
 
def rule_or_condition(sql: str):
    where = _get_where_body(sql)
    if where and re.search(r'\bOR\b', where, re.IGNORECASE):
        return {
            "rule"        : "OR condition in WHERE clause",
            "severity"    : "MEDIUM",
            "category"    : "Selectivity / Index Usage",
            "advice"      : ("OR conditions reduce selectivity and can prevent index usage. "
                             "Consider IN() or UNION ALL instead."),
            "code_example": ("-- Use IN() for same column:\n"
                             "WHERE status IN ('pending', 'cancelled')"),
        }
 
# ── Rule 5 — Missing index on JOIN columns ────────────────────
 
def rule_missing_join_index(sql: str):
    join_cols = _get_join_columns(sql)
    n_joins   = _count_joins(sql)
    if join_cols and n_joins >= 1:
        pairs = [f"({a}, {b})" for a, b in join_cols]
        return {
            "rule"        : f"JOIN columns likely missing indexes ({n_joins} JOIN detected)",
            "severity"    : "HIGH",
            "category"    : "Missing Index",
            "advice"      : (f"Detected {len(join_cols)} JOIN condition(s) on: "
                             f"{', '.join(pairs)}. Both sides need an index."),
            "code_example": ("CREATE INDEX IF NOT EXISTS idx_orders_customer_id "
                             "ON orders(customer_id);\n"
                             "CREATE INDEX IF NOT EXISTS idx_customers_id "
                             "ON customers(id);"),
        }
 
# ── Rule 6 — Too many JOINs ───────────────────────────────────
 
def rule_too_many_joins(sql: str):
    n = _count_joins(sql)
    if n >= 3:
        return {
            "rule"        : f"Too many JOINs ({n} detected)",
            "severity"    : "MEDIUM",
            "category"    : "Query Complexity",
            "advice"      : (f"This query has {n} JOINs. High JOIN counts increase "
                             "planner complexity and risk of suboptimal plans. "
                             "Consider CTEs to simplify."),
            "code_example": ("WITH recent AS (\n"
                             "  SELECT id, customer_id FROM orders\n"
                             "  WHERE order_date >= '2023-01-01'\n"
                             ")\n"
                             "SELECT c.name FROM recent r\n"
                             "JOIN customers c ON r.customer_id = c.id;"),
        }
 
# ── Rule 7 — Non-SARGable (function on column) ────────────────
 
def rule_non_sargable(sql: str):
    patterns = [
        (r'\bUPPER\s*\(',        'UPPER(column)'),
        (r'\bLOWER\s*\(',        'LOWER(column)'),
        (r'\bYEAR\s*\(',         'YEAR(column)'),
        (r'\bMONTH\s*\(',        'MONTH(column)'),
        (r'\bDATE\s*\(',         'DATE(column)'),
        (r'\bTO_CHAR\s*\(',      'TO_CHAR(column)'),
        (r'\bEXTRACT\s*\(',      'EXTRACT(... FROM column)'),
        (r'\bCAST\s*\(',         'CAST(column AS ...)'),
    ]
    where = _get_where_body(sql)
    if not where:
        return None
    for pat, label in patterns:
        if re.search(pat, where, re.IGNORECASE):
            return {
                "rule"        : f"Non-SARGable condition: {label} in WHERE",
                "severity"    : "HIGH",
                "category"    : "Selectivity / Index Usage",
                "advice"      : (f"{label} in WHERE prevents index usage — "
                                 "the DB applies the function to every row."),
                "code_example": ("-- Non-SARGable:\nWHERE YEAR(order_date) = 2023\n\n"
                                 "-- SARGable:\n"
                                 "WHERE order_date BETWEEN '2023-01-01' AND '2023-12-31'"),
            }
 
# ── Rule 8 — SELECT DISTINCT ──────────────────────────────────
 
def rule_select_distinct(sql: str):
    if re.search(r'\bSELECT\s+DISTINCT\b', sql, re.IGNORECASE):
        return {
            "rule"        : "SELECT DISTINCT overuse",
            "severity"    : "LOW",
            "category"    : "Query Design",
            "advice"      : ("DISTINCT sorts and deduplicates all rows — expensive. "
                             "It often signals a missing JOIN condition."),
            "code_example": ("-- Use EXISTS instead:\n"
                             "WHERE EXISTS (SELECT 1 FROM orders o "
                             "WHERE o.customer_id = c.id)"),
        }
 
# ── Rule 9 — No LIMIT on large table ──────────────────────────
 
def rule_missing_limit(sql: str):
    tables    = _get_tables(sql)
    large     = tables & LARGE_TABLES
    has_limit = bool(re.search(r'\bLIMIT\b', sql, re.IGNORECASE))
    has_agg   = bool(re.search(
        r'\b(COUNT|SUM|AVG|MAX|MIN)\s*\(', sql, re.IGNORECASE))
    if large and not has_limit and not has_agg:
        return {
            "rule"        : f"No LIMIT on large table [{', '.join(large)}]",
            "severity"    : "LOW",
            "category"    : "Sequential Scan",
            "advice"      : ("Without LIMIT, PostgreSQL fetches ALL matching rows. "
                             "Add LIMIT for pagination."),
            "code_example": ("ORDER BY created_at DESC\nLIMIT 50 OFFSET 0;"),
        }
 
# ── Rule 10 — Large IN() clause ───────────────────────────────
 
def rule_large_in_clause(sql: str):
    for match in re.findall(r'\bIN\s*\(([^)]+)\)', sql, re.IGNORECASE):
        values = [v.strip() for v in match.split(',')]
        if len(values) > 10:
            return {
                "rule"        : f"Large IN() clause ({len(values)} values)",
                "severity"    : "MEDIUM",
                "category"    : "Selectivity",
                "advice"      : (f"IN() with {len(values)} values generates a large OR chain. "
                                 "Use a subquery or temp table instead."),
                "code_example": ("-- Use subquery:\n"
                                 "WHERE id IN (SELECT id FROM target_ids)"),
            }
 
# ── Main analyzer ─────────────────────────────────────────────
 
RULES = [
    rule_select_star,
    rule_missing_where,
    rule_leading_wildcard,
    rule_or_condition,
    rule_missing_join_index,
    rule_too_many_joins,
    rule_non_sargable,
    rule_select_distinct,
    rule_missing_limit,
    rule_large_in_clause,
]
 
def analyze(sql: str) -> list:
    """
    Runs all 10 performance rules on the SQL query.
    Returns list of issues sorted by severity: HIGH → MEDIUM → LOW.
    """
    issues = []
    for rule_fn in RULES:
        try:
            result = rule_fn(sql)
            if result:
                issues.append(result)
        except Exception as e:
            logger.warning(f"Rule {rule_fn.__name__} failed: {e}")
 
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    issues.sort(key=lambda x: order.get(x["severity"], 3))
 
    logger.info(
        f"Rule engine found {len(issues)} issue(s) "
        f"out of {len(RULES)} rules checked.")
    return issues
 
 
if __name__ == "__main__":
    sql = """
    SELECT * FROM orders o
    JOIN customers c ON o.customer_id = c.id
    JOIN order_items oi ON o.id = oi.order_id
    JOIN products p ON oi.product_id = p.id
    WHERE c.country = 'Morocco'
      AND p.category = 'Electronics'
      AND o.status = 'delivered'
      AND o.order_date BETWEEN '2023-01-01' AND '2024-12-31';
    """
    results = analyze(sql)
    for r in results:
        icon = {"HIGH":"🔴","MEDIUM":"🟠","LOW":"🟡"}[r["severity"]]
        print(f"{icon} [{r['severity']}] {r['rule']}")
        print(f"   → {r['advice'][:70]}...\n")