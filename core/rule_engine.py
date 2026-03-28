"""
rule_engine.py — Detects common SQL anti-patterns using rule-based analysis.
"""
import re
from utils.logger import get_logger

logger = get_logger(__name__)

def analyze(sql: str) -> list[dict]:
    """
    Returns a list of detected issues with advice.
    Each issue: {"rule": str, "severity": str, "advice": str}
    """
    issues = []
    sql_upper = sql.upper()

    if re.search(r"SELECT\s+\*", sql_upper):
        issues.append({
            "rule": "SELECT *",
            "severity": "MEDIUM",
            "advice": "Specify only needed columns to reduce I/O."
        })

    if not re.search(r"\bWHERE\b", sql_upper):
        issues.append({
            "rule": "Missing WHERE clause",
            "severity": "HIGH",
            "advice": "Add a WHERE clause to avoid full table scans."
        })

    if re.search(r"OR\b", sql_upper):
        issues.append({
            "rule": "OR in WHERE clause",
            "severity": "LOW",
            "advice": "OR can prevent index usage. Consider UNION instead."
        })

    if re.search(r"LIKE\s+'%", sql_upper):
        issues.append({
            "rule": "Leading wildcard LIKE",
            "severity": "HIGH",
            "advice": "LIKE '%value' disables index usage. Avoid leading wildcards."
        })

    logger.info(f"Rule engine found {len(issues)} issue(s).")
    return issues
