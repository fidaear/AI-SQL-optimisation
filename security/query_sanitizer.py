"""
query_sanitizer.py — Ensures only safe SELECT queries are processed.
"""
import re
from utils.logger import get_logger

logger = get_logger(__name__)

FORBIDDEN = ["DROP", "DELETE", "INSERT", "UPDATE", "TRUNCATE", "ALTER", "CREATE", "--", ";--"]

def sanitize(sql: str) -> tuple[bool, str]:
    """
    Returns (is_safe, message).
    Only SELECT queries are allowed.
    """
    sql_upper = sql.upper().strip()

    if not sql_upper.startswith("SELECT"):
        return False, "❌ Only SELECT queries are allowed."

    for word in FORBIDDEN:
        if word in sql_upper:
            return False, f"❌ Forbidden keyword detected: {word}"

    logger.info("Query passed sanitization.")
    return True, "✅ Query is safe."
