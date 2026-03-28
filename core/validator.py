"""
validator.py — Validates SQL syntax before processing.
"""
import sqlparse
from utils.logger import get_logger

logger = get_logger(__name__)

def validate_sql(sql: str) -> tuple[bool, str]:
    """Returns (is_valid, message)."""
    try:
        parsed = sqlparse.parse(sql.strip())
        if not parsed or not parsed[0].tokens:
            return False, "❌ Empty or invalid SQL."
        logger.info("SQL validation passed.")
        return True, "✅ SQL is valid."
    except Exception as e:
        return False, f"❌ Validation error: {e}"
