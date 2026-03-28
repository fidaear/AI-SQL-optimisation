"""
ab_tester.py — Compares query performance before and after optimization.
"""
from core.executor import run_query
from utils.logger import get_logger

logger = get_logger(__name__)

def run_ab_test(sql_a: str, sql_b: str) -> dict:
    """
    Runs both queries and returns a performance comparison.
    Version A = original, Version B = optimized.
    """
    _, time_a = run_query(sql_a)
    _, time_b = run_query(sql_b)

    improvement = round(((time_a - time_b) / time_a) * 100, 1) if time_a > 0 else 0

    result = {
        "version_a_ms": time_a,
        "version_b_ms": time_b,
        "improvement_pct": improvement,
        "winner": "B (Optimized)" if time_b < time_a else "A (Original)"
    }
    logger.info(f"A/B Test result: {result}")
    return result
