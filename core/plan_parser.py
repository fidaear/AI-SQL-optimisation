"""
plan_parser.py — Parses the output of EXPLAIN ANALYZE into structured data.
"""
import re
from utils.logger import get_logger

logger = get_logger(__name__)

def parse_plan(plan_rows: list[dict]) -> dict:
    """
    Extracts key metrics from EXPLAIN ANALYZE output.
    Returns a dict with: scan_type, execution_time, rows_scanned.
    """
    result = {"scan_type": "Unknown", "execution_time_ms": 0.0, "rows_scanned": 0}

    for row in plan_rows:
        line = row.get("plan", "")

        if "Seq Scan" in line:
            result["scan_type"] = "Sequential Scan"
        elif "Index Scan" in line:
            result["scan_type"] = "Index Scan"
        elif "Bitmap" in line:
            result["scan_type"] = "Bitmap Scan"

        time_match = re.search(r"actual time=[\d.]+\.\.([\d.]+)", line)
        if time_match:
            result["execution_time_ms"] = float(time_match.group(1))

        rows_match = re.search(r"rows=(\d+)", line)
        if rows_match:
            result["rows_scanned"] = int(rows_match.group(1))

    logger.info(f"Plan parsed: {result}")
    return result
