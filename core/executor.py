"""
executor.py — Executes SQL queries and EXPLAIN plans on PostgreSQL.
"""
from db.connection_manager import get_connection, close_connection
from utils.logger import get_logger

logger = get_logger(__name__)

def run_explain(sql: str) -> list[dict]:
    """Runs EXPLAIN ANALYZE and returns the plan rows."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN ANALYZE {sql}")
        rows = cur.fetchall()
        logger.info("EXPLAIN ANALYZE executed successfully.")
        return [{"plan": row[0]} for row in rows]
    except Exception as e:
        logger.error(f"Execution error: {e}")
        return []
    finally:
        close_connection(conn)

def run_query(sql: str) -> tuple[list, float]:
    """Executes the SQL and returns (rows, execution_time_ms)."""
    import time
    conn = get_connection()
    try:
        cur = conn.cursor()
        start = time.time()
        cur.execute(sql)
        rows = cur.fetchall()
        elapsed = round((time.time() - start) * 1000, 2)
        logger.info(f"Query executed in {elapsed}ms.")
        return rows, elapsed
    except Exception as e:
        logger.error(f"Query error: {e}")
        return [], 0.0
    finally:
        close_connection(conn)
