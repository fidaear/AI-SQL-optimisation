"""
statistics_extractor.py — Extracts table statistics from PostgreSQL.
"""
from db.connection_manager import get_connection, close_connection
from utils.logger import get_logger

logger = get_logger(__name__)

def get_table_stats(table_name: str) -> dict:
    """Returns row count and existing indexes for a given table."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(f"SELECT reltuples::BIGINT FROM pg_class WHERE relname = %s", (table_name,))
        row = cur.fetchone()
        row_count = row[0] if row else 0

        cur.execute("""
            SELECT indexname FROM pg_indexes WHERE tablename = %s
        """, (table_name,))
        indexes = [r[0] for r in cur.fetchall()]

        logger.info(f"Stats for '{table_name}': {row_count} rows, {len(indexes)} indexes.")
        return {"table": table_name, "row_count": row_count, "existing_indexes": indexes}
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {}
    finally:
        close_connection(conn)
