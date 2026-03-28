"""
connection_manager.py — Manages PostgreSQL database connections.
"""
import psycopg2
from utils.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from utils.logger import get_logger

logger = get_logger(__name__)

def get_connection():
    """Returns a new PostgreSQL connection."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST, port=DB_PORT,
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD
        )
        logger.info("Database connection established.")
        return conn
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        raise

def close_connection(conn):
    if conn:
        conn.close()
        logger.info("Database connection closed.")
