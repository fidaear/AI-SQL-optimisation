"""
connection_manager.py — Manages PostgreSQL database connections.
"""
import psycopg2
import os
from dotenv import load_dotenv
from utils.logger import get_logger

# Charger le .env depuis la racine du projet
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
load_dotenv(os.path.join(_root, '.env'), override=True)

logger = get_logger(__name__)

def get_connection():
    """Returns a new PostgreSQL connection."""
    host     = os.getenv("DB_HOST", "localhost")
    port     = os.getenv("DB_PORT", "5432")
    dbname   = os.getenv("DB_NAME", "postgres")
    user     = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD", "")

    logger.info(f"Connecting to {host}:{port}/{dbname} as {user}")

    try:
        conn = psycopg2.connect(
            host=host, port=port,
            dbname=dbname, user=user, password=password
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