"""
session_manager.py — Manages query sessions and history.
"""
from utils.logger import get_logger

logger = get_logger(__name__)

class SessionManager:
    def __init__(self):
        self.history = []

    def add(self, sql: str, result: dict):
        self.history.append({"sql": sql, "result": result})
        logger.info(f"Session updated. Total queries: {len(self.history)}")

    def get_history(self) -> list:
        return self.history

    def clear(self):
        self.history = []
