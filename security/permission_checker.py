"""
permission_checker.py — Checks user permissions before query execution.
"""
from utils.logger import get_logger

logger = get_logger(__name__)

ALLOWED_USERS = ["admin", "analyst", "developer"]

def check_permission(username: str) -> bool:
    allowed = username in ALLOWED_USERS
    if allowed:
        logger.info(f"User '{username}' granted access.")
    else:
        logger.warning(f"User '{username}' denied access.")
    return allowed
