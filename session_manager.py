"""
db/session_manager.py
=====================
In-memory session store.

Public API (matches test_db.py):
  SessionManager.create_session(**kwargs)        → str  (session_id)
  SessionManager.get_session(session_id)         → dict | None
  SessionManager.update_session(sid, data)       → None
  SessionManager.delete_session(session_id)      → None
  SessionManager.clear_all()                     → None  (test helper)
"""

import uuid
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SessionManager:
    """
    Simple in-memory session store.

    All methods are class methods so no instantiation is needed.
    State is held in a class-level dict (_sessions).
    """

    _sessions: Dict[str, Dict[str, Any]] = {}

    # ── Create ────────────────────────────────────────────────────────────────

    @classmethod
    def create_session(cls, **kwargs) -> str:
        """
        Create a new session and store any provided keyword arguments as
        session data.

        Parameters
        ----------
        **kwargs
            Arbitrary key/value pairs stored in the session
            (e.g. user_id="u1", query="SELECT ...").

        Returns
        -------
        str
            A unique session ID (UUID4).
        """
        session_id = str(uuid.uuid4())
        cls._sessions[session_id] = dict(kwargs)
        logger.debug("Session created: %s", session_id)
        return session_id

    # ── Read ──────────────────────────────────────────────────────────────────

    @classmethod
    def get_session(cls, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data by ID.

        Parameters
        ----------
        session_id : str

        Returns
        -------
        dict | None
            The session data dict, or None if the session does not exist.
        """
        return cls._sessions.get(session_id)

    # ── Update ────────────────────────────────────────────────────────────────

    @classmethod
    def update_session(cls, session_id: str, data: Dict[str, Any]) -> None:
        """
        Merge *data* into an existing session.  Existing keys are preserved;
        keys in *data* are added or overwritten.

        Parameters
        ----------
        session_id : str
        data       : dict
        """
        if session_id in cls._sessions:
            cls._sessions[session_id].update(data)
            logger.debug("Session updated: %s", session_id)
        else:
            logger.warning("update_session called on unknown session: %s", session_id)

    # ── Delete ────────────────────────────────────────────────────────────────

    @classmethod
    def delete_session(cls, session_id: str) -> None:
        """
        Remove a session.  Does NOT raise if the session does not exist.

        Parameters
        ----------
        session_id : str
        """
        cls._sessions.pop(session_id, None)
        logger.debug("Session deleted (or was already absent): %s", session_id)

    # ── Test helper ───────────────────────────────────────────────────────────

    @classmethod
    def clear_all(cls) -> None:
        """
        Remove all sessions.  Called by test setup_method to ensure a clean
        state between tests.
        """
        cls._sessions.clear()
        logger.debug("All sessions cleared")


# ---------------------------------------------------------------------------
# Singleton accessor (kept for backward-compatibility with old imports)
# ---------------------------------------------------------------------------

_singleton: Optional[SessionManager] = None


def get_session_manager() -> SessionManager:
    """Return the module-level SessionManager singleton."""
    global _singleton
    if _singleton is None:
        _singleton = SessionManager()
    return _singleton