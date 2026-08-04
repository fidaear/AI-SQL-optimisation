# AI SQL Optimizer - Secure Database Service Module
# Author: CHATER Douae
# Purpose: Execute queries safely with EXPLAIN analysis, validation, and transaction safety

import logging
from typing import Optional, Dict, Any, List
import pg8000

from db.session_manager import get_session_manager
from security.query_sanitizer import QuerySanitizer, QueryValidationError
from security.permission_checker import PermissionChecker, PermissionDeniedError


logger = logging.getLogger(__name__)


class DatabaseConnectionError(Exception):
    """Raised when database connection fails."""
    pass


class DatabaseQueryError(Exception):
    """Raised when database query execution fails."""
    pass


class QueryValidator:
    """Backward-compatible alias around QuerySanitizer."""

    @staticmethod
    def is_select_only(query: str) -> bool:
        return QuerySanitizer.validate_select_query(query)


class SecureDatabaseService:
    """
    Executes queries securely with EXPLAIN analysis, validation, and rollback safety.
    """

    EXPLAIN_FORMAT = "EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)"
    EXPLAIN_ONLY_FORMAT = "EXPLAIN (FORMAT JSON)"

    def __init__(self, session_manager=None, connection_timeout: int = 10):
        """
        Initialize SecureDatabaseService.

        Args:
            session_manager: Optional SessionManager instance (uses singleton if not provided)
            connection_timeout: Database connection timeout in seconds
        """
        self.connection_timeout = connection_timeout
        self.session_manager = session_manager or get_session_manager()

    def _create_connection(self, credentials):
        """
        Create a secure database connection using dictionary keys.
        """
        try:
            conn = pg8000.connect(
                host=credentials.get('host').strip(),
                database=credentials.get('dbname').strip(),
                user=credentials.get('user').strip(),
                password=credentials.get('password').strip(),   
                timeout=self.connection_timeout
            )
            
            # Correction des logs (on utilise credentials['host'] au lieu de .host)
            logger.info(f"Database connection established to {credentials.get('host')}")
            return conn
        except Exception as e:
            # Correction de l'affichage de l'erreur
            host_info = credentials.get('host', 'unknown')
            logger.error(f"Database connection failed for {host_info}: {str(e)}")
            raise DatabaseConnectionError(
                f"Failed to connect to database at {host_info}: {str(e)}"
            )

    def run_secure_explain(
        self,
        session_id: str,
        user_query: str,
        analyze: bool = True
    ) -> Dict[str, Any]:
        """
        Execute a SELECT query with EXPLAIN analysis in a safe transaction.

        Security guarantees:
        - Validates query is SELECT-only (no DDL/DML)
        - Wraps in EXPLAIN with ANALYZE and BUFFERS
        - Uses READ-ONLY transaction
        - Always ROLLBACKs at the end (no side effects)
        - Credentials never logged or exposed

        Args:
            session_id: Valid session ID from SessionManager
            user_query: User's SQL SELECT query
            analyze: Whether to run ANALYZE (default: True). If False, just explains plan

        Returns:
            Dictionary containing:
            {
                'success': bool,
                'explain_plan': list (JSON parsed),
                'execution_time_ms': float,
                'rows_examined': int,
                'planning_time_ms': float,
                'query': str (original query),
                'timestamp': str (ISO format)
            }

        Raises:
            QueryValidationError: If query is not SELECT-only
            DatabaseConnectionError: If database connection fails
            Exception: For other database/execution errors
        """
        from datetime import datetime
        import time

        Logger = logger  # Avoid credential logging

        # Step 1: Validate query
        try:
            QueryValidator.is_select_only(user_query)
        except QueryValidationError as e:
            Logger.warning(f"Query validation failed: {str(e)}")
            raise
        user_query = user_query.strip().rstrip(';') 
        # Step 2: Get credentials from session
        try:
            session_data = self.session_manager.get_session(session_id)
            if not session_data or 'credentials' not in session_data:
                Logger.warning(f"Invalid or expired session: {session_id[:8]}...")
                raise ValueError("Invalid or expired session. Please create a new session.")
            credentials = session_data['credentials']
        except Exception as e:
            Logger.warning(f"Session validation failed: {str(e)}")
            raise ValueError(str(e))

        # Step 3: Connect to database
        conn = None
        cursor = None
        start_time = time.time()

        try:
            conn = self._create_connection(credentials)

            # Step 4: Build EXPLAIN query
            explain_prefix = self.EXPLAIN_FORMAT if analyze else self.EXPLAIN_ONLY_FORMAT
            explain_query = f"{explain_prefix}\n{user_query}"

            Logger.debug(f"Executing EXPLAIN query (session: {session_id[:8]}...)")

            # Step 5: Execute in transaction (will always rollback)
            cursor = conn.cursor()

            # Set read-only mode for extra safety
            cursor.execute("SET TRANSACTION READ ONLY")

            # Execute EXPLAIN
            cursor.execute(explain_query)
            result = cursor.fetchall()

            # pg8000 returns pre-parsed dicts directly, not JSON strings
            # result[0][0] is already a dict from pg8000, not a JSON string
            explain_plan = result[0][0]
            
            # Ensure it's in list format for consistency
            if isinstance(explain_plan, dict):
                explain_plan = [explain_plan]

            # Extract metrics from the plan
            execution_time_ms = 0
            planning_time_ms = 0
            rows_examined = 0

            if isinstance(explain_plan, list) and len(explain_plan) > 0:
                plan = explain_plan[0]
                execution_time_ms = plan.get('Execution Time', 0)
                planning_time_ms = plan.get('Planning Time', 0)

                # Try to extract total rows
                if 'Plan' in plan:
                    plan_detail = plan['Plan']
                    rows_examined = plan_detail.get('Actual Rows', 
                                                   plan_detail.get('Rows', 0))

            execution_time = time.time() - start_time

            return {
                'success': True,
                'explain_plan': explain_plan,
                'execution_time_ms': execution_time * 1000,  # Convert to milliseconds
                'rows_examined': rows_examined,
                'planning_time_ms': planning_time_ms,
                'query': user_query,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseConnectionError:
            raise

        except Exception as e:
            Logger.error(f"Database error during query execution: {str(e)}")
            raise DatabaseConnectionError(f"Database error: {str(e)}")

        finally:
            # Step 6: Always rollback and cleanup
            if cursor:
                try:
                    cursor.execute("ROLLBACK")
                    cursor.close()
                except:
                    pass

            if conn:
                try:
                    conn.rollback()  # Ensure rollback
                    conn.close()
                    Logger.debug(f"Database connection closed and rolled back (session: {session_id[:8]}...)")
                except:
                    pass

    def run_secure_query(
        self,
        session_id: str,
        user_query: str,
        limit: int = 1000
    ) -> Dict[str, Any]:
        """
        Execute a SELECT query and return results safely (with row limit).

        Security guarantees:
        - Validates query is SELECT-only
        - Uses READ-ONLY transaction
        - Always ROLLBACKs
        - Limits result rows

        Args:
            session_id: Valid session ID
            user_query: SELECT query
            limit: Maximum rows to return (default: 1000)

        Returns:
            Dictionary with results, column names, and metadata
        """
        from datetime import datetime
        import time

        # Validate query
        try:
            QuerySanitizer.validate_select_query(user_query)
        except QueryValidationError as e:
            logger.warning(f"Query validation failed: {str(e)}")
            raise

        # Get credentials
        try:
            session_data = self.session_manager.get_session(session_id)
            if not session_data or 'credentials' not in session_data:
                logger.warning(f"Invalid or expired session: {session_id[:8]}...")
                raise ValueError("Invalid or expired session. Please create a new session.")
            credentials = session_data['credentials']
        except Exception as e:
            raise ValueError(str(e))

        conn = None
        cursor = None
        start_time = time.time()
        user_query = user_query.strip().rstrip(';') 
        try:
            conn = self._create_connection(credentials)
            cursor = conn.cursor()

            # Set read-only
            cursor.execute("SET TRANSACTION READ ONLY")

            limited_query = QuerySanitizer.enforce_limit(user_query, limit)

            cursor.execute(limited_query)

            # Fetch results
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            execution_time = time.time() - start_time

            return {
                'success': True,
                'columns': columns,
                'rows': rows,
                'row_count': len(rows),
                'execution_time_ms': execution_time * 1000,
                'query': user_query,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseConnectionError:
            raise

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise DatabaseConnectionError(f"Database error: {str(e)}")

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass

            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass

    def run_query_with_metrics(
        self,
        session_id: str,
        user_query: str,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Execute a SELECT query and return both execution metrics and data rows.
        
        Returns dict with:
        {
            'success': bool,
            'rows': list of dicts (first 'limit' rows),
            'columns': list of column names,
            'execution_time_ms': float from PostgreSQL EXPLAIN ANALYZE,
            'row_count': int (total rows if available),
            'query': str
        }
        """
        from datetime import datetime
        import time

        logger = logging.getLogger(__name__)

        # Validate query
        try:
            QuerySanitizer.validate_select_query(user_query)
        except QueryValidationError as e:
            logger.warning(f"Query validation failed: {str(e)}")
            raise

        # Get credentials
        try:
            session_data = self.session_manager.get_session(session_id)
            if not session_data or 'credentials' not in session_data:
                logger.warning(f"Invalid or expired session: {session_id[:8]}...")
                raise ValueError("Invalid or expired session. Please create a new session.")
            credentials = session_data['credentials']
        except Exception as e:
            raise ValueError(str(e))
        user_query = user_query.strip().rstrip(';') 
        conn = None
        cursor = None
        start_time = time.time()

        try:
            conn = self._create_connection(credentials)
            cursor = conn.cursor()

            # Set read-only mode
            cursor.execute("SET TRANSACTION READ ONLY")

            # Enforce limit for safety
            limited_query = QuerySanitizer.enforce_limit(user_query, limit)

            # Execute the actual query to get data
            cursor.execute(limited_query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]

            # Extract execution time from query execution
            execution_time = time.time() - start_time

            # Convert pg8000 tuples to dicts
            result_dicts = [
                {columns[i]: row[i] for i in range(len(columns))}
                for row in rows
            ]

            return {
                'success': True,
                'rows': result_dicts,
                'columns': columns,
                'execution_time_ms': execution_time * 1000,
                'row_count': len(rows),
                'query': user_query,
                'timestamp': datetime.utcnow().isoformat()
            }

        except DatabaseConnectionError:
            raise

        except Exception as e:
            logger.error(f"Database error: {str(e)}")
            raise DatabaseConnectionError(f"Database error: {str(e)}")

        finally:
            if cursor:
                try:
                    cursor.close()
                except:
                    pass

            if conn:
                try:
                    conn.rollback()
                    conn.close()
                except:
                    pass


# Factory function for easy instantiation
def get_database_service(session_manager=None) -> SecureDatabaseService:
    """
    Create and return a SecureDatabaseService instance.
    
    Args:
        session_manager: Optional SessionManager instance. If not provided, uses singleton.
    """
    return SecureDatabaseService(session_manager=session_manager)
