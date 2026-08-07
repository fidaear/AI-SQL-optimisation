"""
db_router.py
------------
Handles PostgreSQL session management and full schema introspection.
Place this file at:  backend/routers/db_router.py

Mount in main.py with:
    from routers.db_router import router as db_router
    app.include_router(db_router)
"""

import uuid
import logging
import os
from urllib.parse import quote
import psycopg2
import psycopg2.extras
from typing import Optional
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/db", tags=["database"])

# ---------------------------------------------------------------------------
# In-memory session registry  {session_id -> connection_params}
# For production replace with Redis or encrypted JWT claims.
# ---------------------------------------------------------------------------
from db.session_manager import SessionManager

# Public alias so sibling modules (e.g. db_tester.py) can import the same
# dict without going through a private name.
from db.session_manager import SessionManager
SESSION_STORE = SessionManager._sessions 


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ConnectRequest(BaseModel):
    host:     str = "localhost"
    port:     int = 5432
    dbname:   str
    user:     str
    password: str


class ConnectResponse(BaseModel):
    session_id: str
    sessionId:  str   # camelCase alias for Angular consumers
    dbname:     str
    host:       str
    tables:     int   # quick count so UI can confirm connection immediately


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_connection_string(host: str, port: int, dbname: str, user: str, password: str) -> str:
    """
    Build a PostgreSQL connection URI with proper URL encoding and UTF-8 handling.
    Format: postgresql://user:password@host:port/dbname
    
    All inputs are validated and properly encoded.
    """
    try:
        # Ensure all inputs are strings and strip whitespace
        host = str(host).strip()
        port = int(port)
        dbname = str(dbname).strip()
        user = str(user).strip()
        password = str(password).strip()
        
        logger.debug(f"Building connection string for user={user}, dbname={dbname}")
        
        # Encode strings to UTF-8 and URL-encode to handle special characters
        from urllib.parse import quote
        
        # URL-encode credentials (safe='' encodes everything except unreserved chars)
        encoded_user = quote(user.encode('utf-8') if isinstance(user, str) else user, safe='')
        encoded_password = quote(password.encode('utf-8') if isinstance(password, str) else password, safe='')
        encoded_dbname = quote(dbname.encode('utf-8') if isinstance(dbname, str) else dbname, safe='')
        
        # Build the connection string
        conn_str = f"postgresql://{encoded_user}:{encoded_password}@{host}:{port}/{encoded_dbname}"
        logger.debug(f"Connection string built (masked): postgresql://***:***@{host}:{port}/{encoded_dbname}")
        
        return conn_str
        
    except Exception as e:
        logger.error(f"Error building connection string: {e}", exc_info=True)
        raise

def _get_conn(session_id: str):
    """Open a fresh short-lived connection from stored params."""
    session_data = SessionManager.get_session(session_id)
    if not session_data:
     raise HTTPException(status_code=401, detail="Session not found. Please reconnect.")
    params = session_data.get('credentials') # db_service attend la clé 'credentials'
    if not params:
        raise HTTPException(status_code=401, detail="Session not found. Please reconnect.")
    try:
        # Use connection string format for better encoding handling
        conn_str = _build_connection_string(
            params['host'],
            params['port'],
            params['dbname'],
            params['user'],
            params['password']
        )
        conn = psycopg2.connect(conn_str, connect_timeout=5)
        conn.set_session(readonly=True, autocommit=True)
        return conn
    except psycopg2.OperationalError as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to database: {e}")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.post("/test")
def test_connection(req: ConnectRequest):
    """
    Validate credentials WITHOUT creating a session.
    Opens a connection, runs SELECT 1, then closes immediately.
    Returns 200 on success, 400 on failure.
    """
    try:
        logger.info(f"Test connection request: host={req.host}, port={req.port}, dbname={req.dbname}, user={req.user}")
        logger.info(f"Password length: {len(req.password)} chars")
        logger.debug(f"Password repr: {repr(req.password)}")
        
        password = req.password
        
        # Build connection string with proper encoding for special characters
        logger.info("Building connection string for test...")
        conn_str = _build_connection_string(req.host, req.port, req.dbname, req.user, password)
        logger.info("✓ Connection string built successfully")
        logger.info(f"Connection string is ASCII: {conn_str.isascii()}")
        
        logger.info(f"Testing connection to {req.host}:{req.port}/{req.dbname}")
        logger.info(f"Connection string (first 100 chars): {conn_str[:100]}")
        try:
            conn = psycopg2.connect(conn_str, connect_timeout=5)
        except UnicodeDecodeError as e:
            logger.error(f"Unicode error in connection string: {e}")
            logger.error(f"Connection string: {conn_str}")
            raise
        logger.info("✓ Connection successful")
        
        cur  = conn.cursor()
        cur.execute("SELECT 1")
        logger.info("✓ SELECT 1 query successful")
        cur.close()
        conn.close()
        logger.info("✓ Test connection completed successfully")
        return {"status": "ok"}
        
    except UnicodeDecodeError as e:
        error_msg = f"UTF-8 Encoding Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except ValueError as e:
        error_msg = f"Invalid Password Format: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except psycopg2.OperationalError as e:
        error_msg = f"Database Connection Failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during connection test: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.post("/connect", response_model=ConnectResponse)
def connect(req: ConnectRequest):
    """
    Validate credentials by opening a test connection, then store params
    server-side keyed by a new session_id.
    """
    try:
        logger.info(f"Connect request: host={req.host}, port={req.port}, dbname={req.dbname}, user={req.user}")
        logger.info(f"Password length: {len(req.password)} chars")
        logger.debug(f"Password repr: {repr(req.password)}")
        
        password = req.password
        
        # Build connection string with proper encoding for special characters
        logger.info("Building connection string...")
        conn_str = _build_connection_string(req.host, req.port, req.dbname, req.user, password)
        logger.info("✓ Connection string built successfully")
        logger.info(f"Connection string is ASCII: {conn_str.isascii()}")
        
        logger.info(f"Attempting connection to {req.host}:{req.port}/{req.dbname}")
        try:
            conn = psycopg2.connect(conn_str, connect_timeout=5)
        except UnicodeDecodeError as e:
            logger.error(f"Unicode error in connection string: {e}")
            logger.error(f"Connection string: {conn_str}")
            raise
        logger.info("✓ Connection successful")
        conn.set_session(readonly=True, autocommit=True)
        
        cur = conn.cursor()
        logger.info("Executing table count query...")
        cur.execute("""
            SELECT COUNT(*) FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        """)
        result = cur.fetchone()
        table_count = int(result[0]) if result else 0
        logger.info(f"✓ Found {table_count} tables")
        cur.close()
        conn.close()
        
        # Store original params dict (not connection string) for later retrieval
        session_id = SessionManager.create_session(credentials={
            "host":     req.host,
            "port":     req.port,
            "dbname":   req.dbname,
            "user":     req.user,
            "password": password,
        })
        logger.info(f"✓ Session created: {session_id}")
        
        response = ConnectResponse(
            session_id=session_id,
            sessionId=session_id,   # camelCase alias
            dbname=req.dbname,
            host=req.host,
            tables=table_count,
        )
        logger.info(f"✓ Response ready to send")
        return response
        
    except UnicodeDecodeError as e:
        error_msg = f"UTF-8 Encoding Error: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except ValueError as e:
        error_msg = f"Invalid Password Format: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except psycopg2.OperationalError as e:
        error_msg = f"Database Connection Failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        error_msg = f"Unexpected error during connection: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)


@router.get("/schema")
def get_schema(x_session_id: str = Header(..., alias="X-Session-ID")):
    """
    Full schema introspection:
      - tables with row estimates
      - columns with type, nullable, default, PK flag
      - foreign keys (source → target)
      - indexes (name, columns, unique flag)
      - check & unique constraints
    """
    conn = _get_conn(x_session_id)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # ── Tables + estimated row count ──────────────────────────────────────
    cur.execute("""
        SELECT
            t.table_name,
            obj_description(pc.oid, 'pg_class') AS table_comment,
            COALESCE(pc.reltuples::bigint, 0)   AS row_estimate
        FROM information_schema.tables t
        JOIN pg_class pc ON pc.relname = t.table_name
        WHERE t.table_schema = 'public'
          AND t.table_type   = 'BASE TABLE'
        ORDER BY t.table_name
    """)
    tables_raw = cur.fetchall()

    # ── Columns ───────────────────────────────────────────────────────────
    cur.execute("""
        SELECT
            c.table_name,
            c.column_name,
            c.data_type,
            c.udt_name,
            c.is_nullable,
            c.column_default,
            c.character_maximum_length,
            c.numeric_precision,
            c.numeric_scale,
            c.ordinal_position,
            -- primary key flag
            CASE WHEN pk.column_name IS NOT NULL THEN true ELSE false END AS is_primary_key
        FROM information_schema.columns c
        LEFT JOIN (
            SELECT ku.table_name, ku.column_name
            FROM information_schema.table_constraints tc
            JOIN information_schema.key_column_usage ku
              ON tc.constraint_name = ku.constraint_name
             AND tc.table_schema    = ku.table_schema
            WHERE tc.constraint_type = 'PRIMARY KEY'
              AND tc.table_schema    = 'public'
        ) pk ON pk.table_name = c.table_name AND pk.column_name = c.column_name
        WHERE c.table_schema = 'public'
        ORDER BY c.table_name, c.ordinal_position
    """)
    columns_raw = cur.fetchall()

    # ── Foreign keys ──────────────────────────────────────────────────────
    cur.execute("""
        SELECT
            tc.constraint_name,
            tc.table_name          AS source_table,
            kcu.column_name        AS source_column,
            ccu.table_name         AS target_table,
            ccu.column_name        AS target_column,
            rc.update_rule,
            rc.delete_rule
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON ccu.constraint_name = tc.constraint_name
         AND ccu.table_schema    = tc.table_schema
        JOIN information_schema.referential_constraints rc
          ON rc.constraint_name  = tc.constraint_name
         AND rc.constraint_schema = tc.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema    = 'public'
        ORDER BY source_table, source_column
    """)
    fk_raw = cur.fetchall()

    # ── Indexes ───────────────────────────────────────────────────────────
    cur.execute("""
        SELECT
            t.relname                                   AS table_name,
            i.relname                                   AS index_name,
            ix.indisunique                              AS is_unique,
            ix.indisprimary                             AS is_primary,
            array_to_string(
                ARRAY(
                    SELECT pg_get_indexdef(ix.indexrelid, k + 1, true)
                    FROM generate_subscripts(ix.indkey, 1) AS k
                    ORDER BY k
                ), ', '
            )                                           AS columns
        FROM pg_class t
        JOIN pg_index ix   ON t.oid       = ix.indrelid
        JOIN pg_class i    ON i.oid       = ix.indexrelid
        JOIN pg_namespace n ON n.oid      = t.relnamespace
        WHERE n.nspname = 'public'
          AND t.relkind = 'r'
        ORDER BY t.relname, i.relname
    """)
    indexes_raw = cur.fetchall()

    # ── Unique / Check constraints ─────────────────────────────────────────
    cur.execute("""
        SELECT
            tc.table_name,
            tc.constraint_name,
            tc.constraint_type,
            string_agg(kcu.column_name, ', ' ORDER BY kcu.ordinal_position) AS columns
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema    = kcu.table_schema
        WHERE tc.table_schema   = 'public'
          AND tc.constraint_type IN ('UNIQUE', 'CHECK')
        GROUP BY tc.table_name, tc.constraint_name, tc.constraint_type
        ORDER BY tc.table_name, tc.constraint_type
    """)
    constraints_raw = cur.fetchall()

    conn.close()

    # ── Assemble response ─────────────────────────────────────────────────
    cols_by_table: dict[str, list] = {}
    for c in columns_raw:
        cols_by_table.setdefault(c["table_name"], []).append({
            "name":        c["column_name"],
            "type":        c["udt_name"] if c["data_type"] == "USER-DEFINED" else c["data_type"],
            "nullable":    c["is_nullable"] == "YES",
            "default":     c["column_default"],
            "is_pk":       c["is_primary_key"],
            "max_length":  c["character_maximum_length"],
            "precision":   c["numeric_precision"],
            "scale":       c["numeric_scale"],
        })

    idx_by_table: dict[str, list] = {}
    for ix in indexes_raw:
        if ix["is_primary"]:
            continue
        idx_by_table.setdefault(ix["table_name"], []).append({
            "name":      ix["index_name"],
            "columns":   ix["columns"],
            "is_unique": ix["is_unique"],
        })

    con_by_table: dict[str, list] = {}
    for cn in constraints_raw:
        con_by_table.setdefault(cn["table_name"], []).append({
            "name":    cn["constraint_name"],
            "type":    cn["constraint_type"],
            "columns": cn["columns"],
        })

    tables = []
    for t in tables_raw:
        name = t["table_name"]
        tables.append({
            "name":         name,
            "comment":      t["table_comment"],
            "row_estimate": t["row_estimate"],
            "columns":      cols_by_table.get(name, []),
            "indexes":      idx_by_table.get(name, []),
            "constraints":  con_by_table.get(name, []),
        })

    foreign_keys = [
        {
            "constraint":    fk["constraint_name"],
            "source_table":  fk["source_table"],
            "source_column": fk["source_column"],
            "target_table":  fk["target_table"],
            "target_column": fk["target_column"],
            "on_update":     fk["update_rule"],
            "on_delete":     fk["delete_rule"],
        }
        for fk in fk_raw
    ]

    return {"tables": tables, "foreign_keys": foreign_keys}


@router.delete("/disconnect")
def disconnect(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Remove the session from the server-side registry."""
    SessionManager.delete_session(x_session_id)
    return {"status": "disconnected"}


@router.get("/ping")
def ping(x_session_id: str = Header(..., alias="X-Session-ID")):
    """Quick liveness check — returns DB server version."""
    conn = _get_conn(x_session_id)
    cur  = conn.cursor()
    cur.execute("SELECT version()")
    version = cur.fetchone()[0]
    conn.close()
    return {"status": "ok", "version": version}
#table data
@router.get("/table-data")
def get_table_data(
    table:  str,
    limit:  int = 100,
    offset: int = 0,
    search: str = "",
    sort:   str = "",
    dir:    str = "asc",
    x_session_id: str = Header(..., alias="X-Session-ID"),
):
    """
    Return paginated rows for a single table.
    - Validates the table name against the real schema (prevents injection).
    - Optional full-text search across all text columns.
    - Optional single-column sort.
    """
    conn = _get_conn(x_session_id)
    cur  = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Safety: only allow tables that actually exist in the public schema
    cur.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
    """)
    valid_tables = {r["table_name"] for r in cur.fetchall()}
    if table not in valid_tables:
        conn.close()
        raise HTTPException(status_code=400, detail=f"Unknown table: {table}")

    # Fetch column names + data types for this table
    cur.execute("""
        SELECT column_name, data_type, udt_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
    """, (table,))
    col_rows = cur.fetchall()
    columns  = [c["column_name"] for c in col_rows]
    types    = [c["udt_name"] if c["data_type"] == "USER-DEFINED" else c["data_type"]
                for c in col_rows]

    # Build a safe ORDER BY clause (column name whitelisted against schema)
    order_clause = ""
    if sort and sort in columns:
        safe_dir     = "DESC" if dir.lower() == "desc" else "ASC"
        order_clause = f'ORDER BY "{sort}" {safe_dir}'

    # Build optional search WHERE clause across text/varchar columns
    text_cols   = [c["column_name"] for c in col_rows
                   if c["data_type"] in ("character varying", "text", "varchar", "char", "name")]
    where_clause = ""
    params_list: list = []
    if search and text_cols:
        conditions   = " OR ".join([f'CAST("{c}" AS TEXT) ILIKE %s' for c in text_cols])
        where_clause = f"WHERE {conditions}"
        params_list  = [f"%{search}%" for _ in text_cols]

    # Total count (for pagination)
    cur.execute(f'SELECT COUNT(*) FROM "{table}" {where_clause}', params_list)
    total = cur.fetchone()["count"]

    # Paginated rows
    cur.execute(
        f'SELECT * FROM "{table}" {where_clause} {order_clause} LIMIT %s OFFSET %s',
        params_list + [limit, offset],
    )
    rows = cur.fetchall()
    conn.close()

    return {
        "columns": columns,
        "types":   types,
        "rows":    [list(r.values()) for r in rows],
        "total":   total,
    }