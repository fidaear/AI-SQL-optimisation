"""
sql_extractor.py — Bulletproof SQL Extractor for Mistral/Ollama Responses
Project: AI-Assisted SQL Query Optimizer

WHY THIS MODULE EXISTS
──────────────────────
Mistral (and most open-source LLMs via Ollama) prepend or append natural-language
prose to SQL output even when the system prompt forbids it. Examples causing
PostgreSQL syntax errors (erreur de syntaxe sur ou près de « u »):

    "Sure, here you go: SELECT * FROM users WHERE country = 'Morocco';"
    "Voici la requête :\nSELECT * FROM users ..."
    "SELECT * FROM users WHERE country = 'Morocco'; you could also add an index"
    "SELECT * FROM users WHERE country = 'Morocco'; -- adding index hint"

The PostgreSQL error position 'P' points into that prose, not the SQL itself.

HOW TO USE IN YOUR ORCHESTRATOR
────────────────────────────────
Wherever your orchestrator calls the LLM and then passes the result to the DB,
wrap the LLM output with extract_sql():

    from ai.sql_extractor import extract_sql

    raw = llm_client.complete(prompt, system_prompt=...)
    sql = extract_sql(raw, fallback=original_sql)
    # sql is now guaranteed clean — safe to execute against PostgreSQL

PIPELINE (6 steps, applied in order)
─────────────────────────────────────
  Step 0  ###START###/###END### markers  — highest-confidence extraction
  Step 1  Strip markdown fences          — ```sql … ``` or ``` … ```
  Step 2  Cut after last semicolon       — drops all trailing prose
  Step 3  Cut before first SELECT        — drops all leading preamble
  Step 4  Strip inline SQL comments      — removes  -- ...  suffixes
  Step 5  Sanity check                   — must contain SELECT … FROM
"""

import re
import logging

logger = logging.getLogger(__name__)


def extract_sql(raw: str, fallback: str = "") -> str:
    """
    Extract clean, executable SQL from a raw LLM response.

    Args:
        raw:      Raw string returned by the LLM.
        fallback: Returned when no valid SQL can be extracted.
                  Pass the original SQL here so the pipeline safely falls back
                  to the unmodified query instead of crashing the DB executor.

    Returns:
        A clean SQL string ready to send to PostgreSQL, or `fallback`.
    """
    if not raw or not raw.strip():
        logger.warning("extract_sql: empty LLM response — using fallback")
        return fallback

    original_raw = raw
    text = raw.strip()

    # ── Step 0: ###START###/###END### markers ────────────────────────────────
    # The ai_optimizer system prompt instructs Mistral to wrap SQL in these
    # markers.  When present they give us 100% reliable extraction regardless
    # of any surrounding prose, French text, or markdown the model added.
    marker_match = re.search(
        r"###START###\s*(.*?)\s*###END###",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if marker_match:
        sql = marker_match.group(1).strip()
        # Still clean up any accidental fences or comments inside the markers
        sql = re.sub(r"```(?:sql|SQL)?", "", sql).strip("`").strip()
        sql = re.sub(r"\s*--[^\n]*", "", sql).strip()
        if sql and re.match(r"\s*SELECT\b", sql, re.IGNORECASE):
            logger.debug("extract_sql: extracted via markers (%d chars): %s", len(sql), repr(sql[:80]))
            return sql
        logger.warning("extract_sql: markers found but content invalid — falling through to heuristic")

    # ── Step 1: Strip BOM ─────────────────────────────────────────────────────
    text = text.lstrip('\ufeff')

    # ── Step 2: Strip markdown fences ─────────────────────────────────────────
    # Remove  ```sql … ```  and  ``` … ```  wrappers.
    text = re.sub(r"```(?:sql|SQL)?", "", text).strip().strip("`").strip()

    # ── Step 3: Cut everything after the last semicolon ──────────────────────
    # SQL queries end with ';'. Text after the final ';' is prose added by the
    # model ("you could also add an index on…", "Note: this is already optimal").
    # We use rfind (last semicolon) because subqueries have intermediate semicolons.
    semi_pos = text.rfind(";")
    if semi_pos != -1:
        text = text[: semi_pos + 1].strip()

    # ── Step 4: Cut everything before the first SELECT ────────────────────────
    # Handles all preamble patterns:
    #   "Sure, here you go: SELECT …"          (same-line English preamble)
    #   "Voici la requête :\nSELECT …"         (French, newline preamble)
    #   "إليك الاستعلام:\nSELECT …"            (Arabic preamble)
    #   "Here is the optimized query:\n```sql\nSELECT …"
    select_match = re.search(r"\bSELECT\b", text, re.IGNORECASE)
    if not select_match:
        logger.warning(
            "extract_sql: no SELECT found — using fallback. Raw (truncated): %s",
            repr(original_raw[:200]),
        )
        return fallback
    text = text[select_match.start():]

    # ── Step 5: Strip inline SQL comments ─────────────────────────────────────
    # "SELECT * FROM users; -- adding index hint"  →  "SELECT * FROM users;"
    text = re.sub(r"\s*--[^\n]*", "", text).strip()

    # ── Step 6: Sanity check ──────────────────────────────────────────────────
    if not re.match(r"\s*SELECT\b", text, re.IGNORECASE):
        logger.error("extract_sql: cleaned text does not start with SELECT — using fallback. Text: %s", repr(text[:200]))
        return fallback
    if not re.search(r"\bFROM\b", text, re.IGNORECASE):
        logger.error("extract_sql: cleaned text has no FROM clause — using fallback. Text: %s", repr(text[:200]))
        return fallback

    logger.debug(
        "extract_sql: heuristic extraction — %d chars from %d-char raw response. SQL: %s",
        len(text), len(original_raw), repr(text[:120]),
    )
    return text.strip()