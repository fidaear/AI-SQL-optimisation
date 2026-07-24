"""
ai/text_to_sql_service.py
--------------------------
Text-to-SQL Task: LLM API Integration

Responsibility (and ONLY this — schema formatting / table selection /
safety validation live elsewhere, e.g. ai/text_to_sql_builder.py and
ai/ai_optimizer.py's SQLSafetyValidator):

    1. Build the prompt from (question, schema_prompt)
    2. Call the company-hosted LLM via the shared LLMClient
    3. Strip markdown code fences / stray prose from the response
    4. Return a plain SQL string

This module never talks to OpenAI/Groq/any public cloud — LLMClient is
constructed with zero arguments here, so it always resolves its config
from LLM_BASE_URL / LLM_API_KEY / LLM_MODEL in the environment (see
ai/llm_client.py). If those env vars point somewhere else, that's a
deployment config issue, not something this module should special-case.
"""

import re
import logging
from typing import Optional

from ai.llm_client import LLMClient, LLMClientError, LLMCallType

logger = logging.getLogger(__name__)


class TextToSQLServiceError(Exception):
    """Raised when SQL generation fails (LLM error, empty/unusable output)."""


SYSTEM_PROMPT = """\
LANGUAGE: Respond in ENGLISH only.

ROLE: You are a senior PostgreSQL developer.
TASK: Translate the user's natural-language question into a single, valid
PostgreSQL SELECT statement, using ONLY the tables and columns given in the
DATABASE SCHEMA section of the prompt.

RULES:
1. Output ONLY the SQL statement between ###START### and ###END### markers.
   No explanations, no greetings, no markdown fences, no comments.
2. Use ONLY table and column names that literally appear in the DATABASE
   SCHEMA section. Never invent a column or table name, even if it seems
   like an obvious/expected name for this kind of schema.
3. The statement must be a single SELECT (or a CTE ending in SELECT).
   Never generate INSERT / UPDATE / DELETE / DDL of any kind.
4. If the question cannot be answered with the given schema, output:
   ###START###
   ###END###
   (an empty block) rather than guessing at tables/columns that don't exist.

OUTPUT FORMAT — follow EXACTLY:
###START###
<single valid PostgreSQL SELECT statement>
###END###
"""

_START_MARKER = "###START###"
_END_MARKER   = "###END###"


def _build_prompt(question: str, schema_prompt: str) -> str:
    parts = []
    if schema_prompt and schema_prompt.strip():
        parts.append(schema_prompt.strip())
        parts.append("")
    parts.append("QUESTION:")
    parts.append(question.strip())
    return "\n".join(parts)


def _clean_sql_output(raw: str) -> str:
    """
    Strip markers, markdown code fences, and stray prose from the raw LLM
    response, returning just the SQL text.

    Handles, in order:
      1. ###START### / ###END### markers, if present (preferred path)
      2. ```sql ... ``` or ``` ... ``` fences, if the model ignored the
         marker format and used markdown instead
      3. Plain trimming as a last resort
    """
    if not raw:
        return ""

    text = raw.strip()

    # 1. Prefer content between explicit markers
    if _START_MARKER in text:
        start = text.index(_START_MARKER) + len(_START_MARKER)
        end = text.index(_END_MARKER, start) if _END_MARKER in text else len(text)
        text = text[start:end].strip()

    # 2. Strip markdown code fences (```sql ... ``` or ``` ... ```)
    text = re.sub(r"```sql\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```\s*", "", text)

    # 3. Drop any leading label like "SQL:" the model sometimes adds
    text = re.sub(r"^\s*SQL\s*:\s*", "", text, flags=re.IGNORECASE)

    return text.strip().rstrip(";").strip()


def generate_sql(
    question: str,
    schema_prompt: str,
    llm_client: Optional[LLMClient] = None,
) -> str:
    """
    Generate a SQL query from a natural-language question.

    Args:
        question:      The user's natural-language question.
        schema_prompt:  Pre-formatted schema context (table/column info) to
                        ground the LLM — typically produced by
                        ai.text_to_sql_builder.format_schema_for_prompt().
        llm_client:     Optional injected LLMClient (mainly for tests).
                        If omitted, a client is constructed from env vars
                        on every call — cheap, since LLMClient itself holds
                        no per-request state.

    Returns:
        A plain SQL string (no markdown fences, no ###START###/###END### markers).

    Raises:
        TextToSQLServiceError: on empty input, LLM failure/timeout, or an
                                empty/unusable response from the model.
    """
    if not question or not question.strip():
        raise TextToSQLServiceError("question must not be empty")

    try:
        client = llm_client or LLMClient()
    except LLMClientError as exc:
        # Missing/invalid config (e.g. LLM_API_KEY not set) — surface as a
        # service-level error so callers don't need to know about LLMClientError.
        logger.error("Text-to-SQL: could not initialise LLMClient: %s", exc)
        raise TextToSQLServiceError(f"LLM client configuration error: {exc}") from exc

    prompt = _build_prompt(question, schema_prompt)

    try:
        raw_response = client.complete(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPT,
            call_type=LLMCallType.TEXT_TO_SQL,
        )
    except LLMClientError as exc:
        # Covers timeouts, connection errors, rate limits after retries,
        # and non-retryable 4xx/5xx — all funneled through LLMClientError.
        logger.error("Text-to-SQL: LLM call failed for question %r: %s", question, exc)
        raise TextToSQLServiceError(f"LLM call failed: {exc}") from exc

    sql = _clean_sql_output(raw_response)

    if not sql:
        logger.warning(
            "Text-to-SQL: LLM returned empty/unusable output for question %r "
            "(raw response, 300 chars): %r",
            question, raw_response[:300],
        )
        raise TextToSQLServiceError(
            "LLM returned an empty or unparseable SQL response."
        )

    logger.info("Text-to-SQL: generated SQL (%d chars) for question %r", len(sql), question)
    return sql