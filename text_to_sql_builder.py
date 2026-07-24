"""
ai/text_to_sql_builder.py  (TEMPORARY STUB)
--------------------------------------------
Placeholder so that core/orchestrator.py can be imported while the real
Text-to-SQL generation logic (Phase 3) is still being built.

Replace this with the real implementation once it's ready — nothing else
needs to change, since orchestrator.py already imports
`generate_sql_from_question` by name from this exact path.
"""

import logging

logger = logging.getLogger(__name__)


def generate_sql_from_question(
    question: str,
    schema_context: dict,
    relevant_tables: list[str] | None = None,
    **kwargs,
) -> str:
    """
    STUB — not yet implemented.

    Once Phase 3 is built, this should call the LLM (Mistral) with the
    question + relevant table schemas (likely produced by
    ai.table_selector.select_relevant_tables_basic) and return a raw SQL
    string.
    """
    logger.warning(
        "generate_sql_from_question() is a STUB — Text-to-SQL generation "
        "is not implemented yet. question=%r", question,
    )
    raise NotImplementedError(
        "generate_sql_from_question() has not been implemented yet "
        "(ai/text_to_sql_builder.py is a placeholder stub)."
    )