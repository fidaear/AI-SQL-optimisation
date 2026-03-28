"""
ai_optimizer.py — Uses the LLM to generate an optimized SQL query.
"""
from ai.prompt_builder import build_optimization_prompt
from ai.llm_client import call_llm
from utils.logger import get_logger

logger = get_logger(__name__)

def get_ai_suggestion(sql: str, issues: list, index_suggestions: list) -> str:
    """Returns AI-generated optimization suggestion as a string."""
    prompt = build_optimization_prompt(sql, issues, index_suggestions)
    logger.info("Sending query to AI optimizer.")
    return call_llm(prompt)
