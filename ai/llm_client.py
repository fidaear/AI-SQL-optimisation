"""
llm_client.py — Handles communication with the Claude LLM API.
"""
import anthropic
from utils.config import LLM_API_KEY
from utils.logger import get_logger

logger = get_logger(__name__)
client = anthropic.Anthropic(api_key=LLM_API_KEY)

def call_llm(prompt: str) -> str:
    """Sends a prompt to Claude and returns the text response."""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}]
        )
        response = message.content[0].text
        logger.info("LLM response received.")
        return response
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return "AI suggestion unavailable."
