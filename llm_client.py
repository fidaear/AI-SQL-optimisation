"""
llm_client.py — LLM API Client with Retry Logic
Task 4: LLM Integration & Validation
Project: AI-Assisted SQL Query Optimizer

Changes in this version:
  - base_url / model / api_key are now read from environment variables.
    NOTHING is hardcoded anymore — the previous version shipped a live
    company API key in source control, which is what this patch removes.
  - api_key has NO fallback default. If LLM_API_KEY is not set, the client
    raises LLMClientError at construction time with a clear message,
    instead of silently sending requests with a bad/missing key.
  - Added LLMCallType.TEXT_TO_SQL — a dedicated token budget for the
    Text-to-SQL generation call (single SELECT statement out, no prose).
  - Everything else (retry/backoff, token budgets, logging) is unchanged
    from the version already in use by ai_optimizer.py.

Required environment variables:
  LLM_API_KEY   (required, no default — client refuses to start without it)
  LLM_BASE_URL  (optional, default = company ministral endpoint)
  LLM_MODEL     (optional, default = "ministral-3:8b")
"""

import os
import time
import logging
from enum import Enum
from typing import Optional
from openai import OpenAI, APIConnectionError, APIStatusError, RateLimitError

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Call type profiles
# ──────────────────────────────────────────────

class LLMCallType(Enum):
    """
    Different call types need different token budgets:

    SQL           — general query-rewrite, needs headroom for complex
                    multi-join queries. 800 tokens.
    TEXT_TO_SQL   — natural-language question -> single SELECT statement.
                    Output is always short and strictly formatted, so a
                    smaller budget keeps latency down. 400 tokens.
    EXPLAIN       — free-form explanation, 2-4 sentences. 200 tokens.
    GENERIC       — default for anything else.
    """
    SQL          = "sql"
    TEXT_TO_SQL  = "text_to_sql"
    EXPLAIN      = "explain"
    GENERIC      = "generic"


# Token budget per call type
_TOKEN_BUDGETS: dict[LLMCallType, int] = {
    LLMCallType.SQL:         800,
    LLMCallType.TEXT_TO_SQL: 400,
    LLMCallType.EXPLAIN:     200,
    LLMCallType.GENERIC:     600,
}

# Env-var driven defaults. Only the endpoint/model have safe fallbacks —
# they're not secrets. The API key deliberately has NO fallback.
_DEFAULT_BASE_URL = os.environ.get(
    "LLM_BASE_URL", "http://102.54.244.89:8088/ollama/api/v1"
)
_DEFAULT_MODEL = os.environ.get("LLM_MODEL", "ministral-3:8b")


# ──────────────────────────────────────────────
# LLMClient
# ──────────────────────────────────────────────

class LLMClient:
    """
    Thin wrapper around the OpenAI-compatible API, pointed at the
    company-hosted LLM server (never api.openai.com / api.groq.com etc.
    unless you explicitly override LLM_BASE_URL — the whole point of
    reading it from env is that it's swappable per-deployment, not that
    it defaults to a public cloud).

    Responsibilities:
      - Hold connection config (base_url, api_key, model) — sourced from env
      - Provide a single `complete(prompt, call_type)` method
      - Use the right token budget for each call type
      - Implement exponential-backoff retry logic
      - Surface clean errors to callers
    """

    DEFAULT_TIMEOUT     = 60
    MAX_RETRIES         = 3
    BACKOFF_BASE        = 1.5   # seconds — 1.5 / 2.25 / 3.375 between retries
    DEFAULT_TEMPERATURE = 0.0   # deterministic output, no creative prose mixed with SQL

    def __init__(
        self,
        base_url:    Optional[str]   = None,
        api_key:     Optional[str]   = None,
        model:       Optional[str]   = None,
        max_retries: int             = MAX_RETRIES,
        timeout:     int             = DEFAULT_TIMEOUT,
        temperature: float           = DEFAULT_TEMPERATURE,
    ):
        base_url = base_url or _DEFAULT_BASE_URL
        model    = model or _DEFAULT_MODEL
        api_key  = api_key or os.environ.get("LLM_API_KEY")

        if not api_key:
            # Fail loudly and immediately — do NOT let this fall through
            # to a request that silently 401s deep inside a pipeline stage.
            raise LLMClientError(
                "LLM_API_KEY is not set. Refusing to start LLMClient without "
                "an API key. Set it in your environment / .env file — it "
                "must never be hardcoded in source."
            )

        self.model       = model
        self.max_retries = max_retries
        self.timeout     = timeout
        self.temperature = temperature

        self._client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        logger.info(
            "LLMClient initialised — model=%s  base_url=%s  temperature=%.1f",
            model, base_url, temperature,
        )

    # ──────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────

    def complete(
        self,
        prompt:        str,
        system_prompt: Optional[str]  = None,
        call_type:     LLMCallType    = LLMCallType.GENERIC,
    ) -> str:
        """
        Send a prompt to the LLM and return the text response.

        Raises:
            LLMClientError: After all retries are exhausted, or on a
                             non-retryable API error (4xx other than 429).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        max_tokens = _TOKEN_BUDGETS[call_type]
        last_error: Exception = Exception("Unknown error")

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "LLM request — attempt %d/%d  call_type=%s  max_tokens=%d",
                    attempt, self.max_retries, call_type.value, max_tokens,
                )
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    timeout=self.timeout,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content or ""

                logger.info(
                    "LLM response [%s] (%d chars): %r",
                    call_type.value, len(content), content[:400],
                )
                return content

            except RateLimitError as exc:
                wait = self.BACKOFF_BASE ** attempt
                logger.warning("Rate limit — waiting %.1fs before retry %d", wait, attempt)
                last_error = exc
                time.sleep(wait)

            except APIConnectionError as exc:
                wait = self.BACKOFF_BASE ** attempt
                logger.warning(
                    "Connection error — waiting %.1fs before retry %d: %s",
                    wait, attempt, exc,
                )
                last_error = exc
                time.sleep(wait)

            except APIStatusError as exc:
                if exc.status_code >= 500:
                    wait = self.BACKOFF_BASE ** attempt
                    logger.warning(
                        "Server error %d — waiting %.1fs before retry %d",
                        exc.status_code, wait, attempt,
                    )
                    last_error = exc
                    time.sleep(wait)
                else:
                    logger.error("Non-retryable API error %d: %s", exc.status_code, exc)
                    raise LLMClientError(f"API error {exc.status_code}: {exc}") from exc

            except Exception as exc:
                logger.error("Unexpected LLM error: %s", exc)
                raise LLMClientError(f"Unexpected error: {exc}") from exc

        raise LLMClientError(
            f"LLM request failed after {self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def ping(self) -> bool:
        """Quick connectivity check. Returns True if the API responds."""
        try:
            self.complete("Reply with the word OK only.", call_type=LLMCallType.GENERIC)
            return True
        except LLMClientError:
            return False


# ──────────────────────────────────────────────
# Custom exception
# ──────────────────────────────────────────────

class LLMClientError(Exception):
    """Raised when the LLM API fails after all retries, or config is invalid."""