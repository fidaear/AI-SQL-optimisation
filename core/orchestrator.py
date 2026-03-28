"""
orchestrator.py — Coordinates the full optimization pipeline.
Ties together: validator → sanitizer → executor → plan_parser
              → rule_engine → index_recommender → ai_optimizer → ab_tester
"""
from security.query_sanitizer import sanitize
from core.validator import validate_sql
from core.executor import run_explain
from core.plan_parser import parse_plan
from core.rule_engine import analyze
from core.index_recommender import generate_index_suggestions
from ai.ai_optimizer import get_ai_suggestion
from utils.logger import get_logger

logger = get_logger(__name__)

def run_pipeline(sql: str) -> dict:
    """
    Full pipeline: takes a raw SQL query, returns all analysis results.
    """
    # 1. Sanitize
    safe, msg = sanitize(sql)
    if not safe:
        return {"error": msg}

    # 2. Validate
    valid, msg = validate_sql(sql)
    if not valid:
        return {"error": msg}

    # 3. Execute EXPLAIN
    plan_rows = run_explain(sql)
    plan = parse_plan(plan_rows)

    # 4. Rule-based analysis
    issues = analyze(sql)

    # 5. Index recommendations  ← YOUR MODULE
    index_suggestions = generate_index_suggestions(sql)

    # 6. AI suggestions
    ai_suggestion = get_ai_suggestion(sql, issues, index_suggestions)

    logger.info("Pipeline completed successfully.")
    return {
        "plan": plan,
        "issues": issues,
        "index_suggestions": index_suggestions,
        "ai_suggestion": ai_suggestion,
    }
