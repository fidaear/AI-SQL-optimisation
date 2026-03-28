"""
prompt_builder.py — Builds structured prompts for the LLM.
"""

def build_optimization_prompt(sql: str, issues: list, index_suggestions: list) -> str:
    issues_text = "\n".join([f"- [{i['severity']}] {i['rule']}: {i['advice']}" for i in issues])
    indexes_text = "\n".join(index_suggestions)

    return f"""You are a PostgreSQL expert. Analyze this SQL query and suggest an optimized version.

SQL QUERY:
{sql}

DETECTED ISSUES:
{issues_text if issues_text else "None"}

INDEX SUGGESTIONS:
{indexes_text if indexes_text else "None"}

Please provide:
1. An optimized version of the SQL query
2. A brief explanation of what you changed and why
3. Expected performance improvement

Be concise and practical.
"""
