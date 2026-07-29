"""
index_recommender.py
====================
Task: Index Recommendation Engine
Branch: feature/index-recommendation-engine
Project: AI-Assisted SQL Query Optimization
"""

import re
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_table_aliases(sql: str) -> dict:
    alias_map = {}
    for match in re.finditer(
        r'(?:FROM|JOIN)\s+(\w+)\s+(?:AS\s+)?(\w+)',
        sql, re.IGNORECASE
    ):
        table, alias = match.group(1), match.group(2)
        if alias.upper() not in ('ON', 'WHERE', 'SET', 'INNER', 'LEFT', 'RIGHT', 'AS'):
            alias_map[alias] = table
            alias_map[table] = table
    return alias_map


def extract_where_columns(sql: str) -> list:
    columns = []
    where_match = re.search(
        r'WHERE(.+?)(?:ORDER\s+BY|GROUP\s+BY|LIMIT|HAVING|$)',
        sql, re.IGNORECASE | re.DOTALL
    )
    if where_match:
        for alias, col in re.findall(r'(\w+)\.(\w+)', where_match.group(1)):
            columns.append((alias, col))
    return columns


def extract_join_columns(sql: str) -> list:
    columns = []
    for match in re.finditer(
        r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)',
        sql, re.IGNORECASE
    ):
        columns.append((match.group(1), match.group(2)))
        columns.append((match.group(3), match.group(4)))
    return columns


def generate_index_suggestions(sql: str) -> list:
    """
    Given a SQL SELECT query, returns a list of CREATE INDEX statements.
    Deduplicates suggestions using a set.
    """
    alias_map   = extract_table_aliases(sql)
    all_columns = extract_where_columns(sql) + extract_join_columns(sql)

    suggestions = []
    seen = set()

    for alias, column in all_columns:
        real_table = alias_map.get(alias, alias)
        dedup_key  = f"{real_table}.{column}"

        if dedup_key not in seen:
            seen.add(dedup_key)
            index_name = f"idx_{real_table}_{column}"
            statement  = (
                f"CREATE INDEX IF NOT EXISTS {index_name} "
                f"ON {real_table}({column});"
            )
            suggestions.append(statement)
            logger.info(f"Index suggestion: {statement}")

    return suggestions