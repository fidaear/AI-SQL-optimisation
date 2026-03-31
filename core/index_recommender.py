import re
from utils.logger import get_logger

logger = get_logger(__name__)


def extract_table_aliases(sql: str) -> dict[str, str]:
    """Maps aliases to real table names. e.g. {'o': 'orders', 'c': 'customers'}"""
    alias_map = {}
    for match in re.finditer(r'(?:FROM|JOIN)\s+(\w+)\s+(\w+)', sql, re.IGNORECASE):
        table, alias = match.group(1), match.group(2)
        if alias.upper() not in ('ON', 'WHERE', 'SET', 'AS', 'INNER', 'LEFT', 'RIGHT'):
            alias_map[alias] = table
            alias_map[table] = table
    return alias_map


def extract_where_columns(sql: str) -> list[tuple[str, str]]:
    """Extracts (alias, column) pairs from the WHERE clause."""
    columns = []
    where_match = re.search(r'WHERE(.+?)(?:ORDER BY|GROUP BY|LIMIT|$)', sql, re.IGNORECASE | re.DOTALL)
    if where_match:
        where_body = where_match.group(1)
        for alias, col in re.findall(r'(\w+)\.(\w+)', where_body):
            columns.append((alias, col))
    return columns


def extract_join_columns(sql: str) -> list[tuple[str, str]]:
    """Extracts (alias, column) pairs from JOIN ON conditions."""
    columns = []
    for on_match in re.finditer(r'ON\s+(\w+)\.(\w+)\s*=\s*(\w+)\.(\w+)', sql, re.IGNORECASE):
        columns.append((on_match.group(1), on_match.group(2)))
        columns.append((on_match.group(3), on_match.group(4)))
    return columns


def generate_index_suggestions(sql: str) -> list[str]:
    """
    Main function — analyses the SQL and returns CREATE INDEX statements.
    Duplicates are automatically prevented using a seen set.
    """
    alias_map = extract_table_aliases(sql)
    all_columns = extract_where_columns(sql) + extract_join_columns(sql)

    suggestions = []
    seen = set()  # Prevents duplicate recommendations

    for alias, column in all_columns:
        real_table = alias_map.get(alias, alias)
        key = f"{real_table}.{column}"

        if key not in seen:
            seen.add(key)
            index_name = f"idx_{real_table}_{column}"
            statement = f"CREATE INDEX IF NOT EXISTS {index_name} ON {real_table}({column});"
            suggestions.append(statement)
            logger.info(f"Index suggestion: {statement}")

    return suggestions


if __name__ == "__main__":
    sql = """
        SELECT *
        FROM orders o
        JOIN customers c ON o.customer_id = c.id
        WHERE o.order_date >= '2025-01-01'
          AND c.region = 'EU';
    """
    results = generate_index_suggestions(sql)
    print(f"\n💡 {len(results)} Index Suggestion(s):\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r}")
