import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# ── Optional fuzzy matching backend ─────────────────────────────────────────
try:
    from rapidfuzz import fuzz as _rapidfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False
    import difflib
    logger.info(
        "table_selector: rapidfuzz not installed — falling back to "
        "difflib.SequenceMatcher for fuzzy matching (less accurate, no extra dep)."
    )


# ── Scoring weights (tunable) ───────────────────────────────────────────────
SCORE_TABLE_EXACT      = 10.0   # question token == table name (singular/plural)
SCORE_TABLE_SUBSTRING  = 6.0    # table name appears as substring in question
SCORE_COLUMN_EXACT     = 3.0    # question token == a column name in this table
SCORE_COLUMN_SUBSTRING = 1.5    # column name appears as substring in question
SCORE_FK_BONUS         = 1.0    # table is FK-linked to another candidate table
FUZZY_MIN_RATIO        = 85     # 0-100 scale; below this, ignore fuzzy hits
SCORE_FUZZY_TABLE      = 4.0
SCORE_FUZZY_COLUMN     = 1.0

_STOPWORDS = {
    "the", "a", "an", "of", "for", "in", "on", "by", "with", "and", "or",
    "to", "is", "are", "was", "were", "show", "me", "get", "find", "list",
    "all", "give", "what", "which", "how", "many", "much", "top", "each",
    "per", "from", "that", "this", "who", "when", "where", "please",
}

_TOKEN_RE = re.compile(r"[a-zA-Z_][a-zA-Z0-9_]*")


def _tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed."""
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _singular(word: str) -> str:
    """Very small heuristic singularizer — good enough for table names."""
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("ses") and len(word) > 4:
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _plural(word: str) -> str:
    if word.endswith("y") and not word.endswith(("ay", "ey", "oy", "uy")):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "ch", "sh")):
        return word + "es"
    return word + "s"


def _fuzzy_ratio(a: str, b: str) -> float:
    """Returns a 0-100 similarity ratio using rapidfuzz or difflib."""
    if _HAS_RAPIDFUZZ:
        return _rapidfuzz.ratio(a, b)
    return difflib.SequenceMatcher(None, a, b).ratio() * 100


def _extract_column_names(table_data) -> list[str]:
    """
    Normalizes whatever shape a table's schema entry is in into a flat
    list of lowercase column name strings.

    Supports:
      - {"columns": ["id", "name", ...]}                     (list[str])
      - {"columns": [{"column_name": "id", ...}, ...]}        (list[dict])
      - {"columns": {"id": "integer", "name": "varchar"}}     (dict[name->type])
    """
    if not isinstance(table_data, dict):
        return []
    raw_cols = table_data.get("columns", [])

    if isinstance(raw_cols, dict):
        return [str(c).lower() for c in raw_cols.keys()]

    if isinstance(raw_cols, list):
        names = []
        for c in raw_cols:
            if isinstance(c, str):
                names.append(c.lower())
            elif isinstance(c, dict):
                name = c.get("column_name") or c.get("name")
                if name:
                    names.append(str(name).lower())
        return names

    return []


def _extract_fk_targets(table_data) -> set[str]:
    """Returns the lowercase set of tables this table has an FK relationship with."""
    if not isinstance(table_data, dict):
        return set()
    fks = table_data.get("foreign_keys", [])
    targets = set()
    for fk in fks:
        if isinstance(fk, dict):
            ref = fk.get("ref_table") or fk.get("references")
            if ref:
                targets.add(str(ref).lower())
    return targets


def select_relevant_tables_basic(
    question: str,
    schema_context: dict,
    limit: int = 5,
) -> list[str]:
    """
    Select the most likely relevant tables for a natural-language question,
    using exact/substring/fuzzy name matching against table & column names.

    Args:
        question:        The user's natural-language question.
        schema_context:  {table_name: {"columns": [...], "foreign_keys": [...]}}
                          Table names are the ONLY valid output values —
                          nothing outside this dict's keys is ever returned.
        limit:           Max number of tables to return (default 5).

    Returns:
        A list of table names (subset of schema_context.keys()), ordered by
        descending relevance score. Empty list if nothing scored > 0
        (a warning is logged in that case — callers should surface this to
        the user rather than silently querying every table).
    """
    if not question or not question.strip():
        logger.warning("select_relevant_tables_basic: empty question — returning []")
        return []

    if not schema_context:
        logger.warning("select_relevant_tables_basic: empty schema_context — returning []")
        return []

    valid_tables = list(schema_context.keys())
    tokens = _tokenize(question)

    if not tokens:
        logger.warning(
            "select_relevant_tables_basic: question '%s' produced no usable "
            "tokens after stopword removal — returning []", question,
        )
        return []

    question_lower = question.lower()
    scores: dict[str, float] = {t: 0.0 for t in valid_tables}
    match_log: dict[str, list[str]] = {t: [] for t in valid_tables}

    # ── Pass 1: table-name matching (exact / substring / fuzzy) ────────────
    for table in valid_tables:
        table_lower = table.lower()
        table_singular = _singular(table_lower)
        table_plural = _plural(table_singular)
        name_variants = {table_lower, table_singular, table_plural}

        # Exact token match against any name variant
        if any(tok in name_variants for tok in tokens):
            scores[table] += SCORE_TABLE_EXACT
            match_log[table].append("exact table-name token match")
        # Substring match: table name (or singular) appears in the raw question
        elif table_singular in question_lower or table_lower in question_lower:
            scores[table] += SCORE_TABLE_SUBSTRING
            match_log[table].append("table-name substring match")
        else:
            # Fuzzy fallback only if nothing exact/substring matched
            best_ratio = max(
                (_fuzzy_ratio(tok, table_singular) for tok in tokens),
                default=0.0,
            )
            if best_ratio >= FUZZY_MIN_RATIO:
                scores[table] += SCORE_FUZZY_TABLE
                match_log[table].append(f"fuzzy table-name match ({best_ratio:.0f}%)")

    # ── Pass 2: column-name matching ────────────────────────────────────────
    for table in valid_tables:
        columns = _extract_column_names(schema_context.get(table, {}))
        if not columns:
            continue

        for col in columns:
            col_singular = _singular(col)
            if col in tokens or col_singular in tokens:
                scores[table] += SCORE_COLUMN_EXACT
                match_log[table].append(f"exact column match ({col})")
            elif col in question_lower:
                scores[table] += SCORE_COLUMN_SUBSTRING
                match_log[table].append(f"column substring match ({col})")
            else:
                best_ratio = max(
                    (_fuzzy_ratio(tok, col) for tok in tokens),
                    default=0.0,
                )
                if best_ratio >= FUZZY_MIN_RATIO:
                    scores[table] += SCORE_FUZZY_COLUMN
                    match_log[table].append(f"fuzzy column match ({col}, {best_ratio:.0f}%)")

    # ── Pass 3: FK-relationship bonus between already-scored candidates ────
    pre_fk_candidates = {t for t, s in scores.items() if s > 0}
    for table in pre_fk_candidates:
        fk_targets = _extract_fk_targets(schema_context.get(table, {}))
        linked = fk_targets & pre_fk_candidates
        if linked:
            scores[table] += SCORE_FK_BONUS
            match_log[table].append(f"FK bonus (linked to {sorted(linked)})")

    # ── Rank & filter ────────────────────────────────────────────────────────
    ranked = sorted(
        ((t, s) for t, s in scores.items() if s > 0),
        key=lambda x: x[1],
        reverse=True,
    )

    if not ranked:
        logger.warning(
            "select_relevant_tables_basic: no table/column matches found for "
            "question=%r across %d known tables (%s) — returning empty list. "
            "Caller should fall back to asking the user to clarify, or pass "
            "the full schema if the pipeline can tolerate it.",
            question, len(valid_tables), valid_tables,
        )
        return []

    top_tables = [t for t, _ in ranked[:limit]]

    logger.info(
        "select_relevant_tables_basic: question=%r -> %s",
        question, [(t, round(s, 1)) for t, s in ranked[:limit]],
    )
    for t in top_tables:
        logger.debug("  '%s' matched via: %s", t, match_log[t])

    # Safety: guarantee we never return anything not in schema_context
    assert all(t in valid_tables for t in top_tables), (
        "select_relevant_tables_basic produced a table not in schema_context — bug"
    )

    return top_tables