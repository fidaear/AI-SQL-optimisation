"""
Prompt Builder
Constructs prompts for LLMs to analyze SQL queries and suggest optimizations.
"""
"""
Prompt Builder for SQL Query Optimization
Constructs structured prompts for LLMs to analyze SQL queries and suggest optimizations.
Ensures SELECT-only safety and includes relevant database metrics.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class OptimizationType(Enum):
    """Types of optimization suggestions."""
    INDEX_RECOMMENDATION = "index_recommendation"
    QUERY_REWRITE = "query_rewrite"
    JOIN_OPTIMIZATION = "join_optimization"
    SUBQUERY_OPTIMIZATION = "subquery_optimization"
    AGGREGATION_OPTIMIZATION = "aggregation_optimization"
    GENERAL_ANALYSIS = "general_analysis"


@dataclass
class QueryMetrics:
    """Metrics from database analysis."""
    execution_time_ms: float
    rows_scanned: int
    rows_returned: int
    estimated_cost: float
    table_sizes: Dict[str, int]  # table_name -> row_count
    index_exists: List[str]
    missing_indexes: List[str]
    query_complexity: str  # simple, moderate, complex
    join_count: int
    subquery_count: int


@dataclass
class PromptContext:
    """Full context for prompt generation."""
    original_query: str
    tables_involved: List[str]
    columns_involved: List[str]
    join_types: List[str]
    metrics: QueryMetrics
    optimization_type: OptimizationType
    # Compact schema text produced by ai.schema_formatter.SchemaFormatter.
    # Optional so existing callers that don't pass it keep working unchanged.
    # When set, it is inserted into the prompt so the LLM knows real column
    # names / types / FKs instead of guessing from the query text alone.
    schema_context: Optional[str] = None


class PromptBuilder:
    """
    Builds structured prompts for LLM-based SQL optimization.
    Enforces SELECT-only safety and includes structured metrics.
    """
    
    # Safety rules - NEVER allow these operations
    FORBIDDEN_OPERATIONS = {
        'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER',
        'TRUNCATE', 'GRANT', 'REVOKE', 'MERGE', 'CALL', 'COPY',
        'VACUUM', 'ANALYZE', 'EXPLAIN'  # EXPLAIN is for analysis only
    }
    
    # Performance thresholds for recommendations (PostgreSQL 18.2)
    # These are baselines; actual thresholds depend on workload context
    THRESHOLDS = {
        'slow_query_ms': 1000,           # Interactive queries should be <1s (adjust for batch jobs)
        'moderate_query_ms': 5000,       # Batch/reporting queries acceptable <5s
        'high_row_scan': 100000,         # Full table scans > 100k rows suggest index opportunity
        'index_recommendation_threshold': 10000,  # Tables with >10k rows may benefit from indexes
        'selectivity_threshold': 0.10,   # Index selectivity should be >10% to be useful
        'join_selectivity_threshold': 0.05,  # Join selectivity <5% suggests hash join benefit
    }
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize prompt builder."""
        self.model = model
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def validate_query_safety(self, query: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that query is SELECT-only (safe).
        
        Returns:
            (is_safe: bool, error_message: str or None)
        """
        query_upper = query.strip().upper()
        
        # Must start with SELECT
        if not query_upper.startswith('SELECT'):
            return False, "Query must start with SELECT"
        
        # Check for forbidden operations
        for operation in self.FORBIDDEN_OPERATIONS:
            if f' {operation} ' in f' {query_upper} ':
                return False, f"Operation '{operation}' is forbidden. Only SELECT is allowed."
        
        # Check for common injection patterns
        if '--' in query or '/*' in query:
            return False, "SQL comments detected. This may indicate injection attempt."
        
        # Check for multiple statements
        if ';' in query.rstrip(';'):  # Multiple ; = multiple statements
            return False, "Multiple statements not allowed. Submit queries one at a time."
        
        return True, None
    
    def build_analysis_prompt(self, context: PromptContext) -> str:
        """
        Build a structured prompt for query analysis and optimization suggestions.
        
        Includes:
        - Original query
        - Database metrics
        - Performance characteristics
        - Optimization rules
        - Chain-of-thought reasoning
        - Concrete examples
        """
        metrics = context.metrics
        
        prompt = f"""You are an expert SQL optimization specialist using PostgreSQL 18.2. Analyze this query and provide optimization suggestions.

You will think step-by-step before making recommendations.

## SYSTEM CONTEXT
- Database Engine: PostgreSQL 18.2
- Query Dialect: PostgreSQL standard SQL
- Available Features: Window functions, CTEs, EXPLAIN ANALYZE, JSON operators

## SAFETY CONSTRAINT
⚠️ CRITICAL: Only provide SELECT-based optimization suggestions.
DO NOT suggest INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, or any DDL operations.
All suggested changes must be read-only and reversible.

## QUERY TO OPTIMIZE
```sql
{context.original_query}
```

{self._format_schema_section(context.schema_context)}
## DATABASE CONTEXT & METRICS
**Scope:**
- Tables: {', '.join(context.tables_involved) if context.tables_involved else 'N/A'}
- Columns Referenced: {len(context.columns_involved)}
- Join Complexity: {context.metrics.join_count} joins, {context.metrics.subquery_count} subqueries

**Performance Telemetry:**
- Execution Time: {metrics.execution_time_ms:.2f}ms (baseline)
- Rows Accessed: {metrics.rows_scanned:,}
- Rows Returned: {metrics.rows_returned:,}
- Query Selectivity: {(metrics.rows_returned / max(metrics.rows_scanned, 1)) * 100:.2f}% (lower = better index candidate)
- Estimated Cost Units: {metrics.estimated_cost:.2f}
- Query Complexity Classification: {metrics.query_complexity}

**Table Statistics:**
{self._format_table_stats(metrics.table_sizes)}

**Current Index Coverage:**
- Active Indexes: {', '.join(metrics.index_exists) if metrics.index_exists else 'None (table scan only)'}
- Recommended Indexes: {', '.join(metrics.missing_indexes) if metrics.missing_indexes else 'None needed'}

## OPTIMIZATION PRINCIPLES (PostgreSQL 18.2)
### Performance Rules
1. **Index Strategy:** Add indexes on high-selectivity columns (>10%) used in WHERE/JOIN
2. **Join Optimization:** Order joins by selectivity (most restrictive first)
3. **Subquery Strategy:** Move dependent subqueries to CTEs for better execution
4. **Aggregation Strategy:** Apply WHERE filters early, then GROUP BY
5. **Column Projection:** Avoid SELECT *. Include only needed columns

### Trade-Off Analysis
- **Index Cost:** ~10-15% write penalty, ~50-200MB storage per 10M rows
- **OLTP vs Batch:** OLTP needs more indexes (read-heavy). Batch jobs less so (write-heavy)

## OPTIMIZATION EXAMPLES (DO THIS)

**Example 1: Missing Index**
PROBLEM: SELECT * FROM users WHERE registration_date > '2023-01-01' (50M rows, 2000ms)
SOLUTION: CREATE INDEX idx_users_reg ON users(registration_date);
RESULT: 2000ms → 50ms (98% faster)

**Example 2: SELECT * to Specific Columns**
PROBLEM: SELECT * FROM products WHERE category = 'Books' (50 columns, slow)
SOLUTION: SELECT id, title, price FROM products WHERE category = 'Books' (3 columns)
RESULT: 20% faster I/O

**Example 3: JOIN Order**
PROBLEM: orders JOIN customers JOIN payments (wrong order = 5000ms)
SOLUTION: payments JOIN orders JOIN customers (filters early = 500ms)
RESULT: 10x faster

## ANALYSIS WORKFLOW
Before suggesting optimizations:
1. **Diagnose:** Is this slow due to [missing index | wrong join order | full scan | SELECT * ]?
2. **Estimate:** How much faster will this be? (30%? 90%?)
3. **Rank:** Most impactful suggestions first
4. **Trade-off:** Is it worth the storage/write cost?

## REQUIRED RESPONSE FORMAT

**DIAGNOSIS:**
[1-2 sentences: root cause. Example: "Full table scan on 50M rows due to missing index"]

**ROOT CAUSE ANALYSIS:**
- Primary Issue: [missing index | wrong join order | SELECT * | full scan]
- Evidence: [cite metric: "Selectivity 0.1%, should be >5% for hash join"]

**RECOMMENDATIONS:**

1. **[Optimization Type]** - [Specific action]
   - Target: [what improves]
   - Impact: [X% speedup from Yms → ~Zms]
   - Effort: [Low | Medium | High]
   - Trade-off: [storage | write penalty]

**OPTIMIZED QUERY:**
Output the rewritten SQL below, with NO markdown fences, NO inline comments, NO explanation text.
The SQL must be executable as-is. If no rewrite is needed, output the original query unchanged.

**SAFETY VERIFICATION:** ✓ SELECT-only (no INSERT/UPDATE/DELETE)

## HARD CONSTRAINTS
- NEVER suggest INSERT, UPDATE, DELETE, CREATE, ALTER, DROP, TRUNCATE
-  NEVER remove WHERE clauses
-  NEVER use dynamic SQL
-  DO suggest: Indexes, WHERE optimization, JOIN reordering, CTEs
"""
        return prompt
    
    def build_index_recommendation_prompt(self, context: PromptContext) -> str:
        """Build prompt specifically for index recommendations (PostgreSQL 18.2)."""
        metrics = context.metrics
        selectivity = (metrics.rows_returned / max(metrics.rows_scanned, 1)) * 100
        
        prompt = f"""You are a PostgreSQL 18.2 database indexing specialist.
Recommend indexes to improve this query ONLY if selectivity is low (<10%) or table is large (>10k rows).

## QUERY
```sql
{context.original_query}
```

## METRICS
- Rows Scanned: {metrics.rows_scanned:,}
- Rows Returned: {metrics.rows_returned:,}
- Selectivity: {selectivity:.2f}% (if <10%, index will help)
- Execution Time: {metrics.execution_time_ms:.0f}ms
- Tables: {', '.join(context.tables_involved)}

## RULE
Recommend index ONLY if:
- Selectivity < 10%, OR
- Table size > 10,000 rows
Otherwise: No index needed

## EXAMPLE (DO THIS)
QUERY: SELECT * FROM orders WHERE order_date > '2023-01-01' (100M rows, 0.5% selectivity)
RECOMMENDATION: CREATE INDEX idx_orders_date ON orders(order_date);
RESULT: 5000ms → 50ms (100x faster)

## RESPONSE FORMAT
IF you recommend indexes:

**Index 1: [Column(s)]**
```sql
CREATE INDEX idx_name ON table_name(column1);
```
- Why: [Which WHERE/JOIN column]
- Speedup: [estimate %]
- Cost: [storage, write penalty]

IF no index needed:
"No index needed: selectivity {selectivity:.1f}% > 10% threshold"

## CONSTRAINTS
- No INSERT/UPDATE/DELETE
- Only indexes on SELECT columns
- PostgreSQL 18.2 syntax
"""
        return prompt
        return prompt
    
    def build_join_optimization_prompt(self, context: PromptContext) -> str:
        """Build prompt for join optimization (PostgreSQL 18.2)."""
        if context.metrics.join_count == 0:
            return "No joins detected in query. JOIN optimization not applicable."
        
        metrics = context.metrics
        selectivity = (metrics.rows_returned / max(metrics.rows_scanned, 1)) * 100
        
        prompt = f"""You are a PostgreSQL 18.2 query optimization expert.
Optimize the JOIN order for this query.

## QUERY
```sql
{context.original_query}
```

## CONTEXT
- Joins: {context.metrics.join_count}
- Current Time: {metrics.execution_time_ms:.0f}ms
- Rows Scanned: {metrics.rows_scanned:,}
- Selectivity: {selectivity:.2f}%

**Table Sizes:**
{self._format_table_stats(metrics.table_sizes)}

## RULE
Optimal JOIN order = START WITH SMALLEST RESULT SET
Filter most restrictive table first, then join smaller results


## RESPONSE FORMAT

**ANALYSIS:**
Is join order optimal? [Current order vs better order]

**RECOMMENDATION:**
If optimization found:
```sql
-- Reordered query
[Your optimized query]
```
Reason: Start with [filtered table] to reduce rows flowing through joins

If already optimal:
"Join order is optimal: filters applied in correct sequence"

## CONSTRAINTS
- Only reorder JOINs, don't change WHERE
- No INSERT/UPDATE/DELETE
- PostgreSQL 18.2 syntax
"""
        return prompt
    
    @staticmethod
    def _format_schema_section(schema_context: Optional[str]) -> str:
        """
        Render the compact schema block (from SchemaFormatter) as its own
        prompt section. Returns "" (no section at all) when no schema
        context was provided, so existing callers that don't pass
        schema_context see zero change in prompt output.
        """
        if not schema_context:
            return ""
        return f"## SCHEMA\n{schema_context}\n\n"

    @staticmethod
    def _format_table_stats(table_sizes: Dict[str, int]) -> str:
        """Format table statistics nicely."""
        if not table_sizes:
            return "No table statistics available"
        
        lines = []
        for table, size in sorted(table_sizes.items(), key=lambda x: x[1], reverse=True):
            lines.append(f"  - {table}: {size:,} rows")
        return "\n".join(lines)
    
    def build_prompt(self, context: PromptContext) -> str:
        """
        Build appropriate prompt based on optimization type.
        
        Args:
            context: PromptContext with query, metrics, and optimization type
            
        Returns:
            Structured prompt string for LLM
        """
        # Validate query safety first
        is_safe, error = self.validate_query_safety(context.original_query)
        if not is_safe:
            self.logger.error(f"Query failed safety check: {error}")
            raise ValueError(f"Query safety validation failed: {error}")
        
        # GUARD: Check query length (Mistral context limit)
        if len(context.original_query) > 5000:
            raise ValueError(f"Query too long ({len(context.original_query)} chars). Max 5000 chars for analysis.")
        
        # Route to appropriate prompt builder
        if context.optimization_type == OptimizationType.INDEX_RECOMMENDATION:
            return self.build_index_recommendation_prompt(context)
        elif context.optimization_type == OptimizationType.JOIN_OPTIMIZATION:
            return self.build_join_optimization_prompt(context)
        else:
            return self.build_analysis_prompt(context)
    
    def split_prompt_sections(self, prompt: str) -> Dict[str, str]:
        """
        Split prompt into sections for debugging/analysis.
        
        Returns dict with sections: safety, query, metrics, rules, format
        """
        sections = {}
        current_section = None
        current_content = []
        
        for line in prompt.split('\n'):
            if line.startswith('##'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                current_section = line.replace('##', '').strip().lower()
                current_content = []
            else:
                current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def estimate_prompt_tokens(self, prompt: str) -> int:
        """
        Estimate token count for cost calculation (rough approximation).
        Actual token count: encode with tiktoken for accuracy.
        """
        # Rough estimate: 1 token ≈ 4 characters
        return len(prompt) // 4
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get recommended model configuration for Mistral SQL rewriting."""
        return {
            "model": self.model,
            # temperature=0 → deterministic output; eliminates creative prose additions
            "temperature": 0.0,
            "top_p": 0.9,
            # 600 tokens is enough for any realistic SQL rewrite; a higher limit
            # tempts the model to append natural-language explanation after the SQL.
            "max_tokens": 600,
        }


class PromptValidator:
    """Validates LLM responses for quality and safety."""
    
    @staticmethod
    def validate_response(response: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that LLM response meets quality criteria.
        
        Returns:
            (is_valid: bool, error_message: str or None)
        """
        if not response or len(response.strip()) == 0:
            return False, "Empty response from LLM"
        
        # Check for minimum content
        if len(response) < 50:
            return False, "Response too short to be useful"
        
        # Check for forbidden operations
        forbidden = {'INSERT', 'UPDATE', 'DELETE', 'DROP', 'CREATE', 'ALTER'}
        response_upper = response.upper()
        
        for operation in forbidden:
            if f' {operation} ' in f' {response_upper} ':
                logger.warning(f"Response suggests {operation} - violates safety rules")
                return False, f"Response suggests forbidden operation: {operation}"
        
        return True, None
    
    @staticmethod
    def extract_sql_from_response(response: str) -> Optional[str]:
        """Extract SQL code blocks from LLM response."""
        import re
        
        # Find SQL code blocks marked with ```sql
        pattern = r'```sql\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # Fallback: try to find any code block
        pattern = r'```\n(.*?)\n```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        return None