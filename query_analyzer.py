# SQL Query Analyzer - Extract fields, tables, joins, where clauses
# Author: AI SQL Optimizer
# Purpose: Parse and analyze SQL queries to extract their components

import logging
import re
from typing import Dict, List, Set, Any, Optional
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Where, Parenthesis, Comparison
from sqlparse.tokens import Keyword, DML, Whitespace, Punctuation

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes SQL SELECT queries to extract components: fields, tables, joins, where clauses."""

    @staticmethod
    def extract_fields(query: str) -> List[str]:
        """
        Extract all SELECT fields/columns from a query.
        
        Returns list of field names, respecting aliases:
        - SELECT name, age FROM users → ['name', 'age']
        - SELECT name AS user_name FROM users → ['user_name']
        - SELECT * FROM users → ['*']
        - SELECT COUNT(*) FROM users → ['COUNT(*)']
        """
        parsed = sqlparse.parse(query)[0]
        fields = []
        
        in_select = False
        select_tokens = []
        
        for token in parsed.tokens:
            # Find SELECT keyword
            if token.ttype in (Keyword, DML) and token.value.upper() == 'SELECT':
                in_select = True
                continue
            
            # Stop at FROM keyword
            if in_select and token.ttype is Keyword and token.value.upper() == 'FROM':
                in_select = False
                break
            
            # Collect tokens between SELECT and FROM
            if in_select and not token.is_whitespace:
                select_tokens.append(token)
        
        # Parse the collected tokens
        for token in select_tokens:
            if isinstance(token, IdentifierList):
                # Multiple fields separated by commas
                for identifier in token.get_identifiers():
                    field = QueryAnalyzer._extract_identifier_name(identifier)
                    if field:
                        fields.append(field)
            elif isinstance(token, Identifier):
                # Single field (possibly with alias)
                field = QueryAnalyzer._extract_identifier_name(token)
                if field:
                    fields.append(field)
            elif token.ttype is Punctuation and token.value == ',':
                continue
            else:
                # Regular token (function call, wildcard, etc.)
                val = token.value.strip()
                if val and val not in ('(', ')'):
                    fields.append(val)
        
        return fields if fields else ['*']

    @staticmethod
    def extract_tables(query: str) -> List[str]:
        """
        Extract all table names from a query.
        
        Returns list of table names, respecting aliases:
        - SELECT * FROM users → ['users']
        - SELECT * FROM users u → ['users']
        - SELECT * FROM users, orders → ['users', 'orders']
        - SELECT * FROM users JOIN orders ON ... → ['users', 'orders']
        """
        parsed = sqlparse.parse(query)[0]
        tables = []
        
        from_seen = False
        where_seen = False
        
        i = 0
        tokens = list(parsed.tokens)
        
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.value.upper().strip()
            
            # Look for FROM keyword
            if token.ttype is Keyword and token_upper == 'FROM':
                from_seen = True
                i += 1
                continue
            
            # Stop at WHERE, GROUP BY, ORDER BY, etc.
            if from_seen and token.ttype is Keyword and token_upper in ('WHERE', 'GROUP', 'ORDER', 'LIMIT', 'UNION', 'EXCEPT', 'INTERSECT'):
                from_seen = False
                where_seen = True
                break
            
            # Process FROM clause tokens
            if from_seen and not token.is_whitespace:
                if isinstance(token, IdentifierList):
                    # Multiple tables
                    for identifier in token.get_identifiers():
                        table = QueryAnalyzer._extract_table_name(identifier)
                        if table:
                            tables.append(table)
                elif isinstance(token, Identifier):
                    # Single table (possibly with alias)
                    table = QueryAnalyzer._extract_table_name(token)
                    if table:
                        tables.append(table)
                elif token.ttype is Keyword and token_upper in ('JOIN', 'LEFT', 'RIGHT', 'INNER', 'CROSS', 'FULL', 'OUTER'):
                    # Junction between tables - collect next identifier
                    i += 1
                    while i < len(tokens):
                        next_token = tokens[i]
                        if next_token.is_whitespace:
                            i += 1
                            continue
                        if isinstance(next_token, Identifier):
                            table = QueryAnalyzer._extract_table_name(next_token)
                            if table:
                                tables.append(table)
                            break
                        elif next_token.ttype is not Keyword:
                            val = next_token.value.strip()
                            if val and val != '(':
                                tables.append(val.split()[0])  # Take first word as table name
                            break
                        i += 1
                    i -= 1  # Adjust since we'll increment at loop end
                elif token.ttype is Punctuation and token.value == ',':
                    # Comma between tables in FROM list
                    continue
            
            i += 1
        
        return list(dict.fromkeys(tables))  # Remove duplicates while preserving order

    @staticmethod
    def extract_joins(query: str) -> List[Dict[str, str]]:
        """
        Extract JOIN information.
        
        Returns list of join info dicts:
        [
            {'type': 'INNER', 'table': 'orders', 'condition': 'users.id = orders.user_id'},
            {'type': 'LEFT', 'table': 'profiles', 'condition': 'users.id = profiles.user_id'}
        ]
        """
        parsed = sqlparse.parse(query)[0]
        joins = []
        
        i = 0
        tokens = list(parsed.tokens)
        
        while i < len(tokens):
            token = tokens[i]
            token_upper = token.value.upper().strip()
            
            # Look for JOIN keyword (with possible LEFT/RIGHT/INNER/CROSS/FULL prefix)
            if token.ttype is Keyword and 'JOIN' in token_upper:
                join_type = token_upper
                
                # Determine join type
                if i > 0:
                    prev_token_upper = tokens[i-1].value.upper().strip()
                    if prev_token_upper in ('LEFT', 'RIGHT', 'INNER', 'CROSS', 'FULL', 'OUTER'):
                        join_type = prev_token_upper + ' ' + join_type
                
                # Find table name after JOIN
                table_name = None
                condition = None
                
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j]
                    
                    if next_token.is_whitespace:
                        j += 1
                        continue
                    
                    if isinstance(next_token, Identifier):
                        table_name = QueryAnalyzer._extract_table_name(next_token)
                        break
                    elif next_token.ttype not in (Keyword, Whitespace):
                        table_name = next_token.value.strip()
                        break
                    
                    j += 1
                
                # Find ON clause
                j = i + 1
                while j < len(tokens):
                    next_token = tokens[j]
                    next_upper = next_token.value.upper().strip()
                    
                    if next_token.ttype is Keyword and next_upper == 'ON':
                        # Collect tokens until next keyword
                        k = j + 1
                        condition_tokens = []
                        while k < len(tokens):
                            cond_token = tokens[k]
                            cond_upper = cond_token.value.upper().strip()
                            
                            if cond_token.ttype is Keyword and cond_upper in ('WHERE', 'JOIN', 'GROUP', 'ORDER', 'LIMIT', 'UNION'):
                                break
                            
                            if not cond_token.is_whitespace and cond_token.value.strip():
                                condition_tokens.append(cond_token.value.strip())
                            
                            k += 1
                        
                        condition = ' '.join(condition_tokens)
                        break
                    
                    j += 1
                
                if table_name:
                    joins.append({
                        'type': join_type.strip(),
                        'table': table_name,
                        'condition': condition or 'No condition'
                    })
            
            i += 1
        
        return joins

    @staticmethod
    def extract_where_clause(query: str) -> Optional[Dict[str, Any]]:
        """
        Extract WHERE clause information.
        
        Returns dict with:
        {
            'raw': 'age > 18 AND city = "New York"',
            'conditions': ['age > 18', 'city = "New York"'],
            'operators': ['AND']
        }
        """
        parsed = sqlparse.parse(query)[0]
        
        where_clause = None
        for token in parsed.tokens:
            if isinstance(token, Where):
                where_clause = token
                break
        
        if not where_clause:
            return None
        
        # Get raw where clause text
        raw = where_clause.value.strip()
        if raw.upper().startswith('WHERE '):
            raw = raw[6:].strip()
        
        # Extract conditions and operators
        conditions = []
        operators = []
        
        # Simple regex-based extraction
        # Split by AND/OR keeping the operators
        parts = re.split(r'\s+(AND|OR)\s+', raw, flags=re.IGNORECASE)
        
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Even indices are conditions
                part = part.strip()
                if part:
                    conditions.append(part)
            else:  # Odd indices are operators
                operators.append(part.upper())
        
        return {
            'raw': raw,
            'conditions': conditions,
            'operators': operators
        }

    @staticmethod
    def _extract_identifier_name(identifier) -> Optional[str]:
        """Extract the actual name from an Identifier (handling aliases)."""
        if not isinstance(identifier, Identifier):
            return identifier.value.strip() if identifier else None
        
        # Check if has alias (AS keyword)
        identifier_str = str(identifier)
        
        # If has AS, take the part after AS
        if ' AS ' in identifier_str.upper():
            parts = re.split(r'\s+AS\s+', identifier_str, flags=re.IGNORECASE)
            return parts[-1].strip()
        
        # Otherwise take all parts
        real_name = identifier.get_real_name()
        if real_name:
            return real_name
        
        return identifier.value.strip()

    @staticmethod
    def _extract_table_name(identifier) -> Optional[str]:
        """Extract table name from an Identifier (handling aliases)."""
        if not isinstance(identifier, Identifier):
            return identifier.value.strip() if identifier else None
        
        # get_real_name() returns the table name without alias
        real_name = identifier.get_real_name()
        return real_name if real_name else identifier.value.strip()

    @staticmethod
    def analyze_full(query: str) -> Dict[str, Any]:
        """
        Perform complete query analysis and return all components.
        
        Returns:
        {
            'fields': ['name', 'age'],
            'tables': ['users'],
            'joins': [{'type': 'INNER', 'table': 'orders', 'condition': '...'}],
            'where_clause': {'raw': '...', 'conditions': [...], 'operators': [...]},
            'complexity': 'simple' | 'moderate' | 'complex'
        }
        """
        fields = QueryAnalyzer.extract_fields(query)
        tables = QueryAnalyzer.extract_tables(query)
        joins = QueryAnalyzer.extract_joins(query)
        where = QueryAnalyzer.extract_where_clause(query)
        
        # Determine complexity
        complexity = 'simple'
        if len(joins) > 0 or where is not None:
            complexity = 'moderate'
        if len(joins) > 2 or (where and len(where['conditions']) > 3):
            complexity = 'complex'
        
        return {
            'fields': fields,
            'tables': tables,
            'joins': joins,
            'where_clause': where,
            'complexity': complexity,
            'field_count': len(fields),
            'table_count': len(tables),
            'join_count': len(joins),
            'condition_count': len(where['conditions']) if where else 0
        }
