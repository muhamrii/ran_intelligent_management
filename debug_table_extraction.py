#!/usr/bin/env python3
"""Debug the table cache and pattern issues"""

import sys
import os
sys.path.append('/workspaces/ran_intelligent_management')

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def debug_cache_loading():
    """Debug what's happening with the cache"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    print("Debugging Cache Loading:")
    print("=" * 30)
    
    # Check what's in the cache
    cache = chatbot._table_metadata_cache
    print(f"Cache keys: {list(cache.keys())}")
    
    for key, value in cache.items():
        if isinstance(value, list):
            print(f"{key}: {len(value)} items - {value[:5]}")
        else:
            print(f"{key}: {value}")
    
    # Try to manually load tables
    print("\nManual table loading test:")
    try:
        with integrator.driver.session() as session:
            result = session.run("MATCH (t:Table) RETURN t.name as name LIMIT 10")
            tables = [record['name'] for record in result]
            print(f"Direct query found {len(tables)} tables: {tables}")
    except Exception as e:
        print(f"Direct query failed: {e}")

def debug_pattern():
    """Debug the pattern matching"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    print("\nDebugging Pattern Matching:")
    print("=" * 30)
    
    # Check the pattern
    pattern = chatbot._table_name_pattern
    print(f"Current pattern: {pattern.pattern}")
    
    # Test various patterns
    test_query = "What is in TDD_FRAME_STRUCT table?"
    
    import re
    
    # Test different regex patterns
    patterns = [
        r'\b[A-Z][A-Z0-9_]*[A-Z0-9]\b',  # All caps with underscores
        r'\b[A-Za-z][A-Za-z0-9_]*\b',    # Any identifier
        r'\b[A-Z][a-zA-Z0-9_]*\b',       # Starts with capital
        r'([A-Z][A-Z0-9_]+|[A-Z][a-z][a-zA-Z0-9]*)',  # Caps or PascalCase
    ]
    
    for pattern_str in patterns:
        matches = re.findall(pattern_str, test_query)
        print(f"Pattern '{pattern_str}': {matches}")

if __name__ == "__main__":
    debug_cache_loading()
    debug_pattern()
