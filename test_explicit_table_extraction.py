#!/usr/bin/env python3
"""Test explicit table extraction to debug the poor P@1 performance"""

import sys
import os
sys.path.append('/workspaces/ran_intelligent_management')

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot
import json

def test_explicit_table_extraction():
    """Test the enhanced table extraction against explicit table queries"""
    
    # Initialize chatbot with Neo4j connection
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    # Test queries from the improved ground truth
    test_queries = [
        "What is in TDD_FRAME_STRUCT table?",
        "Show me CELL_LOCAL_RELATED table data",
        "Describe SectorEquipmentFunction table",
        "Find CELL_EQUIPMENT_RELATED information",
        "What columns are in TDD_FRAME_STRUCT_RELATED?",
        "Show TDD_FRAME_STRUCT_RELATED.frame_config details",
        "Get MANAGED_ELEMENT.element_type data",
        "Query TDD_Frame_Struct table",  # Case variation
        "tdd_frame_struct information",   # Lowercase
        "TddFrameStruct details"          # PascalCase
    ]
    
    print("Testing Enhanced Table Extraction:")
    print("=" * 50)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Query: '{query}'")
        
        # Test extraction method
        extracted_table = chatbot._extract_table_name(query)
        print(f"   Extracted: {extracted_table}")
        
        # Get actual response
        try:
            response = chatbot.process_query(query)
            top_table = response.get('top_tables', [{}])[0].get('table_name', 'None')
            explicit_table = response.get('key_results', {}).get('explicit_table', 'None')
            
            print(f"   Top result: {top_table}")
            print(f"   Explicit table: {explicit_table}")
            print(f"   Match: {'✓' if extracted_table and extracted_table == top_table else '✗'}")
            
        except Exception as e:
            print(f"   Error: {e}")

def test_table_metadata():
    """Test if our table metadata cache is working"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    print("\nTesting Table Metadata Cache:")
    print("=" * 30)
    
    # Check cache
    all_tables = chatbot._table_metadata_cache.get('all_tables', [])
    print(f"Cached tables: {len(all_tables)}")
    
    # Check some expected tables
    test_tables = ['TDD_FRAME_STRUCT', 'CELL_LOCAL_RELATED', 'SectorEquipmentFunction']
    for table in test_tables:
        if table in all_tables:
            print(f"✓ {table} found in cache")
        else:
            print(f"✗ {table} NOT in cache")
            # Try variations
            variations = [table.lower(), table.upper(), 
                         ''.join(word.capitalize() for word in table.split('_'))]
            for var in variations:
                if var in all_tables:
                    print(f"  Found variation: {var}")
                    break
def test_pattern_matching():
    """Test the regex patterns"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    print("\nTesting Pattern Matching:")
    print("=" * 25)
    
    test_cases = [
        "What is in TDD_FRAME_STRUCT table?",
        "Show TDD_FRAME_STRUCT_RELATED.frame_config",
        "Query TDD_Frame_Struct",
        "tdd_frame_struct information",
        "TddFrameStruct details"
    ]
    
    for query in test_cases:
        print(f"\nQuery: '{query}'")
        
        # Test pattern
        matches = chatbot._table_name_pattern.findall(query)
        print(f"Pattern matches: {matches}")

if __name__ == "__main__":
    test_explicit_table_extraction()
    test_table_metadata()
    test_pattern_matching()
