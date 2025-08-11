#!/usr/bin/env python3
"""Test with ACTUAL table names from the knowledge graph"""

import sys
import os
sys.path.append('/workspaces/ran_intelligent_management')

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def test_with_real_tables():
    """Test explicit table extraction with actual KG tables"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    # Test with ACTUAL table names from the KG
    real_test_queries = [
        "What is in SectorEquipmentFunction table?",
        "Show me EUtranCellFDD table data", 
        "Describe NRCellDU table",
        "Find AnrFunction information",
        "What columns are in TimeSettings?",
        "Show CellPerformance.throughput details",
        "Get MeContext.element_type data",
        "Query CarrierAggregationFunction table",
        "sectorequipmentfunction information",  # lowercase
        "EUtranCellFdd details",               # Case variation
        "Show EUtranCellRelation data",
        "Find PowerSaving table",
        "Get UeMC configuration",
        "Describe Transceiver table",
        "What's in RadioBearerTable?"
    ]
    
    print("Testing with REAL Knowledge Graph Tables:")
    print("=" * 55)
    
    correct_extractions = 0
    correct_rankings = 0
    
    for i, query in enumerate(real_test_queries, 1):
        print(f"\n{i:2d}. Query: '{query}'")
        
        # Test extraction method
        extracted_table = chatbot._extract_table_name(query)
        print(f"    Extracted: {extracted_table}")
        
        # Get actual response
        try:
            response = chatbot.process_query(query)
            top_table = response.get('top_tables', [{}])[0].get('table_name', 'None')
            explicit_table = response.get('key_results', {}).get('explicit_table', 'None')
            score = response.get('top_tables', [{}])[0].get('total_score', 0)
            
            print(f"    Top result: {top_table} (score: {score:.1f})")
            print(f"    Explicit table: {explicit_table}")
            
            # Check if extraction is correct
            extraction_correct = extracted_table and extracted_table == top_table
            if extraction_correct:
                correct_extractions += 1
                print(f"    Extraction: ✓")
            else:
                print(f"    Extraction: ✗")
            
            # Check if ranking is correct (explicit table should be top)
            ranking_correct = explicit_table and explicit_table == top_table
            if ranking_correct:
                correct_rankings += 1
                print(f"    Ranking: ✓")
            else:
                print(f"    Ranking: ✗")
                
        except Exception as e:
            print(f"    Error: {e}")
    
    print(f"\n" + "=" * 55)
    print(f"Results Summary:")
    print(f"  Correct extractions: {correct_extractions}/{len(real_test_queries)} ({correct_extractions/len(real_test_queries)*100:.1f}%)")
    print(f"  Correct rankings: {correct_rankings}/{len(real_test_queries)} ({correct_rankings/len(real_test_queries)*100:.1f}%)")

if __name__ == "__main__":
    test_with_real_tables()
