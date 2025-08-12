#!/usr/bin/env python3
"""Test script to validate UI enhancements match Python performance"""

import sys
import os
sys.path.append('/workspaces/ran_intelligent_management')

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def test_ui_enhancements():
    """Test that the UI enhancements properly reflect our Python improvements"""
    
    print("🧪 Testing UI Enhancement Validation")
    print("=" * 50)
    
    # Initialize the same way the UI does
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    chatbot = EnhancedRANChatbot(integrator, use_caching=True)
    
    # Test 1: Verify table extraction functionality 
    print("\n1️⃣ Testing Table Extraction Enhancement:")
    test_queries = [
        "Show me SectorEquipmentFunction table",
        "What is in AnrFunction table?",
        "Describe EUtranCellFDD table structure"
    ]
    
    extraction_successes = 0
    for query in test_queries:
        extracted = chatbot._extract_table_name(query)
        # Parse expected table name from query
        expected = None
        if "SectorEquipmentFunction" in query:
            expected = "SectorEquipmentFunction"
        elif "AnrFunction" in query:
            expected = "AnrFunction"
        elif "EUtranCellFDD" in query:
            expected = "EUtranCellFDD"
        
        success = extracted == expected
        extraction_successes += success
        status = "✅" if success else "❌"
        print(f"   {status} '{query}'")
        print(f"      Expected: {expected}, Got: {extracted}")
    
    extraction_rate = extraction_successes / len(test_queries) * 100
    print(f"\n   📊 Extraction Success Rate: {extraction_rate:.1f}%")
    
    # Test 2: Verify enhanced processing
    print("\n2️⃣ Testing Enhanced Processing:")
    test_query = "Show me SectorEquipmentFunction table data"
    
    try:
        result = chatbot.enhanced_process_query(test_query)
        
        # Check key enhancements
        checks = {
            "Parallel Aggregated Type": result.get('type') == 'parallel_aggregated',
            "Explicit Table Detected": bool(result.get('key_results', {}).get('explicit_table')),
            "Top Tables Present": bool(result.get('top_tables')),
            "Processing Time Recorded": bool(result.get('processing_time_ms')),
            "Entity Extraction": bool(result.get('entities'))
        }
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"   {status} {check}")
        
        enhancement_score = sum(checks.values()) / len(checks) * 100
        print(f"\n   📊 Enhancement Score: {enhancement_score:.1f}%")
        
    except Exception as e:
        print(f"   ❌ Enhanced processing failed: {e}")
    
    # Test 3: Verify cache performance
    print("\n3️⃣ Testing Cache Performance:")
    
    cache_stats = chatbot.get_cache_stats()
    cached_tables = len(chatbot._table_metadata_cache.get('all_tables', []))
    
    cache_checks = {
        "Cache Stats Available": bool(cache_stats),
        "Tables Cached": cached_tables > 200,  # Should have 273 tables
        "Cache Interface Working": hasattr(chatbot, '_table_metadata_cache')
    }
    
    for check, passed in cache_checks.items():
        status = "✅" if passed else "❌"
        detail = ""
        if check == "Tables Cached":
            detail = f" ({cached_tables} tables)"
        print(f"   {status} {check}{detail}")
    
    cache_score = sum(cache_checks.values()) / len(cache_checks) * 100
    print(f"\n   📊 Cache Performance: {cache_score:.1f}%")
    
    # Test 4: Query type classification (for UI display)
    print("\n4️⃣ Testing Query Classification:")
    
    query_tests = [
        ("Show me SectorEquipmentFunction table", "explicit_table"),
        ("Get CellPerformance.throughput data", "column_specific"), 
        ("Find power optimization tables", "domain_specific"),
        ("Search for handover configuration", "entity_focused")
    ]
    
    classification_successes = 0
    for query, expected_type in query_tests:
        # Simulate the classification logic from the UI
        query_lower = query.lower()
        explicit_table = chatbot._extract_table_name(query)
        
        if explicit_table:
            classified_type = "explicit_table"
        elif '.' in query and any(word[0].isupper() for word in query.split()):
            classified_type = "column_specific"
        elif any(domain in query_lower for domain in ['power', 'frequency', 'timing', 'performance']):
            classified_type = "domain_specific"
        elif any(entity in query_lower for entity in ['handover', 'interference', 'network', 'cell']):
            classified_type = "entity_focused"
        else:
            classified_type = "general_inquiry"
        
        success = classified_type == expected_type
        classification_successes += success
        status = "✅" if success else "❌"
        print(f"   {status} '{query}' → {classified_type}")
    
    classification_rate = classification_successes / len(query_tests) * 100
    print(f"\n   📊 Classification Accuracy: {classification_rate:.1f}%")
    
    # Overall summary
    print("\n" + "=" * 50)
    print("📋 UI Enhancement Validation Summary:")
    print(f"   • Table Extraction: {extraction_rate:.1f}%")
    print(f"   • Enhanced Processing: {enhancement_score:.1f}%")
    print(f"   • Cache Performance: {cache_score:.1f}%")
    print(f"   • Query Classification: {classification_rate:.1f}%")
    
    overall_score = (extraction_rate + enhancement_score + cache_score + classification_rate) / 4
    print(f"\n🎯 Overall UI Enhancement Score: {overall_score:.1f}%")
    
    if overall_score >= 80:
        print("🎉 Excellent! UI enhancements are working properly!")
    elif overall_score >= 60:
        print("✅ Good! UI enhancements are mostly working.")
    else:
        print("⚠️ Issues detected. UI enhancements need adjustment.")

if __name__ == "__main__":
    test_ui_enhancements()
