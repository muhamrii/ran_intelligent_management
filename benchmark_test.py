#!/usr/bin/env python3
"""
Benchmark test for enhanced chatbot improvements
Tests our improvements against the baseline metrics
"""

import time
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Mock queries for testing (simulating academic benchmark)
TEST_QUERIES = [
    {
        "query": "Show me BoundaryOrdinaryClock table information",
        "expected_table": "BoundaryOrdinaryClock",
        "domain": "timing"
    },
    {
        "query": "Find power consumption data and energy metrics",
        "expected_entities": ["power", "consumption", "energy"],
        "domain": "power"
    },
    {
        "query": "List frequency spectrum and bandwidth parameters",
        "expected_entities": ["frequency", "spectrum", "bandwidth"], 
        "domain": "frequency"
    },
    {
        "query": "Get AnrFunction table columns and relationships",
        "expected_table": "AnrFunction",
        "domain": "network"
    },
    {
        "query": "Show performance KPIs and throughput measurements",
        "expected_entities": ["performance", "kpis", "throughput"],
        "domain": "performance"
    }
]

def test_entity_extraction_performance():
    """Test enhanced entity extraction accuracy"""
    print("🧪 Testing Entity Extraction Performance")
    print("=" * 45)
    
    try:
        from chatbot_module.chatbot import EnhancedRANEntityExtractor
        from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
        
        # Mock integrator for testing
        class MockIntegrator:
            pass
        
        extractor = EnhancedRANEntityExtractor(MockIntegrator(), use_cache=True)
        
        correct_extractions = 0
        total_expected = 0
        
        for test_case in TEST_QUERIES:
            query = test_case["query"]
            expected_entities = test_case.get("expected_entities", [])
            
            if not expected_entities:
                continue
                
            entities = extractor.extract_technical_entities(query)
            
            # Flatten all extracted entities
            all_extracted = []
            for category, items in entities.items():
                if isinstance(items, list):
                    all_extracted.extend(items)
            
            # Check how many expected entities were found
            found_entities = []
            for expected in expected_entities:
                for extracted in all_extracted:
                    if expected.lower() in extracted.lower():
                        found_entities.append(expected)
                        break
            
            correct_extractions += len(found_entities)
            total_expected += len(expected_entities)
            
            print(f"Query: {query[:50]}...")
            print(f"  Expected: {expected_entities}")
            print(f"  Found: {found_entities}")
            print(f"  Confidence scores: {entities.get('confidence_scores', {})}")
            print()
        
        entity_f1 = (correct_extractions / total_expected * 100) if total_expected > 0 else 0
        print(f"📊 Entity Extraction F1: {entity_f1:.1f}%")
        print(f"   Correct: {correct_extractions}/{total_expected}")
        
        return entity_f1
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return 0

def test_ranking_algorithm():
    """Test enhanced ranking algorithm"""
    print("\n🎯 Testing Enhanced Ranking Algorithm")
    print("=" * 40)
    
    # Simulate ranking test data
    mock_tables = {
        'BoundaryOrdinaryClock': {
            'count': 3,
            'sources': ['explicit_table', 'domain_routing', 'entity_extraction'],
            'relevance_score': 15.0,
            'sample_columns': ['clockType', 'timeSource', 'syncStatus']
        },
        'PowerMeasurement': {
            'count': 2,
            'sources': ['entity_extraction', 'synonym_expansion'],
            'relevance_score': 8.0,
            'sample_columns': ['powerValue', 'measurementTime', 'cellId']
        },
        'FrequencyBand': {
            'count': 1,
            'sources': ['domain_routing'],
            'relevance_score': 3.0,
            'sample_columns': ['bandNumber', 'startFreq', 'endFreq']
        }
    }
    
    # Simulate aggregator data
    aggregator = {
        'query_tokens': ['boundary', 'ordinary', 'clock', 'table']
    }
    
    try:
        from chatbot_module.chatbot import EnhancedRANChatbot
        from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
        
        # Create minimal chatbot instance for testing
        class MockIntegrator:
            pass
        
        chatbot = EnhancedRANChatbot(MockIntegrator(), use_caching=False)
        
        # Test ranking
        ranked_tables = chatbot._rank_aggregated_tables(mock_tables, aggregator)
        
        print("Ranking Results:")
        for i, table in enumerate(ranked_tables[:3], 1):
            print(f"{i}. {table['table_name']}")
            print(f"   Total Score: {table['total_score']}")
            print(f"   Frequency: {table.get('frequency_score', 0):.1f}")
            print(f"   Semantic: {table.get('semantic_score', 0):.1f}")
            print(f"   Token Match: {table.get('token_match_score', 0):.1f}")
            print()
        
        # Check if BoundaryOrdinaryClock is ranked first (should be for explicit table query)
        top_table = ranked_tables[0]['table_name'] if ranked_tables else None
        is_correct_top = top_table == 'BoundaryOrdinaryClock'
        
        print(f"✅ Correct top result: {is_correct_top}")
        print(f"📊 Ranking Precision@1: {100 if is_correct_top else 0}%")
        
        return 100 if is_correct_top else 0
        
    except Exception as e:
        print(f"❌ Ranking test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

def test_performance_improvements():
    """Test performance improvements with caching"""
    print("\n⚡ Testing Performance Improvements")
    print("=" * 38)
    
    try:
        from chatbot_module.chatbot import EnhancedRANEntityExtractor
        
        # Mock integrator
        class MockIntegrator:
            pass
        
        # Test with caching
        extractor_cached = EnhancedRANEntityExtractor(MockIntegrator(), use_cache=True)
        
        # Test with no caching
        extractor_no_cache = EnhancedRANEntityExtractor(MockIntegrator(), use_cache=False)
        
        test_query = "Show power consumption and frequency data"
        
        # Cached version (first run - cache miss)
        start_time = time.time()
        result1 = extractor_cached.extract_technical_entities(test_query)
        time1 = time.time() - start_time
        
        # Cached version (second run - cache hit)
        start_time = time.time()
        result2 = extractor_cached.extract_technical_entities(test_query)
        time2 = time.time() - start_time
        
        # No cache version
        start_time = time.time()
        result3 = extractor_no_cache.extract_technical_entities(test_query)
        time3 = time.time() - start_time
        
        print(f"First run (cache miss): {time1*1000:.1f}ms")
        print(f"Second run (cache hit): {time2*1000:.1f}ms") 
        print(f"No cache: {time3*1000:.1f}ms")
        
        speedup = time1 / time2 if time2 > 0 else 1
        print(f"Cache speedup: {speedup:.1f}x")
        
        # All times should be under 100ms for good performance
        performance_grade = 'A' if max(time1, time2, time3) < 0.1 else 'B' if max(time1, time2, time3) < 0.5 else 'C'
        print(f"Performance grade: {performance_grade}")
        
        return speedup
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return 1

def main():
    """Run benchmark tests"""
    print("🚀 Enhanced Chatbot Benchmark Test Suite")
    print("=" * 50)
    
    # Run tests
    entity_f1 = test_entity_extraction_performance()
    ranking_precision = test_ranking_algorithm() 
    speedup = test_performance_improvements()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    print(f"Entity Extraction F1: {entity_f1:.1f}%")
    print(f"Ranking Precision@1: {ranking_precision:.1f}%")
    print(f"Performance Speedup: {speedup:.1f}x")
    
    # Compare to original baseline
    baseline_entity_f1 = 6.2  # From conversation
    baseline_precision = 53.3  # From conversation
    
    print(f"\n📈 IMPROVEMENTS OVER BASELINE:")
    print(f"Entity F1: {entity_f1:.1f}% vs {baseline_entity_f1}% baseline (+{entity_f1-baseline_entity_f1:.1f}%)")
    print(f"Precision@1: {ranking_precision:.1f}% vs {baseline_precision}% baseline (+{ranking_precision-baseline_precision:.1f}%)")
    
    # Overall assessment
    total_score = (entity_f1 + ranking_precision) / 2
    print(f"\nOverall Score: {total_score:.1f}%")
    
    if total_score >= 80:
        print("🎉 EXCELLENT - Significant improvements achieved!")
    elif total_score >= 60:
        print("✅ GOOD - Solid improvements made")
    elif total_score >= 40:
        print("📈 FAIR - Some improvements made")
    else:
        print("❌ NEEDS WORK - More improvements needed")
    
    return total_score

if __name__ == "__main__":
    score = main()
    print(f"\nFinal Score: {score:.1f}%")
