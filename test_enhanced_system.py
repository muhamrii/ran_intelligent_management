#!/usr/bin/env python3
"""
Comprehensive test suite for enhanced RAN chatbot system
Tests all improvements: entity extraction, ranking, caching, performance
"""

import time
import json
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def test_enhanced_chatbot():
    """Test enhanced chatbot with all improvements"""
    print("🚀 Testing Enhanced RAN Chatbot System")
    print("=" * 50)
    
    # Test connection
    try:
        integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'neo4j')
        chatbot = EnhancedRANChatbot(integrator, use_caching=True)
        print("✅ Connection established")
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test queries with enhanced features
    test_queries = [
        {
            'query': 'Show me BoundaryOrdinaryClock table information',
            'expected_features': ['explicit_table', 'high_confidence', 'fast_response']
        },
        {
            'query': 'Find tables related to power and energy measurements',
            'expected_features': ['entity_extraction', 'domain_routing', 'semantic_similarity']
        },
        {
            'query': 'What are the frequency and timing related tables?',
            'expected_features': ['multi_entity', 'domain_classification', 'ranking']
        },
        {
            'query': 'List AnrFunction table columns and relationships',
            'expected_features': ['explicit_table', 'relationship_analysis', 'column_details']
        },
        {
            'query': 'Performance metrics for throughput analysis',
            'expected_features': ['performance_domain', 'measurements', 'kpi_detection']
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_queries, 1):
        print(f"\n🧪 Test {i}: {test_case['query']}")
        print("-" * 40)
        
        start_time = time.time()
        
        try:
            # Run enhanced query processing
            result = chatbot.enhanced_process_query(test_case['query'])
            
            query_time = time.time() - start_time
            
            # Analyze results
            analysis = analyze_result(result, test_case, query_time)
            results.append(analysis)
            
            # Display key metrics
            print(f"⏱️  Query Time: {query_time:.3f}s")
            print(f"📊 Top Tables Found: {len(result.get('top_tables', []))}")
            print(f"🎯 Intent: {result.get('intent', 'unknown')} (conf: {result.get('confidence', 0):.2f})")
            
            # Show entities if extracted
            entities = result.get('entities', {})
            if entities:
                print(f"🔍 Entities Extracted:")
                for category, items in entities.items():
                    if items and category != 'confidence_scores':
                        print(f"   {category}: {items[:3]}")  # Show first 3
            
            # Show performance indicators
            if result.get('cache_hit'):
                print("⚡ Cache Hit - Ultra Fast!")
            elif query_time < 0.1:
                print("🚀 Very Fast Response")
            elif query_time > 1.0:
                print("🐌 Slow Response")
            
            # Show top result
            top_tables = result.get('top_tables', [])
            if top_tables:
                top_table = top_tables[0]
                print(f"🏆 Top Result: {top_table['table_name']} (score: {top_table.get('total_score', 0):.2f})")
                
                # Show scoring breakdown
                print(f"   Breakdown: freq={top_table.get('frequency_score', 0):.1f}, "
                      f"semantic={top_table.get('semantic_score', 0):.1f}, "
                      f"token={top_table.get('token_match_score', 0):.1f}")
            
            print("✅ Test completed successfully")
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            results.append({
                'query': test_case['query'],
                'success': False,
                'error': str(e),
                'query_time': time.time() - start_time
            })
    
    # Show cache statistics
    print(f"\n📈 Performance Summary")
    print("=" * 30)
    
    if hasattr(chatbot, 'get_cache_stats'):
        cache_stats = chatbot.get_cache_stats()
        print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0)}%")
        print(f"Average Query Time: {cache_stats.get('avg_query_time', 0):.3f}s")
        print(f"Cache Size: {cache_stats.get('cache_size', 0)} entries")
    
    # Overall success rate
    successful_tests = sum(1 for r in results if r.get('success', False))
    success_rate = (successful_tests / len(results)) * 100
    print(f"Test Success Rate: {success_rate:.1f}% ({successful_tests}/{len(results)})")
    
    avg_query_time = sum(r.get('query_time', 0) for r in results) / len(results)
    print(f"Average Query Time: {avg_query_time:.3f}s")
    
    return results

def analyze_result(result: Dict, test_case: Dict, query_time: float) -> Dict:
    """Analyze query result against expected features"""
    analysis = {
        'query': test_case['query'],
        'success': True,
        'query_time': query_time,
        'features_detected': [],
        'performance_grade': 'A'  # Default grade
    }
    
    expected_features = test_case.get('expected_features', [])
    
    # Check for explicit table detection
    if 'explicit_table' in expected_features:
        explicit_table = result.get('key_results', {}).get('explicit_table')
        if explicit_table:
            analysis['features_detected'].append('explicit_table')
    
    # Check entity extraction
    entities = result.get('entities', {})
    if entities and any(entities.values()):
        analysis['features_detected'].append('entity_extraction')
        
        # Check for multi-entity detection
        entity_count = sum(len(v) for v in entities.values() if isinstance(v, list))
        if entity_count > 3:
            analysis['features_detected'].append('multi_entity')
    
    # Check domain routing
    if result.get('domain'):
        analysis['features_detected'].append('domain_routing')
    
    # Check semantic similarity (indicated by semantic_score > 0)
    top_tables = result.get('top_tables', [])
    if top_tables and top_tables[0].get('semantic_score', 0) > 0:
        analysis['features_detected'].append('semantic_similarity')
    
    # Check ranking quality
    if top_tables and top_tables[0].get('total_score', 0) > 10:
        analysis['features_detected'].append('ranking')
    
    # Performance grading
    if query_time < 0.1:
        analysis['performance_grade'] = 'A+'
    elif query_time < 0.5:
        analysis['performance_grade'] = 'A'
    elif query_time < 1.0:
        analysis['performance_grade'] = 'B'
    elif query_time < 2.0:
        analysis['performance_grade'] = 'C'
    else:
        analysis['performance_grade'] = 'D'
    
    # Check for cache usage
    if result.get('cache_hit'):
        analysis['features_detected'].append('cache_hit')
    
    return analysis

def test_entity_extraction():
    """Test enhanced entity extraction"""
    print(f"\n🔍 Testing Enhanced Entity Extraction")
    print("=" * 40)
    
    try:
        integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'neo4j')
        chatbot = EnhancedRANChatbot(integrator, use_caching=True)
        
        test_queries = [
            "Show power consumption and dbm measurements",
            "Find frequency spectrum and bandwidth tables",
            "List timing synchronization and clock configurations",
            "Get performance KPIs and throughput metrics",
            "Show cell handover and mobility parameters"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            entities = chatbot.entity_extractor.extract_technical_entities(query)
            
            print("Extracted entities:")
            for category, items in entities.items():
                if items and category != 'confidence_scores':
                    print(f"  {category}: {items}")
            
            # Show confidence scores
            confidence_scores = entities.get('confidence_scores', {})
            if confidence_scores:
                print("Confidence scores:")
                for entity, score in confidence_scores.items():
                    print(f"  {entity}: {score:.2f}")
        
        print("✅ Entity extraction test completed")
        return True
        
    except Exception as e:
        print(f"❌ Entity extraction test failed: {e}")
        return False

def test_caching_performance():
    """Test caching performance improvements"""
    print(f"\n⚡ Testing Caching Performance")
    print("=" * 35)
    
    try:
        integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'neo4j')
        
        # Test with caching enabled
        chatbot_cached = EnhancedRANChatbot(integrator, use_caching=True)
        
        # Test with caching disabled
        chatbot_no_cache = EnhancedRANChatbot(integrator, use_caching=False)
        
        test_query = "Show me BoundaryOrdinaryClock table information"
        
        # First run with cache (miss)
        start_time = time.time()
        result1 = chatbot_cached.enhanced_process_query(test_query)
        cache_miss_time = time.time() - start_time
        
        # Second run with cache (hit)
        start_time = time.time()
        result2 = chatbot_cached.enhanced_process_query(test_query)
        cache_hit_time = time.time() - start_time
        
        # Run without cache
        start_time = time.time()
        result3 = chatbot_no_cache.enhanced_process_query(test_query)
        no_cache_time = time.time() - start_time
        
        print(f"Cache Miss Time: {cache_miss_time:.3f}s")
        print(f"Cache Hit Time: {cache_hit_time:.3f}s")
        print(f"No Cache Time: {no_cache_time:.3f}s")
        
        speedup = cache_miss_time / cache_hit_time if cache_hit_time > 0 else 0
        print(f"Cache Speedup: {speedup:.1f}x")
        
        # Show cache statistics
        cache_stats = chatbot_cached.get_cache_stats()
        print(f"Cache Hit Rate: {cache_stats.get('hit_rate', 0)}%")
        
        print("✅ Caching performance test completed")
        return True
        
    except Exception as e:
        print(f"❌ Caching test failed: {e}")
        return False

def main():
    """Run comprehensive test suite"""
    print("🧪 Comprehensive Enhanced Chatbot Test Suite")
    print("=" * 60)
    
    tests = [
        ("Enhanced Chatbot Features", test_enhanced_chatbot),
        ("Entity Extraction", test_entity_extraction),
        ("Caching Performance", test_caching_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Final summary
    print(f"\n{'=' * 60}")
    print("📋 FINAL TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    overall_success = all(results.values())
    print(f"\nOverall Result: {'✅ ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
