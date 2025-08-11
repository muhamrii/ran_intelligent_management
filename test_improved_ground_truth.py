#!/usr/bin/env python3
"""
Test Enhanced Chatbot with Improved Ground Truth
Compare performance using the new realistic ground truth data
"""

import pandas as pd
import numpy as np
import time
import sys
import os
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def calculate_ir_metrics(ground_truth_df: pd.DataFrame, chatbot_results: List[Dict]) -> Dict[str, float]:
    """Calculate IR metrics with the improved ground truth"""
    
    # Initialize metrics
    precision_at_1 = []
    precision_at_5 = []
    precision_at_10 = []
    recall_at_5 = []
    recall_at_10 = []
    average_precisions = []
    
    for i, (_, row) in enumerate(ground_truth_df.iterrows()):
        if i >= len(chatbot_results):
            break
            
        result = chatbot_results[i]
        
        # Get expected tables (remove empty strings)
        expected_tables = [
            row['expected_table_1'],
            row['expected_table_2'], 
            row['expected_table_3']
        ]
        expected_tables = [t for t in expected_tables if t and str(t).strip()]
        
        if not expected_tables:
            continue
            
        # Get retrieved tables
        retrieved_tables = []
        if 'top_tables' in result:
            retrieved_tables = [t.get('table_name', '') for t in result['top_tables']]
        
        if not retrieved_tables:
            precision_at_1.append(0)
            precision_at_5.append(0)
            precision_at_10.append(0)
            recall_at_5.append(0)
            recall_at_10.append(0)
            average_precisions.append(0)
            continue
        
        # Calculate precision@k
        relevant_at_1 = len([t for t in retrieved_tables[:1] if t in expected_tables])
        relevant_at_5 = len([t for t in retrieved_tables[:5] if t in expected_tables])
        relevant_at_10 = len([t for t in retrieved_tables[:10] if t in expected_tables])
        
        precision_at_1.append(relevant_at_1 / 1)
        precision_at_5.append(relevant_at_5 / min(5, len(retrieved_tables)))
        precision_at_10.append(relevant_at_10 / min(10, len(retrieved_tables)))
        
        # Calculate recall@k
        recall_at_5.append(relevant_at_5 / len(expected_tables))
        recall_at_10.append(relevant_at_10 / len(expected_tables))
        
        # Calculate Average Precision
        ap = 0
        relevant_found = 0
        for j, table in enumerate(retrieved_tables[:10]):
            if table in expected_tables:
                relevant_found += 1
                precision_at_j = relevant_found / (j + 1)
                ap += precision_at_j
        
        if len(expected_tables) > 0:
            ap = ap / len(expected_tables)
        average_precisions.append(ap)
    
    return {
        'P@1': np.mean(precision_at_1) * 100,
        'P@5': np.mean(precision_at_5) * 100,
        'P@10': np.mean(precision_at_10) * 100,
        'R@5': np.mean(recall_at_5) * 100,
        'R@10': np.mean(recall_at_10) * 100,
        'MAP': np.mean(average_precisions) * 100
    }

def calculate_entity_metrics(ground_truth_df: pd.DataFrame, chatbot_results: List[Dict]) -> Dict[str, float]:
    """Calculate entity extraction metrics"""
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i, (_, row) in enumerate(ground_truth_df.iterrows()):
        if i >= len(chatbot_results):
            break
            
        result = chatbot_results[i]
        
        # Get expected entities
        expected_entities_str = str(row.get('entities', ''))
        if not expected_entities_str or expected_entities_str == 'nan':
            continue
            
        expected_entities = set([e.strip().lower() for e in expected_entities_str.split(',') if e.strip()])
        
        # Get extracted entities
        extracted_entities = set()
        if 'entities' in result:
            entities_dict = result['entities']
            for category, entity_list in entities_dict.items():
                if isinstance(entity_list, list):
                    for entity in entity_list:
                        extracted_entities.add(str(entity).lower().strip())
        
        if not expected_entities:
            continue
            
        # Calculate metrics
        if extracted_entities:
            intersection = expected_entities.intersection(extracted_entities)
            precision = len(intersection) / len(extracted_entities)
            recall = len(intersection) / len(expected_entities)
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0
        else:
            precision = 0
            recall = 0
            f1 = 0
            
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)
    
    return {
        'Entity Precision': np.mean(precision_scores) * 100,
        'Entity Recall': np.mean(recall_scores) * 100,
        'Entity F1': np.mean(f1_scores) * 100
    }

def test_with_improved_ground_truth():
    """Test chatbot with improved ground truth data"""
    print("🧪 Testing Enhanced Chatbot with Improved Ground Truth")
    print("=" * 60)
    
    # Load improved ground truth
    try:
        ir_df = pd.read_csv('improved_ir_ground_truth.csv')
        print(f"✅ Loaded {len(ir_df)} IR test queries")
    except FileNotFoundError:
        print("❌ Improved ground truth file not found. Run improved_ground_truth_generator.py first.")
        return
    
    # Connect to chatbot
    try:
        integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
        chatbot = EnhancedRANChatbot(integrator, use_caching=True)
        print("✅ Connected to enhanced chatbot")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        return
    
    # Run test queries
    print(f"\n🔄 Processing {len(ir_df)} test queries...")
    results = []
    
    start_time = time.time()
    
    for i, (_, row) in enumerate(ir_df.iterrows()):
        query = row['query']
        query_type = row['query_type']
        
        try:
            result = chatbot.enhanced_process_query(query)
            result['query_type'] = query_type
            results.append(result)
            
            # Show progress
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Processed {i + 1}/{len(ir_df)} queries ({elapsed:.1f}s)")
                
        except Exception as e:
            print(f"  ❌ Error processing query {i + 1}: {e}")
            results.append({'error': str(e), 'query_type': query_type})
    
    total_time = time.time() - start_time
    print(f"✅ Completed processing in {total_time:.1f}s")
    
    # Calculate metrics
    print(f"\n📊 Calculating Performance Metrics...")
    
    ir_metrics = calculate_ir_metrics(ir_df, results)
    entity_metrics = calculate_entity_metrics(ir_df, results)
    
    # Display results
    print(f"\n{'=' * 60}")
    print("📈 IMPROVED GROUND TRUTH BENCHMARK RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Information Retrieval Metrics:")
    for metric, value in ir_metrics.items():
        print(f"  {metric:8}: {value:6.1f}%")
    
    print(f"\n🔍 Entity Extraction Metrics:")
    for metric, value in entity_metrics.items():
        print(f"  {metric:18}: {value:6.1f}%")
    
    # Performance by query type
    print(f"\n📋 Performance by Query Type:")
    query_type_performance = {}
    
    for query_type in ir_df['query_type'].unique():
        type_indices = ir_df[ir_df['query_type'] == query_type].index
        type_results = [results[i] for i in type_indices if i < len(results)]
        type_ground_truth = ir_df[ir_df['query_type'] == query_type]
        
        if len(type_results) > 0:
            type_metrics = calculate_ir_metrics(type_ground_truth, type_results)
            query_type_performance[query_type] = type_metrics
            
            print(f"\n  {query_type}:")
            print(f"    P@1: {type_metrics['P@1']:5.1f}%")
            print(f"    MAP: {type_metrics['MAP']:5.1f}%")
    
    # Performance statistics
    avg_response_time = total_time / len(results) * 1000  # ms
    successful_queries = len([r for r in results if 'error' not in r])
    success_rate = (successful_queries / len(results)) * 100
    
    print(f"\n⚡ Performance Statistics:")
    print(f"  Success Rate: {success_rate:5.1f}%")
    print(f"  Avg Response Time: {avg_response_time:5.1f}ms")
    print(f"  Total Queries: {len(results)}")
    
    # Cache statistics
    if hasattr(chatbot, 'get_cache_stats'):
        cache_stats = chatbot.get_cache_stats()
        print(f"  Cache Hit Rate: {cache_stats.get('hit_rate', 0):5.1f}%")
    
    # Compare with baseline
    print(f"\n📈 COMPARISON WITH PREVIOUS RESULTS:")
    print("=" * 40)
    
    baseline_metrics = {
        'P@1': 24.0,
        'P@10': 34.7,
        'R@10': 60.5,
        'MAP': 24.4,
        'Entity F1': 5.1
    }
    
    improvements = []
    for metric in ['P@1', 'P@10', 'R@10', 'MAP']:
        if metric in ir_metrics:
            current = ir_metrics[metric]
            baseline = baseline_metrics[metric]
            improvement = current - baseline
            improvements.append(improvement)
            
            status = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"
            print(f"  {metric:8}: {current:5.1f}% vs {baseline:5.1f}% baseline ({status} {improvement:+5.1f}%)")
    
    # Entity F1 comparison
    if 'Entity F1' in entity_metrics:
        current_ef1 = entity_metrics['Entity F1']
        baseline_ef1 = baseline_metrics['Entity F1']
        ef1_improvement = current_ef1 - baseline_ef1
        improvements.append(ef1_improvement)
        
        status = "📈" if ef1_improvement > 0 else "📉" if ef1_improvement < 0 else "➡️"
        print(f"  Entity F1: {current_ef1:5.1f}% vs {baseline_ef1:5.1f}% baseline ({status} {ef1_improvement:+5.1f}%)")
    
    # Overall assessment
    avg_improvement = np.mean(improvements) if improvements else 0
    print(f"\n🎯 Average Improvement: {avg_improvement:+5.1f}%")
    
    if avg_improvement > 5:
        print("🎉 SIGNIFICANT IMPROVEMENT with better ground truth!")
    elif avg_improvement > 0:
        print("✅ POSITIVE IMPROVEMENT detected")
    else:
        print("📊 Similar performance - ground truth quality matters")
    
    return ir_metrics, entity_metrics, query_type_performance

if __name__ == "__main__":
    test_with_improved_ground_truth()
