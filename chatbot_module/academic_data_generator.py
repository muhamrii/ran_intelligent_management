#!/usr/bin/env python3
"""
Academic Benchmarking Data Generator

This script generates high-quality sample ground truth data for academic IR and NLU 
benchmarking based on the actual knowledge graph structure and domain knowledge.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import re

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

# Configuration
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "neo4j"

# RAN domain-specific expert query templates (more human telco engineer phrasing)
QUERY_TEMPLATES = {
    'power_analysis': [
        "Break down power consumption trend for {table} focusing on peak vs off-peak windows",
        "Correlate {table}.{column} with efficiency indicators to spot abnormal drain",
        "Where are the largest energy optimization gaps according to {concept}?",
        "Summarize average vs 95th percentile power draw from {table}",
        "Identify cells with sustained high load but poor power efficiency using {table}"
    ],
    'frequency_management': [
        "List spectrum utilization anomalies from {table} compared to neighbor allocations",
        "Check which carriers in {table}.{column} are under-utilized",
        "Assess frequency re-use efficiency leveraging {concept}",
        "Provide band occupancy summary from {table}",
        "Highlight potential interference bands inferred from {table}"
    ],
    'performance_metrics': [
        "Show throughput degradation periods found in {table}",
        "Compare KPI variance using {table}.{column} day vs night",
        "Diagnose performance bottlenecks leveraging {concept}",
        "Provide composite performance index from {table}",
        "Surface cells with chronic KPI instability based on {table}"
    ],
    'cell_configuration': [
        "List configuration parameters in {table} driving recent KPI shifts",
        "Extract recent changes to {table}.{column} and correlate with handover success",
        "Summarize antenna/tilt settings from {table}",
        "Identify outlier parameter values in {table}",
        "What config deltas precede performance regressions using {concept}?"
    ],
    'neighbor_relations': [
        "Evaluate neighbor handover imbalance based on {table}",
        "Find excessive handover retry patterns referencing {table}.{column}",
        "Summarize adjacency density from {table}",
        "Detect missing neighbor definitions leveraging {concept}",
        "Correlate failed handovers with neighbor topology from {table}"
    ],
    'timing_sync': [
        "Identify cells with sync drift using {table}",
        "List unstable timing sources referencing {table}.{column}",
        "Assess synchronization health via {concept}",
        "Provide sync accuracy distribution from {table}",
        "Relate timing offsets to performance counters using {table}"
    ]
}

# Common RAN table patterns
RAN_TABLE_PATTERNS = {
    'power': ['power', 'energy', 'consumption', 'dbm', 'watt'],
    'frequency': ['frequency', 'freq', 'spectrum', 'band', 'carrier', 'eutra'],
    'performance': ['throughput', 'kpi', 'quality', 'performance', 'metric'],
    'cell': ['cell', 'sector', 'site', 'antenna', 'config'],
    'neighbor': ['neighbor', 'relation', 'handover', 'adjacency'],
    'sync': ['sync', 'timing', 'synchronization', 'time'],
    'measurement': ['measurement', 'monitor', 'report', 'sample'],
    'optimization': ['optimization', 'tuning', 'adjustment', 'parameter']
}

def categorize_table(table_name: str, table_info: dict = None) -> str:
    """Categorize table based on name and content"""
    name_lower = table_name.lower()
    
    for category, keywords in RAN_TABLE_PATTERNS.items():
        if any(keyword in name_lower for keyword in keywords):
            return category
    
    return 'general'

def find_related_tables(target_table: str, all_tables: list, max_related: int = 5) -> list:
    """Find tables related to the target table"""
    category = categorize_table(target_table)
    related = []
    
    for table in all_tables:
        if table['table_name'] != target_table:
            if categorize_table(table['table_name']) == category:
                related.append(table['table_name'])
    
    # Add some cross-category relationships for realism
    cross_category_maps = {
        'power': ['performance', 'cell'],
        'frequency': ['performance', 'cell'],
        'cell': ['neighbor', 'performance'],
        'performance': ['power', 'frequency']
    }
    
    if category in cross_category_maps:
        for cross_cat in cross_category_maps[category]:
            for table in all_tables:
                if categorize_table(table['table_name']) == cross_cat:
                    related.append(table['table_name'])
    
    return list(set(related))[:max_related]

def generate_ir_ground_truth(neo4j_integrator, num_queries: int = 50) -> pd.DataFrame:
    """Generate IR ground truth data using REAL table names from Neo4j"""
    ir_data = []
    
    # Get REAL tables and their metadata from Neo4j
    with neo4j_integrator.driver.session() as session:
        # Get all tables with column information
        result = session.run("""
            MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
            WITH t, collect(c.name) as columns
            RETURN t.name as table_name, 
                   t.row_count as row_count,
                   t.column_count as column_count,
                   columns[0..10] as sample_columns
            ORDER BY t.row_count DESC
        """)
        real_tables = [dict(record) for record in result]
        
        # Get semantic categories that actually exist
        result = session.run("""
            MATCH ()-[r:CONCEPTUAL_GROUP]-()
            WHERE r.semantic_category IS NOT NULL
            RETURN DISTINCT r.semantic_category as category, count(r) as count
            ORDER BY count DESC
            LIMIT 20
        """)
        real_concepts = [record['category'] for record in result]
    
    if not real_tables:
        print("WARNING: No tables found in Neo4j database!")
        return pd.DataFrame()
    
    print(f"Found {len(real_tables)} real tables and {len(real_concepts)} concepts")
    
    random.seed(42)  # For reproducibility
    
    for i in range(num_queries):
        # Select random category and template
        category = random.choice(list(QUERY_TEMPLATES.keys()))
        template = random.choice(QUERY_TEMPLATES[category])
        
        # Select target table based on category from REAL tables
        category_pattern = category.split('_')[0] if '_' in category else category
        relevant_tables = []
        
        # Find REAL tables matching the category
        for table in real_tables:
            table_name = table['table_name']
            table_category = categorize_table(table_name)
            if (table_category == category_pattern or 
                category_pattern in table_name.lower() or
                any(keyword in table_name.lower() 
                    for keyword in RAN_TABLE_PATTERNS.get(category_pattern, []))):
                relevant_tables.append(table)
        
        if not relevant_tables:
            # Fallback to random table
            target_table = random.choice(tables)['table_name']
            relevant_tables = [target_table]
        else:
            target_table = random.choice(relevant_tables)
        
        # Generate query
        if '{table}' in template and '{column}' not in template:
            query = template.format(table=target_table)
        elif '{concept}' in template and '{column}' not in template:
            concept = random.choice(concepts) if concepts else target_table
            query = template.format(concept=concept)
            # Find tables related to this concept
            concept_tables = []
            for table in tables:
                if (concept.lower() in table['table_name'].lower() or
                    any(word in table['table_name'].lower() for word in concept.lower().split())):
                    concept_tables.append(table['table_name'])
            relevant_tables = concept_tables if concept_tables else relevant_tables
        elif '{column}' in template:
            query = template.replace('.{column}', '').format(table=target_table)
        else:
            query = template
        
        # Find additional related tables
        related_tables = find_related_tables(target_table, tables)
        all_relevant = list(set([target_table] + relevant_tables[:2] + related_tables[:3]))
        
        ir_data.append({
            'query': query,
            'relevant_tables': ','.join(all_relevant[:5]),  # Limit to top 5
            'primary_table': target_table,
            'category': category,
            'num_relevant': len(all_relevant),
            'related_tables': ','.join(related_tables),
        })
    
    return pd.DataFrame(ir_data)

def generate_expected_answer(query: str, primary_table: str, category: str, sample_columns: List[str] | None=None, related: List[str] | None=None) -> str:
    """Generate expert-style expected answer referencing tables, columns and relationships."""
    sample_columns = sample_columns or []
    related = related or []
    cols_fragment = (", ".join(sample_columns[:3]) + (" …" if len(sample_columns) > 3 else "")) if sample_columns else "key KPIs"
    related_fragment = ("; correlated with " + ", ".join(related[:2])) if related else ""
    base = {
        'power_analysis': f"⚡ Power profile from {primary_table} leveraging {cols_fragment}{related_fragment}. Focus on load vs efficiency anomalies.",
        'frequency_management': f"📡 Spectrum utilization from {primary_table} using {cols_fragment}{related_fragment}. Highlights reuse and under-utilized carriers.",
        'performance_metrics': f"📈 Performance composite from {primary_table} summarizing {cols_fragment}{related_fragment}. Tracks throughput and quality variance.",
        'cell_configuration': f"⚙️ Configuration state from {primary_table} (parameters: {cols_fragment}){related_fragment}. Identifies outliers impacting KPIs.",
        'neighbor_relations': f"🔗 Neighbor topology from {primary_table}{related_fragment}. Evaluates handover balance and adjacency density.",
        'timing_sync': f"⏱️ Sync health from {primary_table} via {cols_fragment}{related_fragment}. Flags drift and unstable sources."
    }
    return base.get(category, f"📋 Insights from {primary_table} referencing {cols_fragment}{related_fragment}.")

def extract_entities_from_query(query: str, primary_table: str) -> list:
    """Extract key entities that should appear in responses"""
    entities = [primary_table]
    
    # Common RAN entities
    ran_entities = {
        'power': ['consumedEnergyMeasurement', 'powerOptimization', 'energyEfficiency'],
        'frequency': ['EUtranFrequency', 'freqBand', 'carrierFreq', 'spectrumAllocation'],
        'performance': ['throughputMeasurement', 'qualityIndicator', 'performanceCounter'],
        'cell': ['cellConfiguration', 'antennaConfig', 'cellParameters'],
        'neighbor': ['neighborRelation', 'handoverConfig', 'adjacencyList'],
        'sync': ['synchronizationConfig', 'timingAccuracy', 'syncStatus']
    }
    
    query_lower = query.lower()
    for category, entity_list in ran_entities.items():
        if category in query_lower:
            entities.extend(random.sample(entity_list, min(2, len(entity_list))))
    
    # Extract table.column patterns from query
    table_column_pattern = r'([A-Za-z][A-Za-z0-9_]*)\\.[A-Za-z][A-Za-z0-9_]*'
    matches = re.findall(table_column_pattern, query)
    entities.extend(matches)
    
    return list(set(entities))[:5]  # Limit to 5 entities

def generate_nlu_ground_truth(ir_df: pd.DataFrame) -> pd.DataFrame:
    """Generate NLU ground truth based on IR queries"""
    nlu_data = []
    
    for _, row in ir_df.iterrows():
        query = row['query']
        primary_table = row['primary_table']
        category = row['category']
        
        # Derive sample columns heuristically from query patterns
        sample_columns = []
        for m in re.findall(r"[A-Za-z][A-Za-z0-9_]*\.([A-Za-z][A-Za-z0-9_]*)", query):
            sample_columns.append(m)
        related = row.get('related_tables','').split(',') if isinstance(row.get('related_tables'), str) else []
        answer = generate_expected_answer(query, primary_table, category, sample_columns, related)
        
        # Extract expected entities
        entities = extract_entities_from_query(query, primary_table)
        
        nlu_data.append({
            'query': query,
            'answer': answer,
            'entities': ','.join(entities),
            'intent': category,
            'complexity': 'medium' if len(entities) > 3 else 'simple',
            'domain': 'RAN'
        })
    
    return pd.DataFrame(nlu_data)

def generate_advanced_ir_samples(schema: dict, num_samples: int = 25) -> pd.DataFrame:
    """Generate more sophisticated IR samples using actual table relationships"""
    advanced_samples = []
    
    # Advanced query patterns
    advanced_patterns = [
        "Correlate {table1} with {table2} for optimization insights",
        "Cross-reference {table1} and {table2} performance data",
        "Compare metrics between {table1} and {table2}",
        "Find relationships linking {table1} to {table2}",
        "Analyze joint patterns in {table1} and {table2}",
        "Show dependencies between {table1} and {table2}"
    ]
    
    tables = [t['table_name'] for t in schema.get('tables', {}).get('sample_tables', [])]
    
    for i in range(num_samples):
        # Random pair
        table1, table2 = random.sample(tables, 2)
        
        pattern = random.choice(advanced_patterns)
        query = pattern.format(table1=table1, table2=table2)
        
        # Find related tables based on categories
        cat1 = categorize_table(table1)
        cat2 = categorize_table(table2)
        
        relevant_tables = [table1, table2]
        
        # Add tables from same categories
        for table in tables:
            if (categorize_table(table) in [cat1, cat2] and 
                table not in relevant_tables and 
                len(relevant_tables) < 6):
                relevant_tables.append(table)
        
        advanced_samples.append({
            'query': query,
            'relevant_tables': ','.join(relevant_tables),
            'primary_table': table1,
            'category': 'multi_table_analysis',
            'num_relevant': len(relevant_tables)
        })
    
    return pd.DataFrame(advanced_samples)

def main():
    print("🚀 Starting Academic Benchmarking Data Generation...")
    
    # Initialize connections
    try:
        integrator = RANNeo4jIntegrator(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("Please ensure Neo4j is running and accessible")
        return
    
    # Get schema overview
    print("📊 Analyzing knowledge graph schema...")
    try:
        # Try different method names
        if hasattr(integrator, 'get_schema_overview'):
            schema = integrator.get_schema_overview()
        elif hasattr(integrator, 'get_schema_info'):
            schema = integrator.get_schema_info()
        else:
            # Create basic schema from available methods
            schema = {
                'tables': {'sample_tables': [], 'total_tables': 0},
                'concepts': [],
                'relationships': []
            }
            # Get basic table info if available
            try:
                tables_result = integrator.driver.session().run("SHOW TABLES").data()
                sample_tables = [{'table_name': row.get('name', row.get('table', '')), 'row_count': 100} 
                               for row in tables_result[:50]]
                schema['tables']['sample_tables'] = sample_tables
                schema['tables']['total_tables'] = len(sample_tables)
            except Exception as e:
                print(f"Warning: Could not get table info: {e}")
                # Use default sample data
                sample_tables = [
                    {'table_name': 'EUtranFrequency', 'row_count': 150},
                    {'table_name': 'ConsumedEnergyMeasurement', 'row_count': 200},
                    {'table_name': 'CellConfiguration', 'row_count': 180},
                    {'table_name': 'NeighborRelation', 'row_count': 300},
                    {'table_name': 'PowerOptimization', 'row_count': 120},
                    {'table_name': 'PerformanceCounter', 'row_count': 250},
                    {'table_name': 'SynchronizationConfig', 'row_count': 90},
                    {'table_name': 'AntennaConfig', 'row_count': 160},
                    {'table_name': 'HandoverConfig', 'row_count': 140},
                    {'table_name': 'QualityMeasurement', 'row_count': 220}
                ]
                schema['tables']['sample_tables'] = sample_tables
                schema['tables']['total_tables'] = len(sample_tables)
                
            # Add some concepts
            schema['concepts'] = [
                'power_optimization', 'frequency_management', 'cell_configuration',
                'neighbor_relations', 'performance_monitoring', 'synchronization',
                'antenna_optimization', 'handover_management', 'quality_assessment'
            ]
    except Exception as e:
        print(f"Warning: Schema analysis failed: {e}")
        # Use fallback schema
        schema = {
            'tables': {
                'sample_tables': [
                    {'table_name': 'EUtranFrequency', 'row_count': 150},
                    {'table_name': 'ConsumedEnergyMeasurement', 'row_count': 200},
                    {'table_name': 'CellConfiguration', 'row_count': 180},
                    {'table_name': 'NeighborRelation', 'row_count': 300},
                    {'table_name': 'PowerOptimization', 'row_count': 120},
                    {'table_name': 'PerformanceCounter', 'row_count': 250},
                    {'table_name': 'SynchronizationConfig', 'row_count': 90},
                    {'table_name': 'AntennaConfig', 'row_count': 160},
                    {'table_name': 'HandoverConfig', 'row_count': 140},
                    {'table_name': 'QualityMeasurement', 'row_count': 220}
                ],
                'total_tables': 10
            },
            'concepts': [
                'power_optimization', 'frequency_management', 'cell_configuration',
                'neighbor_relations', 'performance_monitoring', 'synchronization'
            ],
            'relationships': []
        }
    
    print(f"   Tables: {schema.get('tables', {}).get('total_tables', 0)}")
    print(f"   Concepts: {len(schema.get('concepts', []))}")
    
    # Generate IR ground truth
    print("🔍 Generating IR ground truth data...")
    ir_df = generate_ir_ground_truth(schema, 50)
    print(f"   Generated {len(ir_df)} basic IR queries")
    
    # Generate advanced IR samples
    print("🎯 Generating advanced IR samples...")
    advanced_ir_df = generate_advanced_ir_samples(schema, 25)
    print(f"   Generated {len(advanced_ir_df)} advanced IR queries")
    
    # Combine IR data
    combined_ir_df = pd.concat([ir_df, advanced_ir_df], ignore_index=True)
    print(f"   Total IR samples: {len(combined_ir_df)}")
    
    # Generate NLU ground truth
    print("💬 Generating NLU ground truth data...")
    nlu_df = generate_nlu_ground_truth(ir_df)
    advanced_nlu_df = generate_nlu_ground_truth(advanced_ir_df)
    combined_nlu_df = pd.concat([nlu_df, advanced_nlu_df], ignore_index=True)
    print(f"   Generated {len(combined_nlu_df)} NLU examples")
    
    # Save files
    print("💾 Saving generated data...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.dirname(output_dir)  # Go to project root
    
    # Save IR ground truth
    ir_output_path = os.path.join(output_dir, 'sample_ir_ground_truth.csv')
    ir_export_df = combined_ir_df[['query', 'relevant_tables']].copy()
    ir_export_df.to_csv(ir_output_path, index=False)
    print(f"   ✅ IR ground truth: {ir_output_path} ({len(ir_export_df)} samples)")
    
    # Save NLU ground truth  
    nlu_output_path = os.path.join(output_dir, 'sample_nlu_ground_truth.csv')
    nlu_export_df = combined_nlu_df[['query', 'answer', 'entities']].copy()
    nlu_export_df.to_csv(nlu_output_path, index=False)
    print(f"   ✅ NLU ground truth: {nlu_output_path} ({len(nlu_export_df)} samples)")
    
    # Save detailed analysis
    detailed_ir_path = os.path.join(output_dir, 'detailed_ir_analysis.csv')
    combined_ir_df.to_csv(detailed_ir_path, index=False)
    print(f"   ✅ Detailed IR analysis: {detailed_ir_path}")
    
    detailed_nlu_path = os.path.join(output_dir, 'detailed_nlu_analysis.csv')
    combined_nlu_df.to_csv(detailed_nlu_path, index=False)
    print(f"   ✅ Detailed NLU analysis: {detailed_nlu_path}")
    
    # Summary
    print(f"\n📈 Generation Summary:")
    print(f"   • IR queries: {len(combined_ir_df)} ({combined_ir_df['query'].nunique()} unique)")
    print(f"   • NLU examples: {len(combined_nlu_df)}")
    print(f"   • Categories: {len(combined_ir_df['category'].unique())}")
    print(f"   • Avg relevant tables: {combined_ir_df['num_relevant'].mean():.1f}")
    
    print(f"\n🎯 Quality Metrics:")
    print(f"   • Query uniqueness: {100*combined_ir_df['query'].nunique()/len(combined_ir_df):.1f}%")
    print(f"   • Avg entities/query: {combined_nlu_df['entities'].apply(lambda x: len(x.split(',')) if x else 0).mean():.1f}")
    print(f"   • Complex queries: {(combined_nlu_df['complexity'] == 'medium').sum()}/{len(combined_nlu_df)}")
    
    print(f"\n✅ Data generation complete!")
    print(f"   Use the generated sample files in the Academic Benchmarking UI tab")

if __name__ == "__main__":
    main()
