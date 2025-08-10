"""
Fixed Academic Data Generator - Uses REAL Neo4j data for benchmarking
"""

import pandas as pd
import numpy as np
import random
import json
import re
from typing import Dict, List, Any
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator

# Query templates using real RAN terminology
QUERY_TEMPLATES = {
    'power_analysis': [
        "Show power consumption data from {table}",
        "Display energy efficiency metrics in {table}",
        "Find power optimization parameters from {table}",
        "Get power measurements for {concept}",
        "Analyze energy consumption patterns in {table}"
    ],
    'frequency_management': [
        "Display frequency allocation data from {table}",
        "Show spectrum utilization for {concept}",
        "Get frequency band information from {table}",
        "Find carrier configuration in {table}",
        "Analyze spectrum efficiency for {concept}"
    ],
    'performance_metrics': [
        "Show throughput data from {table}",
        "Find KPI measurements in {table}",
        "Display performance counters from {table}",
        "Get quality metrics for {concept}",
        "Analyze network performance in {concept}"
    ],
    'cell_configuration': [
        "Show cell parameters from {table}",
        "Get configuration settings for {concept}",
        "Display antenna parameters in {table}",
        "Find cell setup information for {concept}",
        "Show configuration data from {table}"
    ],
    'neighbor_relations': [
        "Get neighbor relations for {concept}",
        "Display adjacency data from {table}",
        "Show handover configuration from {table}",
        "Find neighbor topology in {table}",
        "Analyze neighbor relationships for {concept}"
    ],
    'timing_sync': [
        "Show timing synchronization from {table}",
        "Get synchronization status for {concept}",
        "Display timing data for {concept}",
        "Find sync parameters in {table}",
        "Show synchronization health from {table}"
    ]
}

def get_real_tables_and_concepts(neo4j_integrator):
    """Get actual tables and concepts from Neo4j database"""
    with neo4j_integrator.driver.session() as session:
        # Get all real tables with metadata
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
        
        # Get real semantic categories
        result = session.run("""
            MATCH ()-[r:CONCEPTUAL_GROUP]-()
            WHERE r.semantic_category IS NOT NULL
            RETURN DISTINCT r.semantic_category as category, count(r) as count
            ORDER BY count DESC
            LIMIT 30
        """)
        real_concepts = [record['category'] for record in result]
        
    return real_tables, real_concepts

def categorize_real_table(table_name: str) -> str:
    """Categorize real table based on name patterns"""
    name_lower = table_name.lower()
    
    # RAN-specific categorization based on actual table naming patterns
    if any(keyword in name_lower for keyword in ['power', 'energy', 'consumption']):
        return 'power_analysis'
    elif any(keyword in name_lower for keyword in ['freq', 'spectrum', 'band', 'carrier', 'eutra']):
        return 'frequency_management'
    elif any(keyword in name_lower for keyword in ['performance', 'kpi', 'quality', 'throughput', 'counter']):
        return 'performance_metrics'
    elif any(keyword in name_lower for keyword in ['cell', 'antenna', 'config', 'setup']):
        return 'cell_configuration'
    elif any(keyword in name_lower for keyword in ['neighbor', 'relation', 'handover', 'anr']):
        return 'neighbor_relations'
    elif any(keyword in name_lower for keyword in ['sync', 'timing', 'clock', 'time']):
        return 'timing_sync'
    else:
        return 'general'

def get_related_tables_real(target_table: str, neo4j_integrator, max_related: int = 5) -> list:
    """Get actually related tables from Neo4j relationships"""
    with neo4j_integrator.driver.session() as session:
        result = session.run("""
            MATCH (t1:Table {name: $table_name})-[:HAS_COLUMN]->(c1:Column)
            MATCH (c1)-[r]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
            WHERE type(r) IN ['CONCEPTUAL_GROUP', 'PATTERN_MATCH', 'NAME_SIMILARITY', 'VALUE_OVERLAP']
              AND t1 <> t2
            WITH t2.name as related_table, count(r) as relationship_count
            RETURN related_table
            ORDER BY relationship_count DESC
            LIMIT $max_related
        """, table_name=target_table, max_related=max_related)
        
        return [record['related_table'] for record in result]

def generate_real_ir_ground_truth(neo4j_integrator, num_queries: int = 75) -> pd.DataFrame:
    """Generate IR ground truth using REAL table names and relationships"""
    print("🔍 Generating IR ground truth with REAL Neo4j data...")
    
    real_tables, real_concepts = get_real_tables_and_concepts(neo4j_integrator)
    
    if not real_tables:
        print("❌ ERROR: No tables found in Neo4j database!")
        return pd.DataFrame()
    
    print(f"✅ Found {len(real_tables)} real tables and {len(real_concepts)} concepts")
    
    ir_data = []
    random.seed(42)  # For reproducibility
    
    for i in range(num_queries):
        # Select random category and template
        category = random.choice(list(QUERY_TEMPLATES.keys()))
        template = random.choice(QUERY_TEMPLATES[category])
        
        # Find tables that match this category
        category_tables = [t for t in real_tables if categorize_real_table(t['table_name']) == category]
        
        if not category_tables:
            # Fallback to any table
            target_table_info = random.choice(real_tables)
        else:
            target_table_info = random.choice(category_tables)
        
        target_table = target_table_info['table_name']
        
        # Get real related tables from Neo4j
        related_tables = get_related_tables_real(target_table, neo4j_integrator)
        
        # Select concept (use real concept or fallback)
        concept = random.choice(real_concepts) if real_concepts else target_table.lower()
        
        # Generate query
        if '{table}' in template and '{concept}' in template:
            query = template.format(table=target_table, concept=concept)
        elif '{table}' in template:
            query = template.format(table=target_table)
        elif '{concept}' in template:
            query = template.format(concept=concept)
        else:
            query = template
        
        # Build relevant tables list (target + related)
        all_relevant = [target_table] + related_tables[:4]
        all_relevant = list(dict.fromkeys(all_relevant))  # Remove duplicates, preserve order
        
        ir_data.append({
            'query': query,
            'relevant_tables': ','.join(all_relevant[:5]),
            'primary_table': target_table,
            'category': category,
            'num_relevant': len(all_relevant),
            'sample_columns': ','.join(target_table_info.get('sample_columns', [])[:3])
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{num_queries} queries...")
    
    print(f"✅ Generated {len(ir_data)} IR ground truth entries")
    return pd.DataFrame(ir_data)

def generate_real_nlu_ground_truth(neo4j_integrator, num_samples: int = 100) -> pd.DataFrame:
    """Generate NLU ground truth using real data"""
    print("🔍 Generating NLU ground truth with REAL Neo4j data...")
    
    real_tables, real_concepts = get_real_tables_and_concepts(neo4j_integrator)
    
    nlu_data = []
    random.seed(42)
    
    for i in range(num_samples):
        # Select random table and category
        table_info = random.choice(real_tables)
        table_name = table_info['table_name']
        category = categorize_real_table(table_name)
        
        # If category is 'general', pick a random valid category
        if category == 'general':
            category = random.choice(list(QUERY_TEMPLATES.keys()))
        
        # Generate natural language query
        template = random.choice(QUERY_TEMPLATES[category])
        concept = random.choice(real_concepts) if real_concepts else table_name.lower()
        
        if '{table}' in template and '{concept}' in template:
            query = template.format(table=table_name, concept=concept)
        elif '{table}' in template:
            query = template.format(table=table_name)
        else:
            query = template.format(concept=concept)
        
        # Generate expected answer with real table/column references
        sample_columns = table_info.get('sample_columns', [])[:3]
        related_tables = get_related_tables_real(table_name, neo4j_integrator, 2)
        
        answer_templates = {
            'power_analysis': f"⚡ Power analysis from {table_name} shows consumption patterns across {', '.join(sample_columns)}",
            'frequency_management': f"📡 Frequency data from {table_name} indicates spectrum utilization via {', '.join(sample_columns)}",
            'performance_metrics': f"📈 Performance metrics from {table_name} reveal KPI trends in {', '.join(sample_columns)}",
            'cell_configuration': f"⚙️ Cell configuration from {table_name} contains parameters: {', '.join(sample_columns)}",
            'neighbor_relations': f"🔗 Neighbor relations from {table_name} show connectivity via {', '.join(sample_columns)}",
            'timing_sync': f"⏱️ Timing synchronization from {table_name} tracks accuracy through {', '.join(sample_columns)}"
        }
        
        answer = answer_templates.get(category, f"📋 Data from {table_name} provides insights via {', '.join(sample_columns)}")
        
        if related_tables:
            answer += f" Related tables: {', '.join(related_tables[:2])}"
        
        # Extract entities for evaluation
        entities = [table_name] + sample_columns + related_tables[:2]
        
        nlu_data.append({
            'query': query,
            'answer': answer,
            'entities': ','.join(entities),
            'primary_table': table_name,
            'category': category
        })
        
        if (i + 1) % 20 == 0:
            print(f"  Generated {i + 1}/{num_samples} NLU samples...")
    
    print(f"✅ Generated {len(nlu_data)} NLU ground truth entries")
    return pd.DataFrame(nlu_data)

def export_real_data(neo4j_integrator, output_dir: str = "."):
    """Export real ground truth data for benchmarking"""
    print("🚀 Exporting REAL academic benchmarking data...")
    
    # Generate and export IR ground truth
    ir_df = generate_real_ir_ground_truth(neo4j_integrator, num_queries=75)
    if not ir_df.empty:
        ir_path = f"{output_dir}/sample_ir_ground_truth.csv"
        ir_df[['query', 'relevant_tables']].to_csv(ir_path, index=False)
        print(f"✅ IR ground truth exported: {ir_path}")
    
    # Generate and export NLU ground truth  
    nlu_df = generate_real_nlu_ground_truth(neo4j_integrator, num_samples=100)
    if not nlu_df.empty:
        nlu_path = f"{output_dir}/sample_nlu_ground_truth.csv"
        nlu_df[['query', 'answer', 'entities']].to_csv(nlu_path, index=False)
        print(f"✅ NLU ground truth exported: {nlu_path}")
    
    print("🎯 Ready for realistic benchmarking with actual database content!")
    return ir_df, nlu_df

if __name__ == "__main__":
    # Test with real Neo4j connection
    neo4j_integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    ir_df, nlu_df = export_real_data(neo4j_integrator)
    print(f"\nSample IR queries:")
    for i, row in ir_df.head(3).iterrows():
        print(f"  {row['query']} -> {row['relevant_tables']}")
