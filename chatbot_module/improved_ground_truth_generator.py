#!/usr/bin/env python3
"""
Improved Ground Truth Generator
Creates high-quality benchmark data aligned with actual KG structure and chatbot outputs
"""

import pandas as pd
import numpy as np
import random
import json
import re
import sys
import os
from typing import Dict, List, Any, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator

class ImprovedGroundTruthGenerator:
    """Generate high-quality ground truth data aligned with actual KG and expected outputs"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.actual_tables = []
        self.actual_columns = {}
        self.semantic_categories = []
        self.table_relationships = {}
        
        # Load actual KG structure
        self._load_kg_structure()
        
        # Enhanced query templates based on actual data patterns
        self.query_templates = {
            'explicit_table': [
                "Show me {table} table information",
                "Get details about {table} table",
                "Display {table} table structure",
                "Find information in {table} table",
                "What is in the {table} table?"
            ],
            'domain_specific': [
                "Find {domain} related tables",
                "Show {domain} data and metrics",
                "Get {domain} configuration parameters",
                "Display {domain} performance data",
                "Analyze {domain} information"
            ],
            'entity_focused': [
                "Show tables with {entity} information",
                "Find {entity} measurements and data",
                "Get {entity} configuration details",
                "Display {entity} performance metrics",
                "Analyze {entity} parameters"
            ],
            'column_specific': [
                "Find tables containing {column} column",
                "Show data for {column} field",
                "Get {column} measurements",
                "Display {column} values",
                "Analyze {column} information"
            ],
            'relationship_based': [
                "Show tables related to {table}",
                "Find connections to {table}",
                "Get tables linked with {table}",
                "Display relationships with {table}",
                "Analyze connections from {table}"
            ]
        }
        
        # Domain-specific entities based on actual RAN data
        self.domain_entities = {
            'power': ['power', 'energy', 'consumption', 'efficiency', 'dbm', 'transmission'],
            'frequency': ['frequency', 'spectrum', 'band', 'carrier', 'eutra', 'bandwidth'],
            'timing': ['timing', 'synchronization', 'clock', 'sync', 'ptp', 'time'],
            'performance': ['performance', 'throughput', 'latency', 'kpi', 'quality', 'metric'],
            'cell': ['cell', 'sector', 'antenna', 'enode', 'base', 'node'],
            'network': ['neighbor', 'handover', 'mobility', 'anr', 'relation', 'adjacency'],
            'configuration': ['config', 'parameter', 'setting', 'threshold', 'setup'],
            'measurement': ['rsrp', 'rsrq', 'sinr', 'cqi', 'bler', 'counter']
        }
    
    def _load_kg_structure(self):
        """Load actual knowledge graph structure for realistic ground truth"""
        with self.integrator.driver.session() as session:
            # Get all tables with their metadata
            result = session.run("""
                MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                WITH t, collect(c.name) as columns, count(c) as col_count
                RETURN t.name as table_name, 
                       coalesce(t.row_count, 0) as row_count,
                       col_count as column_count,
                       columns as all_columns
                ORDER BY t.name
            """)
            
            for record in result:
                table_name = record['table_name']
                self.actual_tables.append(table_name)
                self.actual_columns[table_name] = record['all_columns']
            
            print(f"✅ Loaded {len(self.actual_tables)} actual tables")
            
            # Get semantic categories from relationships
            result = session.run("""
                MATCH ()-[r:CONCEPTUAL_GROUP]-()
                WHERE r.semantic_category IS NOT NULL
                RETURN DISTINCT r.semantic_category as category
                ORDER BY category
            """)
            
            self.semantic_categories = [record['category'] for record in result]
            print(f"✅ Loaded {len(self.semantic_categories)} semantic categories")
            
            # Get table relationships for relationship-based queries
            result = session.run("""
                MATCH (t1:Table)-[r]-(t2:Table)
                WHERE type(r) IN ['NAME_SIMILARITY', 'CONCEPTUAL_GROUP']
                RETURN t1.name as table1, t2.name as table2, type(r) as rel_type, 
                       coalesce(r.similarity_score, r.similarity, 0.0) as score
                ORDER BY score DESC
                LIMIT 200
            """)
            
            for record in result:
                t1, t2 = record['table1'], record['table2']
                if t1 not in self.table_relationships:
                    self.table_relationships[t1] = []
                if t2 not in self.table_relationships:
                    self.table_relationships[t2] = []
                
                self.table_relationships[t1].append({
                    'table': t2, 'type': record['rel_type'], 'score': record['score']
                })
                self.table_relationships[t2].append({
                    'table': t1, 'type': record['rel_type'], 'score': record['score']
                })
            
            print(f"✅ Loaded relationships for {len(self.table_relationships)} tables")
    
    def _categorize_table(self, table_name: str) -> str:
        """Categorize table based on name and content"""
        name_lower = table_name.lower()
        
        # Enhanced categorization based on actual table names
        if any(kw in name_lower for kw in self.domain_entities['power']):
            return 'power'
        elif any(kw in name_lower for kw in self.domain_entities['frequency']):
            return 'frequency'
        elif any(kw in name_lower for kw in self.domain_entities['timing']):
            return 'timing'
        elif any(kw in name_lower for kw in self.domain_entities['performance']):
            return 'performance'
        elif any(kw in name_lower for kw in self.domain_entities['cell']):
            return 'cell'
        elif any(kw in name_lower for kw in self.domain_entities['network']):
            return 'network'
        elif any(kw in name_lower for kw in self.domain_entities['configuration']):
            return 'configuration'
        elif any(kw in name_lower for kw in self.domain_entities['measurement']):
            return 'measurement'
        else:
            return 'general'
    
    def _extract_entities_from_table(self, table_name: str) -> List[str]:
        """Extract meaningful entities from table name and columns"""
        entities = []
        
        # Extract from table name
        table_lower = table_name.lower()
        for domain, keywords in self.domain_entities.items():
            for keyword in keywords:
                if keyword in table_lower:
                    entities.append(keyword)
        
        # Extract from column names
        if table_name in self.actual_columns:
            for column in self.actual_columns[table_name][:10]:  # Limit to avoid too many
                column_lower = column.lower()
                for domain, keywords in self.domain_entities.items():
                    for keyword in keywords:
                        if keyword in column_lower and keyword not in entities:
                            entities.append(keyword)
        
        return entities[:5]  # Limit to 5 most relevant entities
    
    def generate_realistic_ir_ground_truth(self, num_queries: int = 100) -> pd.DataFrame:
        """Generate realistic IR ground truth aligned with actual KG structure"""
        ground_truth_data = []
        
        # Distribution of query types (aligned with actual usage patterns)
        query_type_weights = {
            'explicit_table': 0.4,      # 40% - Most common in real usage
            'domain_specific': 0.25,    # 25% - Domain-based queries
            'entity_focused': 0.20,     # 20% - Entity extraction queries
            'column_specific': 0.10,    # 10% - Column-specific queries
            'relationship_based': 0.05  # 5% - Relationship queries
        }
        
        query_types = list(query_type_weights.keys())
        weights = list(query_type_weights.values())
        
        for i in range(num_queries):
            query_type = np.random.choice(query_types, p=weights)
            
            if query_type == 'explicit_table':
                # Generate explicit table queries (highest precision expected)
                table = random.choice(self.actual_tables)
                template = random.choice(self.query_templates['explicit_table'])
                query = template.format(table=table)
                
                # Ground truth: the exact table should be #1 result
                ground_truth = {
                    'query': query,
                    'expected_table_1': table,
                    'expected_table_2': '',
                    'expected_table_3': '',
                    'query_type': 'explicit_table',
                    'domain': self._categorize_table(table),
                    'entities': ', '.join(self._extract_entities_from_table(table)),
                    'confidence': 1.0
                }
                
            elif query_type == 'domain_specific':
                # Generate domain-based queries
                domain = random.choice(list(self.domain_entities.keys()))
                template = random.choice(self.query_templates['domain_specific'])
                query = template.format(domain=domain)
                
                # Find tables in this domain
                domain_tables = [t for t in self.actual_tables 
                               if self._categorize_table(t) == domain]
                
                if len(domain_tables) >= 3:
                    selected_tables = random.sample(domain_tables, 3)
                elif len(domain_tables) >= 1:
                    selected_tables = domain_tables + [''] * (3 - len(domain_tables))
                else:
                    # Fallback to any tables containing domain keywords
                    selected_tables = [t for t in self.actual_tables 
                                     if any(kw in t.lower() for kw in self.domain_entities[domain])][:3]
                    selected_tables += [''] * (3 - len(selected_tables))
                
                ground_truth = {
                    'query': query,
                    'expected_table_1': selected_tables[0] if len(selected_tables) > 0 else '',
                    'expected_table_2': selected_tables[1] if len(selected_tables) > 1 else '',
                    'expected_table_3': selected_tables[2] if len(selected_tables) > 2 else '',
                    'query_type': 'domain_specific',
                    'domain': domain,
                    'entities': ', '.join(self.domain_entities[domain][:3]),
                    'confidence': 0.8
                }
                
            elif query_type == 'entity_focused':
                # Generate entity-focused queries
                domain = random.choice(list(self.domain_entities.keys()))
                entity = random.choice(self.domain_entities[domain])
                template = random.choice(self.query_templates['entity_focused'])
                query = template.format(entity=entity)
                
                # Find tables containing this entity
                entity_tables = []
                for table in self.actual_tables:
                    if entity in table.lower():
                        entity_tables.append(table)
                    elif table in self.actual_columns:
                        for column in self.actual_columns[table]:
                            if entity in column.lower():
                                entity_tables.append(table)
                                break
                
                # Remove duplicates and limit
                entity_tables = list(set(entity_tables))[:3]
                entity_tables += [''] * (3 - len(entity_tables))
                
                ground_truth = {
                    'query': query,
                    'expected_table_1': entity_tables[0],
                    'expected_table_2': entity_tables[1],
                    'expected_table_3': entity_tables[2],
                    'query_type': 'entity_focused',
                    'domain': domain,
                    'entities': entity,
                    'confidence': 0.7
                }
                
            elif query_type == 'column_specific':
                # Generate column-specific queries
                table = random.choice(self.actual_tables)
                if table in self.actual_columns and self.actual_columns[table]:
                    column = random.choice(self.actual_columns[table])
                    template = random.choice(self.query_templates['column_specific'])
                    query = template.format(column=column)
                    
                    # Find tables with similar columns
                    similar_tables = [table]
                    for other_table in self.actual_tables:
                        if other_table != table and other_table in self.actual_columns:
                            if column in self.actual_columns[other_table]:
                                similar_tables.append(other_table)
                    
                    similar_tables = similar_tables[:3]
                    similar_tables += [''] * (3 - len(similar_tables))
                    
                    ground_truth = {
                        'query': query,
                        'expected_table_1': similar_tables[0],
                        'expected_table_2': similar_tables[1],
                        'expected_table_3': similar_tables[2],
                        'query_type': 'column_specific',
                        'domain': self._categorize_table(table),
                        'entities': column,
                        'confidence': 0.9
                    }
                else:
                    continue  # Skip if no columns available
                    
            elif query_type == 'relationship_based':
                # Generate relationship-based queries
                if self.table_relationships:
                    base_table = random.choice(list(self.table_relationships.keys()))
                    template = random.choice(self.query_templates['relationship_based'])
                    query = template.format(table=base_table)
                    
                    # Get related tables
                    related = self.table_relationships[base_table][:3]
                    related_tables = [r['table'] for r in related]
                    related_tables += [''] * (3 - len(related_tables))
                    
                    ground_truth = {
                        'query': query,
                        'expected_table_1': related_tables[0] if len(related_tables) > 0 else '',
                        'expected_table_2': related_tables[1] if len(related_tables) > 1 else '',
                        'expected_table_3': related_tables[2] if len(related_tables) > 2 else '',
                        'query_type': 'relationship_based',
                        'domain': self._categorize_table(base_table),
                        'entities': ', '.join(self._extract_entities_from_table(base_table)),
                        'confidence': 0.6
                    }
                else:
                    continue  # Skip if no relationships available
            
            ground_truth_data.append(ground_truth)
        
        df = pd.DataFrame(ground_truth_data)
        print(f"✅ Generated {len(df)} realistic IR ground truth queries")
        print(f"📊 Query type distribution:")
        print(df['query_type'].value_counts())
        
        return df
    
    def generate_realistic_nlu_ground_truth(self, num_queries: int = 100) -> pd.DataFrame:
        """Generate realistic NLU ground truth for intent classification"""
        nlu_data = []
        
        # Intent distribution based on actual usage
        intent_weights = {
            'table_details': 0.25,
            'domain_inquiry': 0.20,
            'performance_analysis': 0.15,
            'configuration_query': 0.15,
            'relationship_query': 0.10,
            'schema_query': 0.10,
            'list_query': 0.05
        }
        
        intents = list(intent_weights.keys())
        weights = list(intent_weights.values())
        
        for i in range(num_queries):
            intent = np.random.choice(intents, p=weights)
            
            if intent == 'table_details':
                table = random.choice(self.actual_tables)
                queries = [
                    f"Tell me about {table} table",
                    f"What information is in {table}?",
                    f"Describe {table} table structure",
                    f"Show me details of {table}",
                    f"What columns does {table} have?"
                ]
                query = random.choice(queries)
                
            elif intent == 'domain_inquiry':
                domain = random.choice(list(self.domain_entities.keys()))
                queries = [
                    f"Show me {domain} related information",
                    f"Find {domain} data in the database",
                    f"What {domain} metrics are available?",
                    f"Get {domain} configuration details",
                    f"Display {domain} performance data"
                ]
                query = random.choice(queries)
                
            elif intent == 'performance_analysis':
                queries = [
                    "Show performance metrics and KPIs",
                    "Find throughput and latency data",
                    "Get quality measurements",
                    "Display network performance counters",
                    "Analyze system efficiency metrics"
                ]
                query = random.choice(queries)
                
            elif intent == 'configuration_query':
                queries = [
                    "Show configuration parameters",
                    "Get setup and threshold settings",
                    "Find antenna configuration data",
                    "Display cell parameter settings",
                    "Show network configuration details"
                ]
                query = random.choice(queries)
                
            elif intent == 'relationship_query':
                table = random.choice(self.actual_tables)
                queries = [
                    f"What tables are related to {table}?",
                    f"Show connections with {table}",
                    f"Find similar tables to {table}",
                    f"Get neighbor relationships for {table}",
                    f"Display linked data with {table}"
                ]
                query = random.choice(queries)
                
            elif intent == 'schema_query':
                queries = [
                    "Show database schema overview",
                    "Get table structure summary",
                    "Display all available tables",
                    "Show column relationships",
                    "Get database statistics"
                ]
                query = random.choice(queries)
                
            elif intent == 'list_query':
                domain = random.choice(list(self.domain_entities.keys()))
                queries = [
                    f"List all {domain} tables",
                    f"Show available {domain} data",
                    f"Get all {domain} related information",
                    f"Display {domain} table names",
                    f"Find {domain} databases"
                ]
                query = random.choice(queries)
            
            nlu_data.append({
                'query': query,
                'intent': intent,
                'confidence': random.uniform(0.8, 1.0)
            })
        
        df = pd.DataFrame(nlu_data)
        print(f"✅ Generated {len(df)} realistic NLU ground truth queries")
        print(f"📊 Intent distribution:")
        print(df['intent'].value_counts())
        
        return df
    
    def save_ground_truth(self, ir_df: pd.DataFrame, nlu_df: pd.DataFrame, 
                         ir_filename: str = "improved_ir_ground_truth.csv",
                         nlu_filename: str = "improved_nlu_ground_truth.csv"):
        """Save ground truth data to CSV files"""
        ir_df.to_csv(ir_filename, index=False)
        nlu_df.to_csv(nlu_filename, index=False)
        
        print(f"💾 Saved IR ground truth: {ir_filename}")
        print(f"💾 Saved NLU ground truth: {nlu_filename}")
        
        return ir_filename, nlu_filename

def main():
    """Generate improved ground truth data"""
    print("🚀 Improved Ground Truth Generator")
    print("=" * 50)
    
    # Connect to Neo4j
    try:
        integrator = RANNeo4jIntegrator(
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j", 
            neo4j_password="ranqarag#1"
        )
        print("✅ Connected to Neo4j")
    except Exception as e:
        print(f"❌ Failed to connect to Neo4j: {e}")
        return
    
    # Initialize generator
    generator = ImprovedGroundTruthGenerator(integrator)
    
    # Generate ground truth data
    print("\n📊 Generating realistic ground truth data...")
    
    ir_df = generator.generate_realistic_ir_ground_truth(num_queries=150)
    nlu_df = generator.generate_realistic_nlu_ground_truth(num_queries=100)
    
    # Save to files
    ir_file, nlu_file = generator.save_ground_truth(ir_df, nlu_df)
    
    # Show sample data
    print(f"\n📋 Sample IR Ground Truth:")
    print(ir_df[['query', 'expected_table_1', 'query_type', 'domain']].head())
    
    print(f"\n📋 Sample NLU Ground Truth:")
    print(nlu_df[['query', 'intent']].head())
    
    print(f"\n🎯 Ground Truth Summary:")
    print(f"  IR Queries: {len(ir_df)}")
    print(f"  NLU Queries: {len(nlu_df)}")
    print(f"  Total Actual Tables: {len(generator.actual_tables)}")
    print(f"  Total Semantic Categories: {len(generator.semantic_categories)}")
    
    print(f"\n✅ Improved ground truth generation completed!")
    print(f"Files saved: {ir_file}, {nlu_file}")

if __name__ == "__main__":
    main()
