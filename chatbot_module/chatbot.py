"""
Chatbot Module
Contains all functions related to information retrieval and answer generation
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Any
import logging
import json
from datetime import datetime

class RANNERGenerator:
    """Generate NER training data from Neo4j graph"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.entity_types = {
            'TABLE_NAME': 'MATCH (t:Table) RETURN t.name as entity',
            'COLUMN_NAME': 'MATCH (c:Column) RETURN c.name as entity',
            'CONCEPT_NAME': 'MATCH (con:Concept) RETURN con.name as entity',
            'DATA_TYPE': 'MATCH (c:Column) RETURN DISTINCT c.data_type as entity'
        }
    
    def extract_entities_from_graph(self) -> Dict[str, List[str]]:
        """Extract all entities from Neo4j for NER training"""
        entities = {}
        
        with self.integrator.driver.session() as session:
            for entity_type, query in self.entity_types.items():
                result = session.run(query)
                entities[entity_type] = [record['entity'] for record in result if record['entity']]
        
        return entities
    
    def generate_ner_training_data(self) -> List[Tuple[str, Dict]]:
        """Generate training data for NER model"""
        entities = self.extract_entities_from_graph()
        training_data = []
        
        # Generate synthetic sentences with entities
        templates = [
            "Show me data from {TABLE_NAME} table",
            "What is the {COLUMN_NAME} value?",
            "Filter by {COLUMN_NAME} column",
            "Find tables related to {CONCEPT_NAME}",
            "Show {DATA_TYPE} columns",
            "Get all {TABLE_NAME} records",
            "What columns are in {TABLE_NAME}?",
            "Show me {COLUMN_NAME} statistics",
            "Find similar columns to {COLUMN_NAME}",
            "What is the data type of {COLUMN_NAME}?",
            "List all {DATA_TYPE} fields",
            "Show relationships for {TABLE_NAME}",
            "Get metadata for {COLUMN_NAME}"
        ]
        
        for template in templates:
            for entity_type in entities:
                for entity_value in entities[entity_type][:10]:  # Limit for example
                    if f"{{{entity_type}}}" in template:
                        text = template.replace(f"{{{entity_type}}}", entity_value)
                        start = text.find(entity_value)
                        end = start + len(entity_value)
                        
                        training_example = (text, {
                            'entities': [(start, end, entity_type)]
                        })
                        training_data.append(training_example)
        
        return training_data

    def save_training_data(self, training_data: List[Tuple[str, Dict]], filepath: str):
        """Save NER training data to file"""
        with open(filepath, 'w') as f:
            for text, entities in training_data:
                f.write(json.dumps({"text": text, "entities": entities}) + "\n")

class RANQueryInterface:
    """Interface for querying the RAN knowledge graph"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Perform semantic search across the graph"""
        with self.integrator.driver.session() as session:
            # Search for tables and columns based on query
            result = session.run("""
                MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                WHERE toLower(t.name) CONTAINS toLower($query) 
                   OR toLower(c.name) CONTAINS toLower($query)
                RETURN t.name as table_name, 
                       collect(c.name) as columns,
                       t.row_count as row_count
                ORDER BY t.name
                LIMIT $limit
            """, query=query, limit=limit)
            
            return [dict(record) for record in result]
    
    def find_related_tables(self, table_name: str) -> List[Dict]:
        """Find tables related to a given table through various relationship types"""
        with self.integrator.driver.session() as session:
            # Find related tables through semantic relationships
            result = session.run("""
                MATCH (t1:Table {name: $table_name})-[:HAS_COLUMN]->(c1:Column)
                MATCH (c1)-[r]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
                WHERE type(r) IN ['NAME_SIMILARITY', 'VALUE_OVERLAP', 'PATTERN_MATCH', 'REFERENCES', 'CONCEPTUAL_GROUP']
                  AND t1 <> t2
                RETURN DISTINCT t2.name as related_table,
                       type(r) as relationship_type,
                       count(c2) as common_columns,
                       collect(DISTINCT c2.name) as related_columns
                ORDER BY common_columns DESC
            """, table_name=table_name)
            
            return [dict(record) for record in result]

    def get_table_details(self, table_name: str) -> Dict:
        """Get detailed information about a specific table"""
        with self.integrator.driver.session() as session:
            result = session.run("""
                MATCH (t:Table {name: $table_name})-[:HAS_COLUMN]->(c:Column)
                RETURN t.name as table_name,
                       t.row_count as row_count,
                       t.column_count as column_count,
                       t.created_at as created_at,
                       collect({
                           name: c.name,
                           data_type: c.data_type,
                           null_count: c.null_count,
                           unique_count: c.unique_count,
                           sample_values: c.sample_values
                       }) as columns
            """, table_name=table_name)
            
            record = result.single()
            return dict(record) if record else {}

    def get_column_relationships(self, column_id: str) -> List[Dict]:
        """Get all relationships for a specific column"""
        with self.integrator.driver.session() as session:
            result = session.run("""
                MATCH (c1:Column {id: $column_id})-[r]-(c2:Column)
                RETURN c2.id as related_column,
                       c2.name as column_name,
                       type(r) as relationship_type,
                       properties(r) as relationship_properties
            """, column_id=column_id)
            
            return [dict(record) for record in result]

    def search_by_concept(self, concept_query: str) -> List[Dict]:
        """Search for tables/columns by conceptual similarity"""
        with self.integrator.driver.session() as session:
            result = session.run("""
                MATCH (c1:Column)-[r:CONCEPTUAL_GROUP]-(c2:Column)
                WHERE r.semantic_category CONTAINS toLower($query)
                   OR toLower(c1.name) CONTAINS toLower($query)
                   OR toLower(c2.name) CONTAINS toLower($query)
                RETURN DISTINCT r.semantic_category as concept,
                       collect(DISTINCT c1.name + ' (' + split(c1.id, '.')[0] + ')') as related_columns
                ORDER BY concept
            """, query=concept_query)
            
            return [dict(record) for record in result]

    def get_schema_overview(self) -> Dict:
        """Get an overview of the entire schema"""
        with self.integrator.driver.session() as session:
            # Get table statistics
            tables_result = session.run("""
                MATCH (t:Table)
                RETURN count(t) as total_tables,
                       collect(t.name) as table_names
            """)
            
            # Get relationship statistics
            relationships_result = session.run("""
                MATCH ()-[r]-()
                RETURN type(r) as relationship_type,
                       count(r) as count
                ORDER BY count DESC
            """)
            
            # Get column statistics
            columns_result = session.run("""
                MATCH (c:Column)
                RETURN count(c) as total_columns,
                       count(DISTINCT c.data_type) as unique_data_types,
                       collect(DISTINCT c.data_type) as data_types
            """)
            
            tables_data = tables_result.single()
            columns_data = columns_result.single()
            relationships_data = [dict(record) for record in relationships_result]
            
            return {
                'tables': dict(tables_data) if tables_data else {},
                'columns': dict(columns_data) if columns_data else {},
                'relationships': relationships_data
            }

class RANChatbot:
    """Main chatbot class that combines NER and Query capabilities"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.ner_generator = RANNERGenerator(neo4j_integrator)
        self.query_interface = RANQueryInterface(neo4j_integrator)
        
    def process_query(self, user_query: str) -> Dict:
        """Process a user query and return appropriate response"""
        user_query_lower = user_query.lower()
        
        # Determine query intent
        if any(word in user_query_lower for word in ['show', 'list', 'get', 'find']):
            if 'related' in user_query_lower or 'similar' in user_query_lower:
                # Extract table name from query (simplified)
                words = user_query.split()
                table_name = None
                for word in words:
                    if word not in ['show', 'me', 'find', 'related', 'to', 'tables', 'similar']:
                        table_name = word
                        break
                
                if table_name:
                    results = self.query_interface.find_related_tables(table_name)
                    return {
                        'type': 'related_tables',
                        'query': user_query,
                        'table': table_name,
                        'results': results
                    }
            
            elif 'schema' in user_query_lower or 'overview' in user_query_lower:
                results = self.query_interface.get_schema_overview()
                return {
                    'type': 'schema_overview',
                    'query': user_query,
                    'results': results
                }
            
            else:
                # General semantic search
                results = self.query_interface.semantic_search(user_query)
                return {
                    'type': 'semantic_search',
                    'query': user_query,
                    'results': results
                }
        
        elif any(word in user_query_lower for word in ['details', 'info', 'about']):
            # Extract table name for details
            words = user_query.split()
            table_name = None
            for word in words:
                if word not in ['show', 'me', 'details', 'info', 'about', 'table']:
                    table_name = word
                    break
            
            if table_name:
                results = self.query_interface.get_table_details(table_name)
                return {
                    'type': 'table_details',
                    'query': user_query,
                    'table': table_name,
                    'results': results
                }
        
        # Default to semantic search
        results = self.query_interface.semantic_search(user_query)
        return {
            'type': 'semantic_search',
            'query': user_query,
            'results': results
        }

    def generate_response(self, query_result: Dict) -> str:
        """Generate a natural language response from query results"""
        if query_result['type'] == 'related_tables':
            table = query_result['table']
            results = query_result['results']
            
            if not results:
                return f"No related tables found for '{table}'."
            
            response = f"Found {len(results)} tables related to '{table}':\n"
            for result in results[:5]:  # Limit to top 5
                response += f"- {result['related_table']} (relationship: {result['relationship_type']}, common columns: {result['common_columns']})\n"
            
            return response
        
        elif query_result['type'] == 'table_details':
            table = query_result['table']
            details = query_result['results']
            
            if not details:
                return f"Table '{table}' not found."
            
            response = f"Details for table '{table}':\n"
            response += f"- Rows: {details.get('row_count', 'N/A')}\n"
            response += f"- Columns: {details.get('column_count', 'N/A')}\n"
            response += f"- Created: {details.get('created_at', 'N/A')}\n"
            
            if 'columns' in details:
                response += "Columns:\n"
                for col in details['columns'][:10]:  # Limit to first 10 columns
                    response += f"  - {col['name']} ({col['data_type']})\n"
            
            return response
        
        elif query_result['type'] == 'schema_overview':
            results = query_result['results']
            
            response = "Schema Overview:\n"
            if 'tables' in results:
                response += f"- Total Tables: {results['tables'].get('total_tables', 0)}\n"
            if 'columns' in results:
                response += f"- Total Columns: {results['columns'].get('total_columns', 0)}\n"
                response += f"- Data Types: {results['columns'].get('unique_data_types', 0)}\n"
            
            if 'relationships' in results:
                response += "Relationships:\n"
                for rel in results['relationships'][:5]:
                    response += f"  - {rel['relationship_type']}: {rel['count']}\n"
            
            return response
        
        else:  # semantic_search
            results = query_result['results']
            
            if not results:
                return "No matching tables or columns found."
            
            response = f"Found {len(results)} matching results:\n"
            for result in results[:5]:  # Limit to top 5
                response += f"- Table: {result['table_name']} ({result.get('row_count', 'N/A')} rows)\n"
                if result.get('columns'):
                    response += f"  Columns: {', '.join(result['columns'][:5])}\n"
            
            return response
