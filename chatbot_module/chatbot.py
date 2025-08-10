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
import time
import hashlib
import re
import torch
from datetime import datetime
import os

class OptimizedQueryInterface:
    """Performance-optimized query interface for large KG"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        # Add query result caching
        self.query_cache = {}
        self.cache_ttl = 3600  # 1 hour
        
    def cached_query(self, query: str, params: dict = None):
        """Execute query with caching"""
        cache_key = hashlib.md5(f"{query}{str(params)}".encode()).hexdigest()
        
        if cache_key in self.query_cache:
            cached_result, timestamp = self.query_cache[cache_key]
            if time.time() - timestamp < self.cache_ttl:
                return cached_result
        
        with self.integrator.driver.session() as session:
            result = session.run(query, params or {})
            data = [dict(record) for record in result]
            self.query_cache[cache_key] = (data, time.time())
            return data
    
    def optimized_semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Optimized search with pagination and indexing"""
        search_query = """
            MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
            WHERE toLower(t.name) CONTAINS toLower($search_query) 
               OR toLower(c.name) CONTAINS toLower($search_query)
            OPTIONAL MATCH (c)-[r]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
            WHERE type(r) IN ['NAME_SIMILARITY', 'VALUE_OVERLAP', 'PATTERN_MATCH', 'REFERENCES', 'CONCEPTUAL_GROUP']
            WITH t, c, count(DISTINCT r) as relationship_count, 
                 collect(DISTINCT type(r)) as relationship_types,
                 collect(DISTINCT t2.name) as related_tables,
                 CASE 
                   WHEN toLower(t.name) CONTAINS toLower($search_query) THEN 2.0
                   WHEN toLower(c.name) CONTAINS toLower($search_query) THEN 1.0
                   ELSE 0.5
                 END as relevance_score
            RETURN t.name as table_name, 
                   collect(DISTINCT c.name)[0..5] as top_columns,
                   t.row_count as row_count,
                   t.column_count as column_count,
                   relationship_count,
                   relationship_types,
                   related_tables[0..3] as sample_related_tables,
                   relevance_score
            ORDER BY relevance_score DESC, relationship_count DESC, t.name
            LIMIT $limit
        """
        return self.cached_query(search_query, {'search_query': query, 'limit': limit})

class EnhancedRANEntityExtractor:
    """Advanced entity extraction for RAN domain"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        
        # RAN-specific entity patterns
        self.ran_patterns = {
            'cell_id': r'cell[_\s]*id[_\s]*\d*',
            'frequency': r'(?:\d+\s*(?:mhz|ghz|khz))|\bfreq\b|\bfrequency\b|band(?:width)?',
            'power': r'(?:\d+\s*(?:dbm|watts?|mw))|\bpower\b|\bdbm\b',
            'coordinates': r'lat|lon|latitude|longitude|x_coord|y_coord',
            'timestamps': r'timestamp|time|date|created|updated',
            'identifiers': r'[a-z]+_id|id_[a-z]+|uuid|guid',
            'measurements': r'rsrp|rsrq|sinr|throughput|latency|kpi|metric|cqi|bler'
        }
    
    def extract_technical_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract RAN-specific technical entities"""
        entities = {
            'technical_terms': [],
            'measurements': [],
            'identifiers': [],
            'temporal': [],
            'spatial': []
        }
        
        query_lower = query.lower()
        
        # Pattern-based extraction
        for pattern_name, pattern in self.ran_patterns.items():
            matches = re.findall(pattern, query_lower)
            if matches:
                if pattern_name in ['cell_id', 'identifiers']:
                    entities['identifiers'].extend(matches)
                elif pattern_name in ['frequency', 'power', 'measurements']:
                    entities['measurements'].extend(matches)
                elif pattern_name == 'timestamps':
                    entities['temporal'].extend(matches)
                elif pattern_name == 'coordinates':
                    entities['spatial'].extend(matches)
        
        return entities
    
    def contextualized_search(self, query: str, entities: Dict) -> Tuple[str, Dict]:
        """Generate contextualized Cypher query based on entities"""
        base_query = """
            MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
            WHERE 1=1
        """
        
        conditions = []
        params = {}
        
        if entities['measurements']:
            conditions.append("any(term IN $measurements WHERE toLower(c.name) CONTAINS term)")
            params['measurements'] = entities['measurements']
        
        if entities['identifiers']:
            conditions.append("any(term IN $identifiers WHERE toLower(c.name) CONTAINS term)")
            params['identifiers'] = entities['identifiers']
        
        if entities['temporal']:
            conditions.append("any(term IN $temporal WHERE toLower(c.name) CONTAINS term)")
            params['temporal'] = entities['temporal']
        
        if entities['spatial']:
            conditions.append("any(term IN $spatial WHERE toLower(c.name) CONTAINS term)")
            params['spatial'] = entities['spatial']
        
        if conditions:
            base_query += " AND (" + " OR ".join(conditions) + ")"
        
        base_query += """
            OPTIONAL MATCH (c)-[r:CONCEPTUAL_GROUP]-(c2:Column)
            RETURN t.name as table_name,
                   collect(DISTINCT c.name) as matching_columns,
                   count(r) as conceptual_relationships,
                   t.row_count as row_count
            ORDER BY conceptual_relationships DESC, row_count DESC
            LIMIT 10
        """
        
        return base_query, params

class IntelligentGraphTraversal:
    """Optimized graph traversal for complex relationship queries"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
    
    def multi_hop_relationships(self, table_name: str, max_hops: int = 3) -> List[Dict]:
        """Find multi-hop relationships with path significance scoring"""
        query = """
            MATCH (t1:Table {name: $table_name})-[:HAS_COLUMN]->(c1:Column)
            MATCH path = (c1)-[r*1..$max_hops]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
            WHERE t1 <> t2 AND all(rel IN r WHERE type(rel) IN ['CONCEPTUAL_GROUP', 'PATTERN_MATCH', 'NAME_SIMILARITY'])
            WITH t2, path, 
                 reduce(score = 0, rel IN r | 
                   score + CASE type(rel)
                     WHEN 'CONCEPTUAL_GROUP' THEN 1.0
                     WHEN 'PATTERN_MATCH' THEN 0.8
                     WHEN 'NAME_SIMILARITY' THEN 0.6
                     ELSE 0.3 END
                 ) as path_score,
                 length(path) as path_length
            RETURN t2.name as related_table,
                   avg(path_score) as avg_relationship_strength,
                   count(DISTINCT path) as connection_count,
                   min(path_length) as shortest_path,
                   t2.row_count as row_count
            ORDER BY avg_relationship_strength DESC, connection_count DESC
            LIMIT 15
        """
        
        with self.integrator.driver.session() as session:
            result = session.run(query, table_name=table_name, max_hops=max_hops)
            return [dict(record) for record in result]
    
    def semantic_clustering(self, concept_threshold: float = 0.7) -> List[Dict]:
        """Find semantic clusters in the knowledge graph"""
        query = """
            MATCH (c1:Column)-[r:CONCEPTUAL_GROUP]-(c2:Column)
            WHERE r.confidence > $threshold
            WITH r.semantic_category as cluster_name, 
                 collect(DISTINCT c1.id) + collect(DISTINCT c2.id) as column_ids,
                 avg(r.confidence) as avg_confidence
            UNWIND column_ids as col_id
            MATCH (c:Column {id: col_id})<-[:HAS_COLUMN]-(t:Table)
            WITH cluster_name, avg_confidence, 
                 collect(DISTINCT t.name) as tables,
                 collect(DISTINCT c.name) as columns,
                 count(DISTINCT t) as table_count
            WHERE table_count >= 3
            RETURN cluster_name, avg_confidence, tables[0..10] as sample_tables, 
                   columns[0..15] as sample_columns, table_count
            ORDER BY avg_confidence DESC, table_count DESC
            LIMIT 20
        """
        
        with self.integrator.driver.session() as session:
            result = session.run(query, threshold=concept_threshold)
            return [dict(record) for record in result]

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
        self.ran_keywords = {
            'performance': ['throughput', 'latency', 'kpi', 'performance', 'metric'],
            'power': ['power', 'energy', 'dbm', 'consumption', 'efficiency'],
            'frequency': ['frequency', 'spectrum', 'bandwidth', 'channel', 'carrier'],
            'topology': ['cell', 'site', 'antenna', 'base_station', 'node'],
            'quality': ['rsrp', 'rsrq', 'sinr', 'quality', 'signal'],
            'traffic': ['traffic', 'volume', 'data', 'load', 'usage'],
            'mobility': ['handover', 'mobility', 'roaming', 'tracking'],
            'configuration': ['config', 'parameter', 'setting', 'threshold'],
            'security': ['security', 'authentication', 'encryption'],
            'timing': ['time', 'sync', 'clock', 'timing']
        }
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Perform enhanced semantic search across the graph"""
        with self.integrator.driver.session() as session:
            # Enhanced search with relationship context
            result = session.run("""
                MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                WHERE toLower(t.name) CONTAINS toLower($search_query) 
                   OR toLower(c.name) CONTAINS toLower($search_query)
                OPTIONAL MATCH (c)-[r]-(c2:Column)<-[:HAS_COLUMN]-(t2:Table)
                WHERE type(r) IN ['NAME_SIMILARITY', 'VALUE_OVERLAP', 'PATTERN_MATCH', 'REFERENCES', 'CONCEPTUAL_GROUP']
                WITH t, c, count(DISTINCT r) as relationship_count, 
                     collect(DISTINCT type(r)) as relationship_types,
                     collect(DISTINCT t2.name) as related_tables
                RETURN t.name as table_name, 
                       collect(DISTINCT c.name) as columns,
                       t.row_count as row_count,
                       t.column_count as column_count,
                       relationship_count,
                       relationship_types,
                       related_tables
                ORDER BY relationship_count DESC, t.name
                LIMIT $limit
            """, search_query=query, limit=limit)
            
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
        """Search for tables/columns by RAN conceptual similarity"""
        with self.integrator.driver.session() as session:
            result = session.run("""
                MATCH (c1:Column)-[r:CONCEPTUAL_GROUP]-(c2:Column)
                WHERE toLower(r.semantic_category) CONTAINS toLower($query)
                   OR toLower(r.concept_name) CONTAINS toLower($query)
                   OR toLower(c1.name) CONTAINS toLower($query)
                   OR toLower(c2.name) CONTAINS toLower($query)
                WITH r.semantic_category as concept, 
                     r.concept_name as concept_name,
                     r.grouping_method as method,
                     collect(DISTINCT c1.id) + collect(DISTINCT c2.id) as column_ids
                UNWIND column_ids as col_id
                MATCH (c:Column {id: col_id})<-[:HAS_COLUMN]-(t:Table)
                RETURN concept, 
                       concept_name,
                       method,
                       collect(DISTINCT t.name) as tables,
                       collect(DISTINCT c.name + ' (' + t.name + ')') as columns
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
                       collect(t.name) as table_names,
                       avg(t.row_count) as avg_row_count,
                       max(t.row_count) as max_row_count
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
            
            # Get conceptual groups
            concepts_result = session.run("""
                MATCH ()-[r:CONCEPTUAL_GROUP]-()
                WHERE r.semantic_category IS NOT NULL
                RETURN DISTINCT r.semantic_category as concept,
                       count(r) as usage_count,
                       r.grouping_method as method
                ORDER BY usage_count DESC
            """)
            
            tables_data = tables_result.single()
            columns_data = columns_result.single()
            relationships_data = [dict(record) for record in relationships_result]
            concepts_data = [dict(record) for record in concepts_result]
            
            return {
                'tables': dict(tables_data) if tables_data else {},
                'columns': dict(columns_data) if columns_data else {},
                'relationships': relationships_data,
                'concepts': concepts_data
            }

    def find_pattern_matches(self, pattern_type: str) -> List[Dict]:
        """Find columns that match specific RAN patterns"""
        with self.integrator.driver.session() as session:
            result = session.run("""
                MATCH (c1:Column)-[r:PATTERN_MATCH]-(c2:Column)
                WHERE r.pattern_type = $pattern_type
                MATCH (c1)<-[:HAS_COLUMN]-(t1:Table)
                MATCH (c2)<-[:HAS_COLUMN]-(t2:Table)
                RETURN c1.name as column1,
                       t1.name as table1,
                       c2.name as column2,
                       t2.name as table2,
                       r.confidence as confidence,
                       r.pattern_description as description
                ORDER BY r.confidence DESC
            """, pattern_type=pattern_type)
            
            return [dict(record) for record in result]

    def get_ran_domain_insights(self, domain: str) -> Dict:
        """Get insights for specific RAN domain areas"""
        domain_keywords = self.ran_keywords.get(domain.lower(), [])
        
        if not domain_keywords:
            return {'error': f'Unknown RAN domain: {domain}'}
        
        with self.integrator.driver.session() as session:
            # Find tables and columns related to this domain
            keyword_pattern = '|'.join(domain_keywords)
            result = session.run("""
                MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                WHERE any(keyword IN $keywords WHERE 
                    toLower(t.name) CONTAINS toLower(keyword) OR 
                    toLower(c.name) CONTAINS toLower(keyword))
                OPTIONAL MATCH (c)-[r:CONCEPTUAL_GROUP]-(c2:Column)
                WHERE r.semantic_category = $domain
                RETURN t.name as table_name,
                       collect(DISTINCT c.name) as matching_columns,
                       t.row_count as row_count,
                       count(r) as concept_relationships
                ORDER BY concept_relationships DESC, row_count DESC
            """, keywords=domain_keywords, domain=domain)
            
            tables = [dict(record) for record in result]
            
            # Get pattern matches for this domain
            pattern_result = session.run("""
                MATCH (c1:Column)-[r:PATTERN_MATCH]-(c2:Column)
                WHERE any(keyword IN $keywords WHERE 
                    toLower(r.pattern_description) CONTAINS toLower(keyword))
                RETURN r.pattern_type as pattern,
                       count(r) as pattern_count
                ORDER BY pattern_count DESC
            """, keywords=domain_keywords)
            
            patterns = [dict(record) for record in pattern_result]
            
            return {
                'domain': domain,
                'related_tables': tables,
                'domain_patterns': patterns,
                'keywords_used': domain_keywords
            }

class RANChatbot:
    """Main chatbot class that combines NER and Query capabilities"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.ner_generator = RANNERGenerator(neo4j_integrator)
        self.query_interface = RANQueryInterface(neo4j_integrator)
        
        # RAN-specific intent patterns
        self.intent_patterns = {
            'domain_inquiry': ['performance', 'power', 'frequency', 'topology', 'quality', 'traffic', 'mobility', 'configuration', 'security', 'timing'],
            'pattern_search': ['pattern', 'matching', 'similar structure', 'same format'],
            'concept_search': ['concept', 'conceptual', 'semantic', 'category'],
            'relationship_query': ['related', 'connected', 'linked', 'similar'],
            'schema_query': ['schema', 'overview', 'structure', 'summary'],
            'table_details': ['details', 'info', 'information', 'about', 'describe'],
            'list_query': ['show', 'list', 'get', 'find', 'all']
        }
    
    def detect_intent(self, query: str) -> str:
        """Detect user intent from query"""
        query_lower = query.lower()
        
        # Check for RAN domain inquiries
        for domain in self.intent_patterns['domain_inquiry']:
            if domain in query_lower:
                return 'domain_inquiry'
        
        # Check for pattern-related queries
        if any(pattern in query_lower for pattern in self.intent_patterns['pattern_search']):
            return 'pattern_search'
        
        # Check for concept-related queries
        if any(concept in query_lower for concept in self.intent_patterns['concept_search']):
            return 'concept_search'
        
        # Check for relationship queries
        if any(rel in query_lower for rel in self.intent_patterns['relationship_query']):
            return 'relationship_query'
        
        # Check for schema queries
        if any(schema in query_lower for schema in self.intent_patterns['schema_query']):
            return 'schema_query'
        
        # Check for table details
        if any(detail in query_lower for detail in self.intent_patterns['table_details']):
            return 'table_details'
        
        # Default to list query
        return 'semantic_search'
    
    def extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract entities from user query"""
        entities = {
            'tables': [],
            'columns': [],
            'domains': [],
            'patterns': []
        }
        
        query_lower = query.lower()
        words = query.split()
        
        # Extract RAN domains
        for domain in self.intent_patterns['domain_inquiry']:
            if domain in query_lower:
                entities['domains'].append(domain)
        
        # Extract potential table/column names (simplified heuristic)
        for word in words:
            if '_' in word or word.endswith('_config') or word.endswith('_data'):
                if len(word) > 3:
                    entities['tables'].append(word)
        
        return entities
        
    def process_query(self, user_query: str) -> Dict:
        """Process a user query and return appropriate response"""
        intent = self.detect_intent(user_query)
        entities = self.extract_entities(user_query)
        
        if intent == 'domain_inquiry':
            # Handle RAN domain-specific queries
            domain = entities['domains'][0] if entities['domains'] else 'general'
            results = self.query_interface.get_ran_domain_insights(domain)
            return {
                'type': 'domain_inquiry',
                'query': user_query,
                'domain': domain,
                'results': results
            }
        
        elif intent == 'pattern_search':
            # Extract pattern type from query
            pattern_types = ['id_field', 'timestamp', 'status_field', 'power_related', 'frequency_related']
            pattern_type = None
            for pt in pattern_types:
                if pt.replace('_', ' ') in user_query.lower():
                    pattern_type = pt
                    break
            
            if pattern_type:
                results = self.query_interface.find_pattern_matches(pattern_type)
                return {
                    'type': 'pattern_search',
                    'query': user_query,
                    'pattern_type': pattern_type,
                    'results': results
                }
        
        elif intent == 'concept_search':
            # Search by conceptual similarity
            results = self.query_interface.search_by_concept(user_query)
            return {
                'type': 'concept_search',
                'query': user_query,
                'results': results
            }
        
        elif intent == 'relationship_query':
            # Extract table name from query
            table_name = self._extract_table_name(user_query)
            if table_name:
                results = self.query_interface.find_related_tables(table_name)
                return {
                    'type': 'related_tables',
                    'query': user_query,
                    'table': table_name,
                    'results': results
                }
        
        elif intent == 'schema_query':
            results = self.query_interface.get_schema_overview()
            return {
                'type': 'schema_overview',
                'query': user_query,
                'results': results
            }
        
        elif intent == 'table_details':
            table_name = self._extract_table_name(user_query)
            if table_name:
                results = self.query_interface.get_table_details(table_name)
                return {
                    'type': 'table_details',
                    'query': user_query,
                    'table': table_name,
                    'results': results
                }
        
        # Default to enhanced semantic search
        results = self.query_interface.semantic_search(user_query)
        return {
            'type': 'semantic_search',
            'query': user_query,
            'results': results
        }
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from query (improved heuristic)"""
        words = query.split()
        
        # Look for words that look like table names
        for word in words:
            word_clean = word.strip('.,!?').lower()
            if ('_' in word_clean or 
                word_clean.endswith('config') or 
                word_clean.endswith('data') or
                word_clean.endswith('table') or
                word_clean.endswith('counters') or
                word_clean.endswith('relations')):
                return word_clean.replace('table', '').strip()
        
        return None

    def generate_response(self, query_result: Dict) -> str:
        """Generate a natural language response from query results"""
        if query_result['type'] == 'domain_inquiry':
            domain = query_result['domain']
            results = query_result['results']
            
            if 'error' in results:
                return results['error']
            
            response = f"🔍 **RAN {domain.title()} Domain Analysis:**\n\n"
            
            if results.get('related_tables'):
                response += f"📊 **Related Tables ({len(results['related_tables'])}):**\n"
                for table in results['related_tables'][:5]:
                    response += f"• {table['table_name']} ({table.get('row_count', 'N/A')} rows)\n"
                    if table['matching_columns']:
                        response += f"  - Relevant columns: {', '.join(table['matching_columns'][:3])}\n"
                response += "\n"
            
            if results.get('domain_patterns'):
                response += f"🔍 **Pattern Analysis:**\n"
                for pattern in results['domain_patterns'][:3]:
                    response += f"• {pattern['pattern']}: {pattern['pattern_count']} matches\n"
                response += "\n"
            
            response += f"🎯 **Keywords searched:** {', '.join(results.get('keywords_used', []))}"
            return response
        
        elif query_result['type'] == 'pattern_search':
            pattern_type = query_result['pattern_type']
            results = query_result['results']
            
            if not results:
                return f"No pattern matches found for '{pattern_type}'."
            
            response = f"🔍 **Pattern Matches for '{pattern_type}':**\n\n"
            for result in results[:5]:
                response += f"• **{result['table1']}.{result['column1']}** ↔ **{result['table2']}.{result['column2']}**\n"
                response += f"  - Confidence: {result['confidence']:.2f}\n"
                response += f"  - Description: {result['description']}\n\n"
            
            return response
        
        elif query_result['type'] == 'concept_search':
            results = query_result['results']
            
            if not results:
                return "No conceptual groups found matching your query."
            
            response = f"🧠 **Conceptual Groups Found ({len(results)}):**\n\n"
            for result in results:
                concept = result.get('concept', result.get('concept_name', 'Unknown'))
                response += f"📂 **{concept}** ({result.get('method', 'unknown')} method)\n"
                if result.get('tables'):
                    response += f"  - Tables: {', '.join(result['tables'][:3])}\n"
                if result.get('columns'):
                    response += f"  - Columns: {', '.join(result['columns'][:5])}\n"
                response += "\n"
            
            return response
        
        elif query_result['type'] == 'related_tables':
            table = query_result['table']
            results = query_result['results']
            
            if not results:
                return f"No related tables found for '{table}'."
            
            response = f"🔗 **Tables Related to '{table}' ({len(results)}):**\n\n"
            for result in results[:5]:
                response += f"• **{result['related_table']}**\n"
                response += f"  - Relationship: {result['relationship_type']}\n"
                response += f"  - Common columns: {result['common_columns']}\n"
                if result.get('related_columns'):
                    response += f"  - Related columns: {', '.join(result['related_columns'][:3])}\n"
                response += "\n"
            
            return response
        
        elif query_result['type'] == 'table_details':
            table = query_result['table']
            details = query_result['results']
            
            if not details:
                return f"Table '{table}' not found."
            
            response = f"📋 **Details for Table '{table}':**\n\n"
            response += f"📊 **Statistics:**\n"
            response += f"• Rows: {details.get('row_count', 'N/A'):,}\n"
            response += f"• Columns: {details.get('column_count', 'N/A')}\n"
            response += f"• Created: {details.get('created_at', 'N/A')}\n\n"
            
            if 'columns' in details and details['columns']:
                response += f"📝 **Columns ({len(details['columns'])}):**\n"
                for col in details['columns'][:10]:
                    response += f"• **{col['name']}** ({col['data_type']})\n"
                    if col.get('sample_values'):
                        response += f"  - Sample values: {', '.join(str(v) for v in col['sample_values'][:3])}\n"
                if len(details['columns']) > 10:
                    response += f"... and {len(details['columns']) - 10} more columns\n"
            
            return response
        
        elif query_result['type'] == 'schema_overview':
            results = query_result['results']
            
            response = f"📊 **Knowledge Graph Schema Overview:**\n\n"
            
            if 'tables' in results:
                tables = results['tables']
                response += f"📋 **Tables:** {tables.get('total_tables', 0)}\n"
                if tables.get('avg_row_count'):
                    response += f"• Average rows: {tables['avg_row_count']:.0f}\n"
                if tables.get('max_row_count'):
                    response += f"• Largest table: {tables['max_row_count']:,} rows\n"
                response += "\n"
            
            if 'columns' in results:
                columns = results['columns']
                response += f"📝 **Columns:** {columns.get('total_columns', 0)}\n"
                response += f"• Data types: {columns.get('unique_data_types', 0)}\n"
                if columns.get('data_types'):
                    response += f"• Types: {', '.join(columns['data_types'][:5])}\n"
                response += "\n"
            
            if 'relationships' in results:
                response += f"🔗 **Relationships:**\n"
                total_rels = sum(r['count'] for r in results['relationships'])
                response += f"• Total: {total_rels:,}\n"
                for rel in results['relationships'][:5]:
                    response += f"• {rel['relationship_type']}: {rel['count']:,}\n"
                response += "\n"
            
            if 'concepts' in results and results['concepts']:
                response += f"🧠 **Conceptual Groups:**\n"
                for concept in results['concepts'][:5]:
                    response += f"• {concept['concept']}: {concept['usage_count']} relationships\n"
            
            return response
        
        else:  # semantic_search
            results = query_result['results']
            
            if not results:
                return "No matching tables or columns found."
            
            response = f"🔍 **Search Results ({len(results)}):**\n\n"
            for result in results[:5]:
                response += f"📋 **{result['table_name']}**\n"
                response += f"• Rows: {result.get('row_count', 'N/A'):,}\n"
                response += f"• Columns: {result.get('column_count', 'N/A')}\n"
                
                if result.get('columns'):
                    response += f"• Matching columns: {', '.join(result['columns'][:5])}\n"
                
                if result.get('relationship_count', 0) > 0:
                    response += f"• Relationships: {result['relationship_count']} ({', '.join(result.get('relationship_types', []))})\n"
                
                if result.get('related_tables'):
                    response += f"• Related to: {', '.join(result['related_tables'][:3])}\n"
                
                response += "\n"
            
            if len(results) > 5:
                response += f"... and {len(results) - 5} more results\n"
            
            return response

class EnhancedRANChatbot(RANChatbot):
    """Enhanced chatbot with optimizations and domain-specific capabilities"""

    def __init__(self, neo4j_integrator, use_domain_model: bool = False, model_dir: str | None = None):
        super().__init__(neo4j_integrator)

        # Enhanced components
        self.optimized_query = OptimizedQueryInterface(neo4j_integrator)
        self.entity_extractor = EnhancedRANEntityExtractor(neo4j_integrator)
        self.graph_traversal = IntelligentGraphTraversal(neo4j_integrator)
        # Cache of last domain tables for ranking boosts
        self._last_domain_tables: set[str] = set()
        # Intent routing and synonyms to better guide retrieval
        self.intent_domain_map = {
            'performance_analysis': 'performance',
            'power_optimization': 'power',
            'spectrum_management': 'frequency',
            'cell_configuration': 'configuration',
            'quality_assessment': 'quality',
            'traffic_analysis': 'traffic',
            'fault_detection': 'security',  # map to security/fault domain
            'capacity_planning': 'performance',
            'interference_analysis': 'quality',
            'handover_optimization': 'mobility',
        }
        self.query_synonyms = {
            'power': ['power','dbm','energy','consumption','watts'],
            'frequency': ['frequency','freq','band','bandwidth','carrier','channel'],
            'performance': ['throughput','latency','kpi','utilization','efficiency','metric'],
            'quality': ['rsrp','rsrq','sinr','quality','noise','interference','cqi','bler'],
            'mobility': ['handover','handoff','roaming','x2','ng'],
            'configuration': ['config','parameter','setting','threshold','cell','antenna','tilt'],
            'traffic': ['traffic','volume','bytes','packets','usage','load','congestion']
        }
        
        # Load domain-specific model if available
        self.domain_model = None
        self.domain_tokenizer = None
        # Resolve model directory
        self.model_dir = None
        candidate_dirs = []
        if model_dir:
            candidate_dirs.append(model_dir)
        # Relative to repository root and module folder
        candidate_dirs.extend([
            os.path.join(os.path.dirname(__file__), 'ran_domain_model'),
            os.path.abspath(os.path.join(os.getcwd(), 'chatbot_module', 'ran_domain_model')),
            os.path.abspath(os.path.join(os.getcwd(), 'ran_domain_model')),
        ])
        for d in candidate_dirs:
            if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, 'config.json')):
                self.model_dir = d
                break
        if use_domain_model:
            try:
                # These imports will be available after fine-tuning
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                load_dir = self.model_dir or "./ran_domain_model"
                self.domain_tokenizer = AutoTokenizer.from_pretrained(load_dir)
                self.domain_model = AutoModelForSequenceClassification.from_pretrained(load_dir)
                print("Domain-specific model loaded successfully")
            except Exception as e:
                print(f"Domain-specific model not found: {e}")
                print("Using default intent detection")
    
    def enhanced_process_query(self, user_query: str) -> Dict:
        """Parallel processing with aggregated results from all retrieval strategies"""
        start_time = time.time()
        
        # Get intent prediction
        intent, confidence = self.predict_intent(user_query)
        
        # Initialize result aggregator (add query token tracking for improved ranking)
        tokens = [t for t in re.split(r"[^a-z0-9]+", user_query.lower()) if t and len(t) > 1]
        result_aggregator = {
            'all_tables': {},  # table_name -> {count, sources, details, sample_columns}
            'all_columns': {},  # column_name -> {count, sources, table}
            'key_results': {},  # detailed per-process results
            'processing_stats': {},
            'query_tokens': tokens
        }
        
        # Run all processes in parallel and collect results
        parallel_results = self._run_parallel_processes(user_query, intent, result_aggregator)
        
        # Aggregate and rank all tables/columns
        top_tables = self._rank_aggregated_tables(result_aggregator['all_tables'], result_aggregator)
        # Fallback: if we have zero tables, attempt semantic search over individual query tokens to recover something
        if not top_tables:
            try:
                seen = set()
                for tok in result_aggregator.get('query_tokens', [])[:4]:
                    if len(tok) < 3:  # skip very short tokens
                        continue
                    results = self.query_interface.semantic_search(tok)
                    for r in results[:5]:
                        tname = r.get('table_name')
                        if tname and tname not in seen:
                            seen.add(tname)
                            self._add_table_to_aggregator(tname, 'fallback_token_search', r, result_aggregator)
                # Re-rank after fallback
                top_tables = self._rank_aggregated_tables(result_aggregator['all_tables'], result_aggregator)
            except Exception:
                pass
        top_columns = self._rank_aggregated_columns(result_aggregator['all_columns'])
        # Ensure explicitly referenced table (if any) is surfaced even if low evidence
        explicit_table = result_aggregator['key_results'].get('explicit_table')
        if explicit_table and explicit_table not in [t['table_name'] for t in top_tables]:
            # Add explicit table with high priority score
            top_tables.insert(0, {
                'table_name': explicit_table,
                'frequency': 1,
                'sources': ['explicit_reference'],
                'relevance_score': 5.0,  # High relevance for explicit mention
                'total_score': 10.0,     # High total score
                'source_diversity': 1,
                'token_match_score': 2.0,
                'matched_query_tokens': [],
                'column_variety': 0,
                'sample_columns': []
            })
        elif explicit_table:
            # Boost existing explicit table SIGNIFICANTLY  
            for table in top_tables:
                if table['table_name'] == explicit_table:
                    table['total_score'] += 15.0  # Much larger boost
                    table['relevance_score'] += 10.0
                    table['explicit_boost'] = 15.0  # Track the boost
                    break
            # Re-sort after boosting
            top_tables.sort(key=lambda x: (x['total_score'], x['token_match_score'], x['source_diversity']), reverse=True)
        
        # Generate comprehensive response
        response = self._generate_aggregated_response(
            user_query, intent, top_tables, top_columns, 
            result_aggregator['key_results'], parallel_results
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        return {
            'type': 'parallel_aggregated',
            'query': user_query,
            'intent': intent,
            'confidence': confidence,
            'top_tables': top_tables[:10],  # Top 10 tables
            'top_columns': top_columns[:15],  # Top 15 columns
            'key_results': result_aggregator['key_results'],
            'parallel_results': parallel_results,
            'entities': result_aggregator['key_results'].get('entities', {}),
            'domain': result_aggregator['key_results'].get('domain'),
            'processing_time_ms': processing_time,
            'response': response,
            'debug': {
                'path': 'parallel_aggregated',
                'processes_run': list(result_aggregator['processing_stats'].keys()),
                'processing_stats': result_aggregator['processing_stats']
            }
        }

    def _run_parallel_processes(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Run all retrieval processes and aggregate results"""
        parallel_results = {}
        
        # 1. Intent-based Domain Routing
        try:
            domain_result = self._process_domain_routing(user_query, intent, aggregator)
            parallel_results['domain_routing'] = domain_result
            aggregator['processing_stats']['domain_routing'] = 'success'
        except Exception as e:
            parallel_results['domain_routing'] = {'error': str(e)}
            aggregator['processing_stats']['domain_routing'] = f'error: {e}'
        
        # 2. Explicit Table Extraction
        try:
            table_result = self._process_table_extraction(user_query, intent, aggregator)
            parallel_results['table_extraction'] = table_result
            aggregator['processing_stats']['table_extraction'] = 'success'
        except Exception as e:
            parallel_results['table_extraction'] = {'error': str(e)}
            aggregator['processing_stats']['table_extraction'] = f'error: {e}'
        
        # 3. Entity Extraction & Contextualized Search
        try:
            entity_result = self._process_entity_extraction(user_query, intent, aggregator)
            parallel_results['entity_extraction'] = entity_result
            aggregator['processing_stats']['entity_extraction'] = 'success'
        except Exception as e:
            parallel_results['entity_extraction'] = {'error': str(e)}
            aggregator['processing_stats']['entity_extraction'] = f'error: {e}'
        
        # 4. Synonym Expansion & Multi-term Search
        try:
            synonym_result = self._process_synonym_expansion(user_query, intent, aggregator)
            parallel_results['synonym_expansion'] = synonym_result
            aggregator['processing_stats']['synonym_expansion'] = 'success'
        except Exception as e:
            parallel_results['synonym_expansion'] = {'error': str(e)}
            aggregator['processing_stats']['synonym_expansion'] = f'error: {e}'
        
        # 5. Concept Search
        try:
            concept_result = self._process_concept_search(user_query, intent, aggregator)
            parallel_results['concept_search'] = concept_result
            aggregator['processing_stats']['concept_search'] = 'success'
        except Exception as e:
            parallel_results['concept_search'] = {'error': str(e)}
            aggregator['processing_stats']['concept_search'] = f'error: {e}'
        
        # 6. Multi-hop Relationships (for complex queries)
        try:
            relationship_result = self._process_relationship_analysis(user_query, intent, aggregator)
            parallel_results['relationship_analysis'] = relationship_result
            aggregator['processing_stats']['relationship_analysis'] = 'success'
        except Exception as e:
            parallel_results['relationship_analysis'] = {'error': str(e)}
            aggregator['processing_stats']['relationship_analysis'] = f'error: {e}'
        
        return parallel_results

    def _process_domain_routing(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process domain-based routing"""
        if intent not in self.intent_domain_map:
            return {'status': 'no_domain_mapping'}
        
        domain = self.intent_domain_map[intent]
        insights = self.query_interface.get_ran_domain_insights(domain)
        # Track domain tables for downstream ranking boosts
        try:
            self._last_domain_tables = {t.get('table_name') for t in insights.get('related_tables', []) if t.get('table_name')}
        except Exception:
            self._last_domain_tables = set()
        
        # Store key results
        aggregator['key_results']['domain'] = domain
        aggregator['key_results']['domain_insights'] = insights
        
        # Extract tables and columns
        if insights and insights.get('related_tables'):
            for table_info in insights['related_tables']:
                table_name = table_info.get('table_name')
                if table_name:
                    self._add_table_to_aggregator(table_name, 'domain_routing', table_info, aggregator)
                    # Add columns if available
                    for col in table_info.get('matching_columns', []):
                        self._add_column_to_aggregator(col, table_name, 'domain_routing', aggregator)
        
        return {
            'status': 'success',
            'domain': domain,
            'tables_found': len(insights.get('related_tables', [])) if insights else 0,
            'insights': insights
        }

    def _process_table_extraction(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process explicit table name extraction"""
        explicit_table = self._extract_table_name(user_query)
        
        if not explicit_table:
            return {'status': 'no_explicit_table'}
        
        # Get table details
        table_details = None
        related_tables = []
        
        try:
            table_details = self.query_interface.get_table_details(explicit_table)
            if table_details:
                self._add_table_to_aggregator(explicit_table, 'table_extraction', table_details, aggregator)
                # Add all columns from this table
                for col_info in table_details.get('columns', []):
                    col_name = col_info.get('name')
                    if col_name:
                        self._add_column_to_aggregator(col_name, explicit_table, 'table_extraction', aggregator)
        except Exception:
            # Suppress errors during table detail extraction to keep pipeline resilient
            pass
        
        try:
            related_tables = self.query_interface.find_related_tables(explicit_table)
            for related in related_tables:
                # Some interfaces return 'related_table'
                table_name = related.get('related_table') or related.get('table_name')
                if table_name:
                    self._add_table_to_aggregator(table_name, 'table_extraction_related', related, aggregator)
        except Exception:
            pass
        
        # Store key results
        aggregator['key_results']['explicit_table'] = explicit_table
        aggregator['key_results']['table_details'] = table_details
        
        return {
            'status': 'success',
            'explicit_table': explicit_table,
            'table_details_found': table_details is not None,
            'related_tables_found': len(related_tables),
            'table_details': table_details,
            'related_tables': related_tables
        }

    def _process_entity_extraction(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process entity extraction and contextualized search"""
        entities = self.entity_extractor.extract_technical_entities(user_query)
        # Enrich with explicit table name detections from KG
        try:
            entities = self.entity_enrich_with_tables(user_query, entities)
        except Exception:
            pass
        
        # Store key results
        aggregator['key_results']['entities'] = entities
        
        contextualized_results = []
        if entities['measurements'] or entities['identifiers']:
            try:
                cypher_query, params = self.entity_extractor.contextualized_search(user_query, entities)
                contextualized_results = self.optimized_query.cached_query(cypher_query, params)
                
                # Extract tables and columns from results
                for result in contextualized_results:
                    table_name = result.get('table_name')
                    if table_name:
                        self._add_table_to_aggregator(table_name, 'entity_extraction', result, aggregator)
                    
                    # Extract column names from result
                    for key, value in result.items():
                        if 'column' in key.lower() or 'field' in key.lower():
                            if isinstance(value, str) and value:
                                self._add_column_to_aggregator(value, table_name, 'entity_extraction', aggregator)
                
            except Exception:
                pass
        
        return {
            'status': 'success',
            'entities_extracted': entities,
            'contextualized_results_count': len(contextualized_results),
            'contextualized_results': contextualized_results[:5]  # Limit for response size
        }

    def _process_synonym_expansion(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process synonym expansion and multi-term search"""
        expanded_terms = self._expand_query_terms(user_query, intent)
        
        semantic_results = []
        seen_tables = set()
        
        for term in expanded_terms[:10]:  # Limit expansion
            try:
                results = self.optimized_query.optimized_semantic_search(term, limit=8)
            except Exception:
                try:
                    results = self.query_interface.semantic_search(term)
                except Exception:
                    results = []
            
            for result in results:
                table_name = result.get('table_name')
                if table_name and table_name not in seen_tables:
                    # Add relevance boost for exact matches
                    if term.lower() in table_name.lower():
                        result['relevance_boost'] = 2.0
                    else:
                        result['relevance_boost'] = 1.0
                    
                    self._add_table_to_aggregator(table_name, 'synonym_expansion', result, aggregator)
                    semantic_results.append(result)
                    seen_tables.add(table_name)
                    
                    # Add columns if available
                    for col in result.get('columns', []):
                        if isinstance(col, str):
                            self._add_column_to_aggregator(col, table_name, 'synonym_expansion', aggregator)
                        elif isinstance(col, dict) and col.get('name'):
                            self._add_column_to_aggregator(col['name'], table_name, 'synonym_expansion', aggregator)
        
        return {
            'status': 'success',
            'expanded_terms': expanded_terms,
            'semantic_results_count': len(semantic_results),
            'unique_tables_found': len(seen_tables)
        }

    def _process_concept_search(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process concept-based search"""
        concept_results = []
        
        try:
            # Try concept search if intent matches
            if intent == 'concept_search' or any(word in user_query.lower() for word in ['concept', 'pattern', 'category']):
                results = self.graph_traversal.semantic_clustering()
                
                for result in results:
                    concept_name = result.get('concept_name') or result.get('cluster_name')
                    if concept_name:
                        for table_name in result.get('sample_tables', []):
                            self._add_table_to_aggregator(table_name, 'concept_search', result, aggregator)
                        concept_results.append(result)
            
            # Also try general concept matching
            concept_keywords = ['power', 'frequency', 'cell', 'performance', 'neighbor', 'sync', 'timing']
            query_lower = user_query.lower()
            
            for keyword in concept_keywords:
                if keyword in query_lower:
                    try:
                        # Search for tables containing the concept keyword
                        search_results = self.query_interface.semantic_search(keyword)
                        for result in search_results[:3]:  # Limit per concept
                            table_name = result.get('table_name')
                            if table_name:
                                self._add_table_to_aggregator(table_name, f'concept_search_{keyword}', result, aggregator)
                    except Exception:
                        pass
        
        except Exception:
            pass
        
        return {
            'status': 'success',
            'concept_results_count': len(concept_results),
            'concept_results': concept_results[:3]  # Limit for response size
        }

    def _process_relationship_analysis(self, user_query: str, intent: str, aggregator: Dict) -> Dict:
        """Process multi-hop relationship analysis"""
        relationship_results = []
        
        # Check if query involves relationships
        if any(word in user_query.lower() for word in ['related', 'connected', 'linked', 'associated', 'relationship']):
            table_name = self._extract_table_name(user_query)
            if table_name:
                try:
                    results = self.graph_traversal.multi_hop_relationships(table_name)
                    
                    for result in results:
                        related_table = result.get('related_table')
                        if related_table:
                            self._add_table_to_aggregator(related_table, 'relationship_analysis', result, aggregator)
                        relationship_results.append(result)
                
                except Exception:
                    pass
        
        return {
            'status': 'success',
            'relationship_results_count': len(relationship_results),
            'relationship_results': relationship_results[:5]  # Limit for response size
        }

    def _add_table_to_aggregator(self, table_name: str, source: str, details: Dict, aggregator: Dict):
        """Add table to aggregated results"""
        if table_name not in aggregator['all_tables']:
            aggregator['all_tables'][table_name] = {
                'count': 0,
                'sources': [],
                'details': {},
                'relevance_score': 0,
                'sample_columns': set()
            }
        
        aggregator['all_tables'][table_name]['count'] += 1
        aggregator['all_tables'][table_name]['sources'].append(source)
        aggregator['all_tables'][table_name]['details'][source] = details
        
        # Add relevance boost
        boost = details.get('relevance_boost', 1.0)
        aggregator['all_tables'][table_name]['relevance_score'] += boost
        # Harvest columns if present in various shapes
        col_candidates = []
        if isinstance(details, dict):
            for key in ['columns', 'matching_columns', 'top_columns']:
                v = details.get(key)
                if isinstance(v, list):
                    for c in v:
                        if isinstance(c, str):
                            col_candidates.append(c)
                        elif isinstance(c, dict) and c.get('name'):
                            col_candidates.append(c['name'])
        for c in col_candidates[:8]:  # cap per insertion
            if c:
                aggregator['all_tables'][table_name]['sample_columns'].add(c)

    def _add_column_to_aggregator(self, column_name: str, table_name: str, source: str, aggregator: Dict):
        """Add column to aggregated results"""
        if column_name not in aggregator['all_columns']:
            aggregator['all_columns'][column_name] = {
                'count': 0,
                'sources': [],
                'tables': set(),
                'relevance_score': 0
            }
        
        aggregator['all_columns'][column_name]['count'] += 1
        aggregator['all_columns'][column_name]['sources'].append(source)
        aggregator['all_columns'][column_name]['tables'].add(table_name)
        aggregator['all_columns'][column_name]['relevance_score'] += 1

    def _rank_aggregated_tables(self, all_tables: Dict, aggregator: Dict | None = None) -> List[Dict]:
        """Rank tables using multi-factor scoring with token matching & pseudo-IDF.
        Factors (additive weighted):
          - frequency: how many processes surfaced the table
          - relevance_score: accumulated boosts from source processors
          - diversity: number of distinct retrieval sources
          - domain_boost: if table in last domain tables set
          - token_match: overlap between query tokens and table / sample columns (BM25-like)
          - column_variety: number of harvested sample columns
        Provides per-table debug signal vector to aid evaluation / tuning.
        """
        ranked_tables: List[Dict] = []
        query_tokens = set((aggregator or {}).get('query_tokens', []))
        # Build corpus statistics for pseudo-IDF over sample columns
        token_df: Dict[str,int] = {}
        for info in all_tables.values():
            seen_tokens = set()
            cols = list(info.get('sample_columns') or [])
            for c in cols:
                for tok in re.split(r"[^a-z0-9]+", c.lower()):
                    if tok and len(tok) > 1:
                        seen_tokens.add(tok)
            for tok in seen_tokens:
                token_df[tok] = token_df.get(tok,0)+1
        N = max(len(all_tables),1)
        for table_name, info in all_tables.items():
            diversity = len(set(info['sources']))
            freq_component = info['count'] * 2.0
            relevance_component = info['relevance_score'] * 1.0
            diversity_component = diversity * 1.2
            domain_component = 1.5 if (table_name in getattr(self, '_last_domain_tables', set())) else 0.0
            # Token match scoring
            sample_cols = list(info.get('sample_columns') or [])
            col_tokens = []
            for c in sample_cols:
                col_tokens.extend([t for t in re.split(r"[^a-z0-9]+", c.lower()) if t and len(t)>1])
            if table_name:
                col_tokens.extend([t for t in re.split(r"[^a-z0-9]+", table_name.lower()) if t and len(t)>1])
            token_match_score = 0.0
            matched_tokens = set()
            for tok in query_tokens:
                tf = col_tokens.count(tok)
                if tf:
                    df = token_df.get(tok,1)
                    idf = np.log( (N - df + 0.5) / (df + 0.5) + 1 )  # BM25 style idf
                    # Simple BM25-like tf normalization
                    token_match_score += (tf / (tf + 1.5)) * idf * 2.0
                    matched_tokens.add(tok)
            column_variety = min(len(set(sample_cols)), 12)
            column_variety_component = 0.3 * (column_variety/12)
            total = freq_component + relevance_component + diversity_component + domain_component + token_match_score + column_variety_component
            ranked_tables.append({
                'table_name': table_name,
                'frequency': info['count'],
                'sources': list(set(info['sources'])),
                'relevance_score': info['relevance_score'],
                'source_diversity': diversity,
                'domain_boost': domain_component,
                'token_match_score': round(token_match_score,4),
                'matched_query_tokens': list(matched_tokens),
                'column_variety': column_variety,
                'total_score': round(total,4),
                'sample_columns': list(info.get('sample_columns') or [])
            })
        ranked_tables.sort(key=lambda x: (x['total_score'], x['token_match_score'], x['source_diversity']), reverse=True)
        return ranked_tables

    def _rank_aggregated_columns(self, all_columns: Dict) -> List[Dict]:
        """Rank columns by frequency and table association"""
        ranked_columns = []
        
        for column_name, info in all_columns.items():
            diversity = len(info['tables'])
            score = info['count'] + info['relevance_score'] + (0.5 * diversity)
            
            ranked_columns.append({
                'column_name': column_name,
                'frequency': info['count'],
                'sources': list(set(info['sources'])),
                'tables': list(info['tables']),
                'table_count': len(info['tables']),
                'total_score': score
            })
        
        # Sort by frequency and table diversity
        ranked_columns.sort(key=lambda x: (x['total_score'], x['table_count']), reverse=True)
        
        return ranked_columns

    # --- Normalization utilities (for future improved entity evaluation & matching) ---
    @staticmethod
    def normalize_identifier(name: str) -> str:
        """Normalize table/column/entity identifiers for comparison.
        Lowercase, remove non-alphanumerics, collapse underscores.
        """
        if not isinstance(name, str):
            return ''
        name = name.strip().lower()
        name = re.sub(r'[^a-z0-9_]+', '_', name)
        name = re.sub(r'_+', '_', name)
        return name.strip('_')

    def _generate_aggregated_response(self, user_query: str, intent: str, top_tables: List, 
                                    top_columns: List, key_results: Dict, parallel_results: Dict) -> str:
        """Generate comprehensive response from aggregated results"""
        
        if not top_tables and not top_columns:
            return "No results found across all search strategies."
        
        response_parts = []
        
        # Add intent and domain context
        domain = key_results.get('domain')
        if domain:
            response_parts.append(f"🎯 **{intent.replace('_', ' ').title()}** query in **{domain}** domain")
        else:
            response_parts.append(f"🎯 **{intent.replace('_', ' ').title()}** query analysis")
        
        # Top tables section with sample columns for richer semantic content
        if top_tables:
            response_parts.append(f"\n📋 **Top {min(5, len(top_tables))} Tables (multi-strategy ranking):**")
            for i, table in enumerate(top_tables[:5], 1):
                sources_str = ", ".join(table['sources'][:3])
                if len(table['sources']) > 3:
                    sources_str += f" (+{len(table['sources'])-3} more)"
                sample_cols = []
                # Retrieve sample columns from aggregator if available
                # (We look up in all_tables via stored details)
                # Locate aggregator by scanning original structures (cheap set comprehension)
                try:
                    # aggregator not directly passed; reconstruct via parallel_results domain details
                    # fallback: use key_results table_details or domain_insights
                    pass
                except Exception:
                    pass
                # We stored sample columns inside all_tables; reuse top_tables element if extended earlier
                if 'sample_columns' in table:
                    sample_cols = list(table['sample_columns'])
                # Provide fallback route using key_results
                if not sample_cols and key_results.get('table_details') and key_results['table_details'].get('table_name') == table['table_name']:
                    for c in key_results['table_details'].get('columns', [])[:5]:
                        if isinstance(c, dict) and c.get('name'):
                            sample_cols.append(c['name'])
                snippet = f"{i}. **{table['table_name']}** (freq {table['frequency']}, diversity {table['source_diversity']}, boost {table.get('domain_boost',0):.1f})"
                if sample_cols:
                    snippet += f" – cols: {', '.join(sample_cols[:4])}"
                response_parts.append(snippet)
        
        # Top columns section
        if top_columns:
            response_parts.append(f"\n🔍 **Top {min(5, len(top_columns))} Most Relevant Columns:**")
            for i, column in enumerate(top_columns[:5], 1):
                tables_str = ", ".join(list(column['tables'])[:2])  # Show first 2 tables
                if len(column['tables']) > 2:
                    tables_str += f" (+{len(column['tables'])-2} more)"
                
                response_parts.append(
                    f"{i}. **{column['column_name']}** "
                    f"(in: {tables_str}, frequency: {column['frequency']})"
                )
        
        # Add key insights
        if key_results.get('entities'):
            entities = key_results['entities']
            if entities.get('measurements') or entities.get('identifiers'):
                response_parts.append(f"\n⚡ **Extracted Entities:** "
                                     f"Measurements: {len(entities.get('measurements', []))}, "
                                     f"Identifiers: {len(entities.get('identifiers', []))}")
        
        # Add processing statistics
        success_count = sum(1 for status in parallel_results.values() 
                           if isinstance(status, dict) and status.get('status') == 'success')
        response_parts.append(f"\n📊 **Search Coverage:** {success_count}/6 processes executed successfully")
        
        return "\n".join(response_parts)

    # --- Extend entity extraction with KG table name detection ---
    def refresh_table_name_cache(self):
        """Populate a cache of table names from KG for better entity spotting."""
        names = set()
        try:
            with self.integrator.driver.session() as session:
                result = session.run("MATCH (t:Table) RETURN t.name as name LIMIT 500")
                for rec in result:
                    n = rec.get('name')
                    if n:
                        names.add(n)
        except Exception:
            pass
        self._kg_table_names = names

    def entity_enrich_with_tables(self, user_query: str, entities: Dict[str, List[str]]):
        if not hasattr(self, '_kg_table_names'):
            self.refresh_table_name_cache()
        q = user_query.lower()
        added = []
        for name in getattr(self, '_kg_table_names', set()):
            if name.lower() in q and name not in entities.get('tables', []):
                entities.setdefault('tables', []).append(name)
                added.append(name)
        if added:
            entities['detected_tables'] = added
        return entities

    def process_query(self, user_query: str) -> Dict:
        """Fallback method for compatibility - uses enhanced processing"""
        try:
            return self.enhanced_process_query(user_query)
        except Exception as e:
            # Ultimate fallback to basic semantic search
            intent, confidence = self.predict_intent(user_query)
            try:
                results = self.query_interface.semantic_search(user_query)
                return {
                    'type': 'basic_fallback',
                    'query': user_query,
                    'intent': intent,
                    'confidence': confidence,
                    'results': results[:5],
                    'response': self.generate_response({
                        'type': 'semantic_search',
                        'results': results[:5]
                    }),
                    'error': str(e)
                }
            except Exception as e2:
                return {
                    'type': 'error',
                    'query': user_query,
                    'intent': intent,
                    'error': str(e2),
                    'response': "I encountered an error processing your query. Please try rephrasing or contact support."
                }

    def _predict_intent_with_model(self, query: str) -> tuple[str, float | None]:
        """Use domain-specific model for intent prediction. Returns (label, confidence)."""
        try:
            if not (self.domain_model and self.domain_tokenizer):
                return self.detect_intent(query), None
            inputs = self.domain_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.domain_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
                conf, predicted_class = torch.max(probs, dim=-1)
            # Map to intent labels (from fine-tuned model)
            intent_labels = [
                'performance_analysis','power_optimization','spectrum_management','cell_configuration',
                'quality_assessment','traffic_analysis','fault_detection','capacity_planning',
                'interference_analysis','handover_optimization'
            ]
            label = intent_labels[predicted_class.item()] if predicted_class.item() < len(intent_labels) else 'semantic_search'
            return label, float(conf.item())
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self.detect_intent(query), None

    def _keyword_intent_baseline(self, query: str) -> str:
        """Lightweight keyword baseline that maps to fine-tuned intents without big models."""
        q = query.lower()
        rules = [
            ('performance_analysis', ['throughput','latency','kpi','performance','utilization','efficiency']),
            ('power_optimization', ['power','energy','dbm','consumption','efficiency','sleep']),
            ('spectrum_management', ['frequency','spectrum','bandwidth','carrier','channel','band']),
            ('cell_configuration', ['config','parameter','setting','threshold','cell','antenna','tilt']),
            ('quality_assessment', ['rsrp','rsrq','sinr','quality','noise','interference','cqi','bler']),
            ('traffic_analysis', ['traffic','volume','bytes','packets','usage','load','congestion']),
            ('fault_detection', ['fault','error','alarm','failure','issue','alert']),
            ('capacity_planning', ['capacity','headroom','forecast','plan','scaling','scale']),
            ('interference_analysis', ['interference','overlap','neighbor','pci','collision','noise']),
            ('handover_optimization', ['handover','handoff','mobility','roaming','x2','ng'])
        ]
        for label, kws in rules:
            if any(k in q for k in kws):
                return label
        return 'performance_analysis'  # reasonable default for RAN

    def predict_intent(self, query: str) -> tuple[str, float | None]:
        """Public API: returns (intent_label, confidence_or_None).
        Uses fine-tuned model if available, else falls back to keyword baseline.
        """
        if self.domain_model:
            return self._predict_intent_with_model(query)
        # baseline
        label = self._keyword_intent_baseline(query)
        return label, None
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from query using improved heuristics.
        Supports patterns like TableName.columnName, snake_case, and PascalCase.
        """
        # First get all known table names from Neo4j for direct matching
        try:
            with self.integrator.driver.session() as session:
                result = session.run("MATCH (t:Table) RETURN t.name as name")
                known_tables = [record['name'] for record in result]
                
            # Direct table name matching (case insensitive) - prioritize longest matches
            query_lower = query.lower()
            matched_tables = []
            for table_name in known_tables:
                if table_name.lower() in query_lower:
                    matched_tables.append((table_name, len(table_name)))
            
            if matched_tables:
                # Return the longest matching table name
                matched_tables.sort(key=lambda x: x[1], reverse=True)
                return matched_tables[0][0]
                
        except Exception:
            pass
        
        # Fallback to pattern matching
        # Dotted pattern Table.column
        m = re.search(r'([A-Za-z][A-Za-z0-9_]*)\.[A-Za-z][A-Za-z0-9_]*', query)
        if m:
            return m.group(1)
            
        # Snake_case tokens
        for tok in re.findall(r'[A-Za-z0-9_]+', query):
            if '_' in tok and len(tok) > 3:
                return tok
                
        # PascalCase words (join adjacent capitalized words)
        camel_words = re.findall(r'[A-Z][a-z]+', query)
        if len(camel_words) >= 2:
            # Try combinations of adjacent words, prioritize longer ones
            candidates = []
            for i in range(len(camel_words)):
                for j in range(i + 2, min(i + 5, len(camel_words) + 1)):  # 2-4 word combinations
                    combined = ''.join(camel_words[i:j])
                    if len(combined) > 8:  # Reasonable table name length
                        candidates.append((combined, len(combined)))
            
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                return candidates[0][0]
        
        return None

    def _expand_query_terms(self, query: str, intent: str | None = None) -> List[str]:
        """Build a list of search terms by expanding with synonyms and tokens."""
        q = query.lower()
        terms = set()
        
        # Add the original query
        terms.add(query.strip())
        
        # Raw words (longer ones first)
        words = re.findall(r'[a-z0-9_]+', q)
        for w in words:
            if len(w) > 2:
                terms.add(w)
        
        # Extract table.column patterns
        table_col_matches = re.findall(r'([A-Za-z][A-Za-z0-9_]*)\.[A-Za-z][A-Za-z0-9_]*', query)
        for table_name in table_col_matches:
            terms.add(table_name)
        
        # Extract camelCase/PascalCase words
        camel_words = re.findall(r'[A-Z][a-z]+|[a-z]+[A-Z][a-z]*', query)
        for word in camel_words:
            terms.add(word.lower())
        
        # Intent-based synonyms
        if intent in self.intent_domain_map:
            dom = self.intent_domain_map[intent]
            terms.add(dom)
            for s in self.query_synonyms.get(dom, []):
                terms.add(s)
        
        # Common domain synonyms if keywords present
        for dom, syns in self.query_synonyms.items():
            if any(syn in q for syn in syns):
                terms.add(dom)
                terms.update(syns)
        
        # Remove very short terms and sort by length (longer first)
        filtered_terms = [t for t in terms if len(t) > 2]
        ordered = sorted(filtered_terms, key=lambda x: (-len(x), x))
        
        # Cap list to avoid excessive queries but ensure we have enough variety
        return ordered[:12]
    
    def _format_contextualized_response(self, results: List[Dict], entities: Dict) -> str:
        """Format response for contextualized search"""
        if not results:
            return "No relevant tables found for your technical query."
        
        response = f"🔍 Technical search results (found {len(results)} matches)\n\n"
        response += f"**Detected entities:**\n"
        for entity_type, values in entities.items():
            if values:
                response += f"• {entity_type.replace('_', ' ').title()}: {', '.join(values)}\n"
        response += "\n"
        
        for result in results:
            response += f"📋 **{result['table_name']}**\n"
            response += f"• Matching columns: {', '.join(result.get('matching_columns', []))}\n"
            response += f"• Conceptual relationships: {result.get('conceptual_relationships', 0)}\n"
            response += f"• Row count: {result.get('row_count', 'N/A')}\n\n"
        return self._humanize_response(response)
    
    def _format_relationship_response(self, results: List[Dict], table_name: str) -> str:
        """Format response for multi-hop relationships"""
        if not results:
            return f"No related tables found for {table_name}."
        
        response = f"🔗 Multi-hop relationships for {table_name}\n\n"
        
        for result in results:
            response += f"📋 {result['related_table']}\n"
            response += f"• Relationship strength: {result.get('avg_relationship_strength', 0):.2f}\n"
            response += f"• Connection count: {result.get('connection_count', 0)}\n"
            response += f"• Shortest path: {result.get('shortest_path', 0)} hops\n"
            response += f"• Row count: {result.get('row_count', 'N/A')}\n\n"
        return self._humanize_response(response)
    
    def _format_clustering_response(self, results: List[Dict]) -> str:
        """Format response for semantic clustering"""
        if not results:
            return "No semantic clusters found."
        
        response = f"🎯 Semantic clusters (found {len(results)} clusters)\n\n"
        
        for result in results:
            response += f"🏷️ {result['cluster_name']}\n"
            response += f"• Confidence: {result.get('avg_confidence', 0):.2f}\n"
            response += f"• Tables involved: {result.get('table_count', 0)}\n"
            response += f"• Sample tables: {', '.join(result.get('sample_tables', [])[:3])}\n"
            response += f"• Sample columns: {', '.join(result.get('sample_columns', [])[:5])}\n\n"
        return self._humanize_response(response)
    
    def _format_optimized_response(self, results: List[Dict]) -> str:
        """Format response for optimized search results"""
        if not results:
            return "No results found."
        
        response = f"⚡ Optimized search results (found {len(results)} matches)\n\n"
        
        for result in results:
            response += f"📋 {result['table_name']} (relevance: {result.get('relevance_score', 0):.1f})\n"
            response += f"• Top columns: {', '.join(result.get('top_columns', []))}\n"
            response += f"• Relationships: {result.get('relationship_count', 0)}\n"
            response += f"• Row count: {result.get('row_count', 'N/A')}\n"
            if result.get('sample_related_tables'):
                response += f"• Related to: {', '.join(result['sample_related_tables'])}\n"
            response += "\n"
        return self._humanize_response(response)

    # --- Human-readable response enhancer (no big model) ---
    def _humanize_response(self, text: str) -> str:
        """Lightweight formatting for clearer, friendlier answers.
        - Normalizes bullets
        - Collapses excessive blank lines
        - Ensures sentence case for headings
        """
        try:
            lines = [l.rstrip() for l in text.splitlines()]
            out = []
            last_blank = False
            for l in lines:
                if not l.strip():
                    if not last_blank:
                        out.append("")
                    last_blank = True
                    continue
                last_blank = False
                # Normalize bullets
                if l.strip().startswith(('- ', '* ', '• ')):
                    l = '• ' + l.strip().lstrip('-*• ').strip()
                # Clean heading markers
                if l.startswith('📊') or l.startswith('🔍') or l.startswith('⚡') or l.startswith('🔗') or l.startswith('🎯') or l.startswith('📋'):
                    # lowercase parenthetical phrases
                    parts = l.split(' ', 1)
                    if len(parts) == 2:
                        tag, rest = parts
                        if rest and rest[0].islower():
                            rest = rest[:1].upper() + rest[1:]
                        l = f"{tag} {rest}"
                out.append(l)
            # Limit trailing blanks
            while out and not out[-1].strip():
                out.pop()
            return "\n".join(out)
        except Exception:
            return text
