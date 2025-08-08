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
            'frequency': r'\d+\s*(mhz|ghz|khz)',
            'power': r'\d+\s*(dbm|watts?|mw)',
            'coordinates': r'lat|lon|latitude|longitude|x_coord|y_coord',
            'timestamps': r'timestamp|time|date|created|updated',
            'identifiers': r'[a-z]+_id|id_[a-z]+|uuid|guid',
            'measurements': r'rsrp|rsrq|sinr|throughput|latency|kpi|metric'
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
    
    def __init__(self, neo4j_integrator, use_domain_model: bool = False):
        super().__init__(neo4j_integrator)
        
        # Enhanced components
        self.optimized_query = OptimizedQueryInterface(neo4j_integrator)
        self.entity_extractor = EnhancedRANEntityExtractor(neo4j_integrator)
        self.graph_traversal = IntelligentGraphTraversal(neo4j_integrator)
        
        # Load domain-specific model if available
        self.domain_model = None
        self.domain_tokenizer = None
        if use_domain_model:
            try:
                # These imports will be available after fine-tuning
                from transformers import AutoTokenizer, AutoModelForSequenceClassification
                self.domain_tokenizer = AutoTokenizer.from_pretrained("./ran_domain_model")
                self.domain_model = AutoModelForSequenceClassification.from_pretrained("./ran_domain_model")
                print("Domain-specific model loaded successfully")
            except Exception as e:
                print(f"Domain-specific model not found: {e}")
                print("Using default intent detection")
    
    def enhanced_process_query(self, user_query: str) -> Dict:
        """Enhanced query processing with optimizations"""
        # Use domain model for intent detection if available
        if self.domain_model:
            intent = self._predict_intent_with_model(user_query)
        else:
            intent = self.detect_intent(user_query)
        
        # Enhanced entity extraction
        entities = self.entity_extractor.extract_technical_entities(user_query)
        
        # Context-aware query generation
        if entities['measurements'] or entities['identifiers']:
            cypher_query, params = self.entity_extractor.contextualized_search(user_query, entities)
            results = self.optimized_query.cached_query(cypher_query, params)
            
            return {
                'type': 'contextualized_search',
                'query': user_query,
                'intent': intent,
                'entities': entities,
                'results': results,
                'response': self._format_contextualized_response(results, entities)
            }
        
        # Multi-hop relationship analysis for complex queries
        if 'related' in user_query.lower() or 'connected' in user_query.lower():
            table_name = self._extract_table_name(user_query)
            if table_name:
                results = self.graph_traversal.multi_hop_relationships(table_name)
                return {
                    'type': 'multi_hop_relationships',
                    'query': user_query,
                    'intent': intent,
                    'table': table_name,
                    'results': results,
                    'response': self._format_relationship_response(results, table_name)
                }
        
        # Semantic clustering for concept queries
        if intent == 'concept_search':
            results = self.graph_traversal.semantic_clustering()
            return {
                'type': 'semantic_clustering',
                'query': user_query,
                'intent': intent,
                'results': results,
                'response': self._format_clustering_response(results)
            }
        
        # Fallback to original processing with optimization
        original_result = super().process_query(user_query)
        
        # Use optimized search if it's a semantic search
        if original_result.get('type') == 'semantic_search':
            optimized_results = self.optimized_query.optimized_semantic_search(user_query)
            original_result['results'] = optimized_results
            original_result['response'] = self._format_optimized_response(optimized_results)
        
        return original_result
    
    def _predict_intent_with_model(self, query: str) -> str:
        """Use domain-specific model for intent prediction"""
        try:
            inputs = self.domain_tokenizer(query, return_tensors="pt", truncation=True, padding=True)
            
            with torch.no_grad():
                outputs = self.domain_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(predictions, dim=-1).item()
            
            # Map to intent labels (these would come from the fine-tuning module)
            intent_labels = ['performance_analysis', 'power_optimization', 'spectrum_management', 
                           'cell_configuration', 'quality_assessment', 'traffic_analysis',
                           'fault_detection', 'capacity_planning', 'interference_analysis', 'handover_optimization']
            
            return intent_labels[predicted_class] if predicted_class < len(intent_labels) else 'semantic_search'
        except Exception as e:
            print(f"Error in model prediction: {e}")
            return self.detect_intent(query)
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name from query using simple heuristics"""
        words = query.split()
        # Look for words that might be table names (simple heuristic)
        for word in words:
            if '_' in word and len(word) > 3:
                return word
        return None
    
    def _format_contextualized_response(self, results: List[Dict], entities: Dict) -> str:
        """Format response for contextualized search"""
        if not results:
            return "No relevant tables found for your technical query."
        
        response = f"🔍 **Technical Search Results** (Found {len(results)} matches)\n\n"
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
        
        return response
    
    def _format_relationship_response(self, results: List[Dict], table_name: str) -> str:
        """Format response for multi-hop relationships"""
        if not results:
            return f"No related tables found for {table_name}."
        
        response = f"🔗 **Multi-hop Relationships for {table_name}**\n\n"
        
        for result in results:
            response += f"📋 **{result['related_table']}**\n"
            response += f"• Relationship strength: {result.get('avg_relationship_strength', 0):.2f}\n"
            response += f"• Connection count: {result.get('connection_count', 0)}\n"
            response += f"• Shortest path: {result.get('shortest_path', 0)} hops\n"
            response += f"• Row count: {result.get('row_count', 'N/A')}\n\n"
        
        return response
    
    def _format_clustering_response(self, results: List[Dict]) -> str:
        """Format response for semantic clustering"""
        if not results:
            return "No semantic clusters found."
        
        response = f"🎯 **Semantic Clusters** (Found {len(results)} clusters)\n\n"
        
        for result in results:
            response += f"🏷️ **{result['cluster_name']}**\n"
            response += f"• Confidence: {result.get('avg_confidence', 0):.2f}\n"
            response += f"• Tables involved: {result.get('table_count', 0)}\n"
            response += f"• Sample tables: {', '.join(result.get('sample_tables', [])[:3])}\n"
            response += f"• Sample columns: {', '.join(result.get('sample_columns', [])[:5])}\n\n"
        
        return response
    
    def _format_optimized_response(self, results: List[Dict]) -> str:
        """Format response for optimized search results"""
        if not results:
            return "No results found."
        
        response = f"⚡ **Optimized Search Results** (Found {len(results)} matches)\n\n"
        
        for result in results:
            response += f"📋 **{result['table_name']}** (Relevance: {result.get('relevance_score', 0):.1f})\n"
            response += f"• Top columns: {', '.join(result.get('top_columns', []))}\n"
            response += f"• Relationships: {result.get('relationship_count', 0)}\n"
            response += f"• Row count: {result.get('row_count', 'N/A')}\n"
            if result.get('sample_related_tables'):
                response += f"• Related to: {', '.join(result['sample_related_tables'])}\n"
            response += "\n"
        
        return response
