"""
Knowledge Graph Builder Module
Contains all functions related to knowledge graph generation
"""

import pandas as pd
import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Any
import logging
import json
from datetime import datetime
import re

class RANNeo4jIntegrator:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = 0.7
        self.setup_constraints()
        
    def setup_constraints(self):
        """Create Neo4j constraints"""
        with self.driver.session() as session:
            constraints = [
                "CREATE CONSTRAINT table_name_unique IF NOT EXISTS FOR (t:Table) REQUIRE t.name IS UNIQUE",
                "CREATE CONSTRAINT column_id_unique IF NOT EXISTS FOR (c:Column) REQUIRE c.id IS UNIQUE",
                "CREATE CONSTRAINT concept_name_unique IF NOT EXISTS FOR (con:Concept) REQUIRE con.name IS UNIQUE"
            ]
            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    logging.info(f"Constraint might already exist: {e}")

    def extract_table_metadata(self, df: pd.DataFrame, table_name: str) -> Dict:
        """Extract comprehensive metadata from DataFrame"""
        metadata = {
            'name': table_name,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': {},
            'created_at': datetime.now().isoformat()
        }
        
        for col in df.columns:
            col_metadata = {
                'name': col,
                'data_type': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique(),
                'sample_values': df[col].dropna().astype(str).head(5).tolist()
            }
            
            # Statistical summary for numeric columns
            if df[col].dtype in ['int64', 'float64']:
                col_metadata['stats'] = {
                    'mean': float(df[col].mean()) if not df[col].isnull().all() else None,
                    'std': float(df[col].std()) if not df[col].isnull().all() else None,
                    'min': float(df[col].min()) if not df[col].isnull().all() else None,
                    'max': float(df[col].max()) if not df[col].isnull().all() else None
                }
            
            metadata['columns'][col] = col_metadata
            
        return metadata

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for text data"""
        return self.embedding_model.encode(texts)

    def discover_semantic_relationships(self, dataframes_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Discover different types of semantic relationships between tables and columns"""
        relationships = []
        
        # 1. NAME_SIMILARITY relationships
        name_relationships = self.discover_name_similarity(dataframes_dict)
        relationships.extend(name_relationships)
        
        # 2. VALUE_OVERLAP relationships
        value_relationships = self.discover_value_overlap(dataframes_dict)
        relationships.extend(value_relationships)
        
        # 3. PATTERN_MATCH relationships
        pattern_relationships = self.discover_pattern_matches(dataframes_dict)
        relationships.extend(pattern_relationships)
        
        # 4. REFERENCES relationships (potential foreign keys)
        reference_relationships = self.discover_references(dataframes_dict)
        relationships.extend(reference_relationships)
        
        return relationships
    
    def discover_name_similarity(self, dataframes_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Discover NAME_SIMILARITY relationships based on column name semantics"""
        relationships = []
        
        # Extract all column information
        all_columns = []
        for table_name, df in dataframes_dict.items():
            for col in df.columns:
                col_id = f"{table_name}.{col}"
                all_columns.append({
                    'id': col_id,
                    'name': col,
                    'table': table_name,
                    'text_for_embedding': col  # Use just column name for name similarity
                })
        
        # Create embeddings for column names
        column_texts = [col['text_for_embedding'] for col in all_columns]
        embeddings = self.create_embeddings(column_texts)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Find name-based relationships
        for i in range(len(all_columns)):
            for j in range(i + 1, len(all_columns)):
                similarity_score = similarity_matrix[i][j]
                
                if similarity_score > self.similarity_threshold:
                    col1 = all_columns[i]
                    col2 = all_columns[j]
                    
                    # Skip if same table
                    if col1['table'] == col2['table']:
                        continue
                    
                    # Determine similarity method
                    method = self.determine_similarity_method(col1['name'], col2['name'], similarity_score)
                    
                    relationships.append({
                        'type': 'NAME_SIMILARITY',
                        'source': col1['id'],
                        'target': col2['id'],
                        'similarity_score': float(similarity_score),
                        'method': method,
                        'confidence': float(similarity_score)
                    })
        
        return relationships
    
    def determine_similarity_method(self, name1: str, name2: str, similarity_score: float) -> str:
        """Determine the method used for name similarity"""
        if name1.lower() == name2.lower():
            return "exact_match"
        elif similarity_score > 0.9:
            return "semantic_embedding"
        else:
            return "fuzzy_match"

    def discover_value_overlap(self, dataframes_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Discover VALUE_OVERLAP relationships based on shared values"""
        relationships = []
        
        # Group columns by their unique values
        value_columns = {}
        
        for table_name, df in dataframes_dict.items():
            for col in df.columns:
                unique_vals = set(df[col].dropna().astype(str).tolist())
                
                # Only consider columns with reasonable number of unique values
                if 2 <= len(unique_vals) <= 100:
                    col_id = f"{table_name}.{col}"
                    value_columns[col_id] = {
                        'values': unique_vals,
                        'total_count': len(df[col].dropna())
                    }
        
        # Find columns with significant value overlap
        col_ids = list(value_columns.keys())
        for i in range(len(col_ids)):
            for j in range(i + 1, len(col_ids)):
                col1_id = col_ids[i]
                col2_id = col_ids[j]
                
                # Skip if same table
                if col1_id.split('.')[0] == col2_id.split('.')[0]:
                    continue
                
                vals1 = value_columns[col1_id]['values']
                vals2 = value_columns[col2_id]['values']
                
                # Calculate overlap metrics
                intersection = vals1.intersection(vals2)
                union = vals1.union(vals2)
                jaccard_sim = len(intersection) / len(union) if len(union) > 0 else 0
                overlap_percentage = len(intersection) / min(len(vals1), len(vals2)) if min(len(vals1), len(vals2)) > 0 else 0
                
                if jaccard_sim > 0.3 or overlap_percentage > 0.5:  # Multiple thresholds
                    relationships.append({
                        'type': 'VALUE_OVERLAP',
                        'source': col1_id,
                        'target': col2_id,
                        'jaccard_similarity': float(jaccard_sim),
                        'overlap_percentage': float(overlap_percentage),
                        'shared_values_count': len(intersection),
                        'shared_sample_values': list(intersection)[:5]  # Sample of shared values
                    })
        
        return relationships
    
    def discover_pattern_matches(self, dataframes_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Discover PATTERN_MATCH relationships based on data patterns"""
        relationships = []
        
        # Define common patterns
        patterns = {
            'id_field': r'.*id$|.*_id$|^id.*',
            'timestamp': r'.*time.*|.*date.*|.*_at$|.*_on$',
            'status_field': r'.*status.*|.*state.*',
            'name_field': r'.*name.*|.*_nm$',
            'count_field': r'.*count.*|.*cnt.*|.*num.*',
            'code_field': r'.*code.*|.*cd$'
        }
        
        # Categorize columns by patterns
        pattern_columns = {pattern: [] for pattern in patterns}
        
        for table_name, df in dataframes_dict.items():
            for col in df.columns:
                col_id = f"{table_name}.{col}"
                col_lower = col.lower()
                
                for pattern_name, pattern_regex in patterns.items():
                    if re.match(pattern_regex, col_lower):
                        pattern_columns[pattern_name].append({
                            'id': col_id,
                            'name': col,
                            'table': table_name,
                            'data_type': str(df[col].dtype)
                        })
                        break
        
        # Create relationships between columns matching same patterns
        for pattern_name, columns in pattern_columns.items():
            if len(columns) < 2:
                continue
                
            for i in range(len(columns)):
                for j in range(i + 1, len(columns)):
                    col1 = columns[i]
                    col2 = columns[j]
                    
                    # Skip if same table
                    if col1['table'] == col2['table']:
                        continue
                    
                    # Higher confidence if data types also match
                    confidence = 0.8
                    if col1['data_type'] == col2['data_type']:
                        confidence = 0.9
                    
                    relationships.append({
                        'type': 'PATTERN_MATCH',
                        'source': col1['id'],
                        'target': col2['id'],
                        'pattern_type': pattern_name,
                        'confidence': confidence,
                        'pattern_description': f"Both fields follow {pattern_name} pattern"
                    })
        
        return relationships
    
    def discover_references(self, dataframes_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Discover REFERENCES relationships (potential foreign keys)"""
        relationships = []
        
        # Find potential ID columns (primary keys)
        id_columns = {}
        
        for table_name, df in dataframes_dict.items():
            for col in df.columns:
                col_lower = col.lower()
                
                # Check if column looks like an ID field
                if ('id' in col_lower and df[col].nunique() == len(df[col].dropna()) and 
                    df[col].nunique() > len(df) * 0.8):  # High uniqueness
                    
                    col_id = f"{table_name}.{col}"
                    unique_values = set(df[col].dropna().astype(str).tolist())
                    id_columns[col_id] = {
                        'values': unique_values,
                        'name': col,
                        'table': table_name,
                        'is_primary': df[col].nunique() == len(df[col].dropna())
                    }
        
        # Find potential foreign key relationships
        all_cols = {}
        for table_name, df in dataframes_dict.items():
            for col in df.columns:
                col_id = f"{table_name}.{col}"
                all_cols[col_id] = {
                    'values': set(df[col].dropna().astype(str).tolist()),
                    'name': col,
                    'table': table_name
                }
        
        # Check for foreign key relationships
        for potential_fk_id, fk_info in all_cols.items():
            for pk_id, pk_info in id_columns.items():
                # Skip if same table or same column
                if (fk_info['table'] == pk_info['table'] or 
                    potential_fk_id == pk_id):
                    continue
                
                # Check if FK values are subset of PK values
                fk_values = fk_info['values']
                pk_values = pk_info['values']
                
                if len(fk_values) == 0:
                    continue
                
                match_percentage = len(fk_values.intersection(pk_values)) / len(fk_values)
                
                if match_percentage > 0.8:  # High match indicates potential FK relationship
                    confidence = match_percentage * (0.9 if pk_info['is_primary'] else 0.7)
                    
                    relationships.append({
                        'type': 'REFERENCES',
                        'source': potential_fk_id,  # Foreign key
                        'target': pk_id,           # Primary key
                        'confidence': float(confidence),
                        'match_percentage': float(match_percentage),
                        'reference_type': 'primary_key' if pk_info['is_primary'] else 'lookup'
                    })
        
        return relationships

    def create_nodes_and_relationships(self, dataframes_dict: Dict[str, pd.DataFrame]):
        """Create all nodes and relationships in Neo4j"""
        
        with self.driver.session() as session:
            # Create table nodes
            for table_name, df in dataframes_dict.items():
                metadata = self.extract_table_metadata(df, table_name)
                
                # Create table node
                session.run("""
                    MERGE (t:Table {name: $name})
                    SET t.row_count = $row_count,
                        t.column_count = $column_count,
                        t.created_at = $created_at
                """, **metadata)
                
                # Create column nodes and relationships
                for col_name, col_metadata in metadata['columns'].items():
                    col_id = f"{table_name}.{col_name}"
                    
                    session.run("""
                        MERGE (c:Column {id: $col_id})
                        SET c.name = $name,
                            c.data_type = $data_type,
                            c.null_count = $null_count,
                            c.unique_count = $unique_count,
                            c.sample_values = $sample_values
                    """, 
                    col_id=col_id,
                    name=col_name,
                    data_type=col_metadata['data_type'],
                    null_count=col_metadata['null_count'],
                    unique_count=col_metadata['unique_count'],
                    sample_values=col_metadata['sample_values']
                    )
                    
                    # Create relationship between table and column
                    session.run("""
                        MATCH (t:Table {name: $table_name})
                        MATCH (c:Column {id: $col_id})
                        MERGE (t)-[:HAS_COLUMN]->(c)
                    """, table_name=table_name, col_id=col_id)
            
            # Create semantic relationships with specific types
            relationships = self.discover_semantic_relationships(dataframes_dict)
            
            for rel in relationships:
                if rel['type'] == 'NAME_SIMILARITY':
                    session.run("""
                        MATCH (c1:Column {id: $source})
                        MATCH (c2:Column {id: $target})
                        MERGE (c1)-[r:NAME_SIMILARITY]-(c2)
                        SET r.similarity_score = $similarity_score,
                            r.method = $method,
                            r.confidence = $confidence
                    """, **rel)
                    
                elif rel['type'] == 'VALUE_OVERLAP':
                    session.run("""
                        MATCH (c1:Column {id: $source})
                        MATCH (c2:Column {id: $target})
                        MERGE (c1)-[r:VALUE_OVERLAP]-(c2)
                        SET r.jaccard_similarity = $jaccard_similarity,
                            r.overlap_percentage = $overlap_percentage,
                            r.shared_values_count = $shared_values_count,
                            r.shared_sample_values = $shared_sample_values
                    """, **rel)
                    
                elif rel['type'] == 'PATTERN_MATCH':
                    session.run("""
                        MATCH (c1:Column {id: $source})
                        MATCH (c2:Column {id: $target})
                        MERGE (c1)-[r:PATTERN_MATCH]-(c2)
                        SET r.pattern_type = $pattern_type,
                            r.confidence = $confidence,
                            r.pattern_description = $pattern_description
                    """, **rel)
                    
                elif rel['type'] == 'REFERENCES':
                    session.run("""
                        MATCH (c1:Column {id: $source})
                        MATCH (c2:Column {id: $target})
                        MERGE (c1)-[r:REFERENCES]->(c2)
                        SET r.confidence = $confidence,
                            r.match_percentage = $match_percentage,
                            r.reference_type = $reference_type
                    """, **rel)
            
            # Generate conceptual groupings
            self.generate_conceptual_groups()

    def generate_conceptual_groups(self, min_cluster_size: int = 3):
        """Generate CONCEPTUAL_GROUP relationships based on clustering"""
        
        with self.driver.session() as session:
            # Get all columns with their embeddings
            result = session.run("""
                MATCH (c:Column)
                RETURN c.id as col_id, c.name as col_name
            """)
            
            columns = [(record['col_id'], record['col_name']) for record in result]
            
            if len(columns) < min_cluster_size:
                return
            
            # Create embeddings for clustering
            column_names = [col[1] for col in columns]
            embeddings = self.create_embeddings(column_names)
            
            # Perform clustering
            n_clusters = min(10, len(columns) // 3)  # Dynamic cluster count
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Create CONCEPTUAL_GROUP relationships
            for cluster_id in range(n_clusters):
                cluster_columns = [columns[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                
                if len(cluster_columns) >= min_cluster_size:
                    # Calculate cluster coherence (average similarity within cluster)
                    cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    if len(cluster_embeddings) > 1:
                        cluster_similarity = cosine_similarity(cluster_embeddings).mean()
                        confidence = float(cluster_similarity)
                    else:
                        confidence = 1.0
                    
                    # Determine semantic category based on common words
                    semantic_category = self.determine_semantic_category([col[1] for col in cluster_columns])
                    
                    # Create relationships between all columns in the same cluster
                    for i in range(len(cluster_columns)):
                        for j in range(i + 1, len(cluster_columns)):
                            col1_id, col1_name = cluster_columns[i]
                            col2_id, col2_name = cluster_columns[j]
                            
                            # Skip if same table (conceptual groups are cross-table)
                            if col1_id.split('.')[0] == col2_id.split('.')[0]:
                                continue
                            
                            session.run("""
                                MATCH (c1:Column {id: $source})
                                MATCH (c2:Column {id: $target})
                                MERGE (c1)-[r:CONCEPTUAL_GROUP]-(c2)
                                SET r.cluster_id = $cluster_id,
                                    r.cluster_confidence = $confidence,
                                    r.semantic_category = $semantic_category
                            """, 
                            source=col1_id,
                            target=col2_id,
                            cluster_id=f"cluster_{cluster_id}",
                            confidence=confidence,
                            semantic_category=semantic_category
                            )
    
    def determine_semantic_category(self, column_names: List[str]) -> str:
        """Determine semantic category based on common patterns in column names"""
        all_names = ' '.join(column_names).lower()
        
        categories = {
            'identifiers': ['id', 'key', 'code', 'ref'],
            'measurements': ['count', 'size', 'length', 'width', 'height', 'weight', 'volume'],
            'performance_metrics': ['rate', 'speed', 'throughput', 'latency', 'efficiency'],
            'status_fields': ['status', 'state', 'flag', 'active', 'enabled'],
            'temporal': ['time', 'date', 'created', 'updated', 'modified'],
            'configuration': ['config', 'setting', 'parameter', 'option', 'preference'],
            'network_specific': ['cell', 'frequency', 'power', 'signal', 'antenna', 'handover']
        }
        
        category_scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in all_names)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'general'

    def close(self):
        """Close the Neo4j driver connection"""
        if self.driver:
            self.driver.close()
