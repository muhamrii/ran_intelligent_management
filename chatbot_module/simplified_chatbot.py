"""
Simplified Chatbot for Testing - Optimized for Performance
"""

from chatbot_module.chatbot import RANQueryInterface
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
import re
import time

class SimplifiedRANChatbot:
    """Simplified chatbot optimized for performance and explicit table detection"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.query_interface = RANQueryInterface(neo4j_integrator)
        
        # Cache table names for fast lookup
        self._all_table_names = None
        self._load_table_names()
    
    def _load_table_names(self):
        """Load all table names for fast lookup"""
        try:
            with self.integrator.driver.session() as session:
                result = session.run("MATCH (t:Table) RETURN t.name as name")
                self._all_table_names = [record['name'] for record in result]
        except Exception:
            self._all_table_names = []
    
    def _extract_table_name(self, query: str) -> str:
        """Extract table name with direct matching"""
        if not self._all_table_names:
            return None
            
        query_lower = query.lower()
        
        # Find longest matching table name
        matches = []
        for table_name in self._all_table_names:
            if table_name.lower() in query_lower:
                matches.append((table_name, len(table_name)))
        
        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return matches[0][0]
        
        return None
    
    def enhanced_process_query(self, user_query: str) -> dict:
        """Simplified processing focused on explicit table detection and basic search"""
        start_time = time.time()
        
        # Extract explicit table first
        explicit_table = self._extract_table_name(user_query)
        
        # If explicit table found, prioritize it
        if explicit_table:
            try:
                # Get details for explicit table
                table_details = self.query_interface.get_table_details(explicit_table)
                
                # Get related tables
                related_tables = self.query_interface.find_related_tables(explicit_table)
                
                # Build top tables with explicit table first
                top_tables = [{
                    'table_name': explicit_table,
                    'total_score': 100.0,  # Highest score for explicit mention
                    'sources': ['explicit_mention'],
                    'relevance_score': 50.0,
                    'explicit_boost': 50.0
                }]
                
                # Add related tables with lower scores
                for i, related in enumerate(related_tables[:5]):
                    related_table = related.get('related_table') or related.get('table_name')
                    if related_table and related_table != explicit_table:
                        top_tables.append({
                            'table_name': related_table,
                            'total_score': 20.0 - (i * 2),
                            'sources': ['related_to_explicit'],
                            'relevance_score': 10.0 - i
                        })
                
                processing_time = (time.time() - start_time) * 1000
                
                return {
                    'type': 'explicit_table_focused',
                    'query': user_query,
                    'top_tables': top_tables,
                    'key_results': {
                        'explicit_table': explicit_table,
                        'table_details': table_details
                    },
                    'processing_time_ms': processing_time,
                    'response': f"📋 Found explicit table: {explicit_table} with {len(related_tables)} related tables"
                }
                
            except Exception as e:
                pass
        
        # Fallback to basic semantic search
        try:
            results = self.query_interface.semantic_search(user_query, limit=10)
            
            top_tables = []
            for i, result in enumerate(results):
                table_name = result.get('table_name')
                if table_name:
                    top_tables.append({
                        'table_name': table_name,
                        'total_score': 10.0 - i,
                        'sources': ['semantic_search'],
                        'relevance_score': 5.0 - (i * 0.5)
                    })
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                'type': 'semantic_search',
                'query': user_query,
                'top_tables': top_tables,
                'processing_time_ms': processing_time,
                'response': f"🔍 Found {len(top_tables)} relevant tables"
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return {
                'type': 'error',
                'query': user_query,
                'top_tables': [],
                'processing_time_ms': processing_time,
                'response': f"❌ Error: {str(e)}"
            }
