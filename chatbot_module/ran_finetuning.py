"""
RAN Domain-Specific Fine-tuning Module
Fine-tunes language models specifically for RAN knowledge graph queries and entities
"""

import json
import torch
import numpy as np
import os
import gc
from typing import Dict, List, Tuple, Any
from datasets import Dataset
import logging
from datetime import datetime

# These will need to be installed: pip install transformers datasets torch
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        Trainer, 
        TrainingArguments,
        pipeline
    )
    import transformers
    TRANSFORMERS_AVAILABLE = True
    TRANSFORMERS_VERSION = transformers.__version__
    print(f"Transformers version: {TRANSFORMERS_VERSION}")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TRANSFORMERS_VERSION = None
    print("Warning: transformers library not installed. Run: pip install transformers datasets torch")

def check_transformers_compatibility():
    """Check transformers version compatibility"""
    if not TRANSFORMERS_AVAILABLE:
        return False, "Transformers not installed"
    
    try:
        version_parts = TRANSFORMERS_VERSION.split('.')
        major = int(version_parts[0])
        minor = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        if major < 4:
            return False, f"Transformers version {TRANSFORMERS_VERSION} too old. Please upgrade to 4.0+"
        elif major == 4 and minor < 20:
            return True, f"Transformers {TRANSFORMERS_VERSION} - some features may be limited"
        else:
            return True, f"Transformers {TRANSFORMERS_VERSION} - fully compatible"
    except:
        return True, "Version check failed, proceeding anyway"

class RANDomainModelTrainer:
    """Train a RAN-specific model for intent classification and entity recognition"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.model_name = "distilbert-base-uncased"
        
        # RAN-specific intents for fine-tuning
        self.ran_intents = {
            'performance_analysis': 'Analyze network performance metrics and KPIs',
            'power_optimization': 'Optimize power consumption and efficiency',
            'spectrum_management': 'Manage frequency spectrum allocation and bandwidth',
            'cell_configuration': 'Configure cell parameters and settings',
            'quality_assessment': 'Assess signal quality and coverage metrics',
            'traffic_analysis': 'Analyze network traffic patterns and load',
            'fault_detection': 'Detect and diagnose network faults and issues',
            'capacity_planning': 'Plan network capacity and resource allocation',
            'interference_analysis': 'Analyze and mitigate interference sources',
            'handover_optimization': 'Optimize handover procedures and mobility'
        }
        
        # RAN domain vocabulary - expanded based on your semantic categories
        self.ran_vocabulary = {
            'performance_terms': ['kpi', 'throughput', 'latency', 'performance', 'metric', 'benchmark', 'efficiency'],
            'power_terms': ['power', 'energy', 'consumption', 'efficiency', 'dbm', 'watts', 'battery', 'management'],
            'spectrum_terms': ['frequency', 'spectrum', 'bandwidth', 'channel', 'carrier', 'mhz', 'ghz', 'allocation'],
            'cell_terms': ['cell', 'site', 'antenna', 'base_station', 'node', 'enb', 'gnb', 'tower'],
            'quality_terms': ['rsrp', 'rsrq', 'sinr', 'quality', 'signal', 'coverage', 'interference', 'assessment'],
            'traffic_terms': ['traffic', 'volume', 'data', 'load', 'usage', 'throughput', 'session', 'analysis'],
            'mobility_terms': ['handover', 'mobility', 'roaming', 'tracking', 'movement', 'transition', 'management'],
            'config_terms': ['config', 'parameter', 'setting', 'threshold', 'value', 'configuration', 'parameters'],
            'topology_terms': ['topology', 'network', 'structure', 'layout', 'connection', 'architecture', 'design'],
            'timing_terms': ['timing', 'synchronization', 'sync', 'clock', 'time', 'coordination', 'alignment'],
            'security_terms': ['security', 'authentication', 'encryption', 'protection', 'access', 'features', 'safety'],
            'general_terms': ['general', 'overview', 'summary', 'information', 'data', 'metrics', 'statistics']
        }
    
    def generate_training_data(self) -> List[Dict]:
        """Generate domain-specific training data from KG with proper entities"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for training")
            
        training_data = []
        
        try:
            # First, get real data from the knowledge graph
            with self.integrator.driver.session() as session:
                print("🔍 Extracting real KG data for training...")
                
                # Get comprehensive data from the knowledge graph
                kg_data_query = """
                    MATCH (c1:Column)-[r:CONCEPTUAL_GROUP]-(c2:Column)
                    WHERE r.semantic_category IS NOT NULL
                    MATCH (t:Table)-[:HAS_COLUMN]->(c1)
                    WITH r.semantic_category as category, 
                         c1.name as column_name,
                         t.name as table_name,
                         coalesce(c1.data_type, 'varchar') as data_type,
                         coalesce(t.row_count, 0) as row_count,
                         coalesce(t.column_count, 0) as column_count,
                         count(r) as relationship_count
                    RETURN category, column_name, table_name, data_type, row_count, column_count, relationship_count
                    ORDER BY relationship_count DESC
                    LIMIT 50000
                """
                
                result = session.run(kg_data_query)
                kg_entities = []
                category_data = {}
                
                for record in result:
                    category = record['category']
                    column_name = record['column_name'] 
                    table_name = record['table_name']
                    data_type = record['data_type']
                    row_count = record['row_count'] or 0
                    column_count = record['column_count'] or 0
                    relationship_count = record['relationship_count'] or 0
                    
                    entity_info = {
                        'table_name': table_name,
                        'column_name': column_name,
                        'data_type': data_type,
                        'category': category,
                        'row_count': row_count,
                        'column_count': column_count,
                        'relationship_count': relationship_count
                    }
                    kg_entities.append(entity_info)
                    
                    # Group by category for better organization
                    if category not in category_data:
                        category_data[category] = []
                    category_data[category].append(entity_info)
                
                print(f"📊 Extracted {len(kg_entities)} real entities from {len(category_data)} categories")
                
                # Generate training data using real KG entities
                for category, entities in category_data.items():
                    intent = self._map_category_to_intent(category)
                    templates = self._get_category_templates(category)
                    
                    # Generate diverse queries for this category
                    for entity in entities[:30]:  # Limit per category to control size
                        table_name = entity['table_name']
                        column_name = entity['column_name']
                        data_type = entity['data_type']
                        
                        for template in templates[:2]:  # Use first 2 templates per entity
                            try:
                                # Generate query with real entity names
                                query = template.format(
                                    column=column_name, 
                                    table=table_name
                                )
                                
                                # Always create rich entities for every training sample
                                entities_in_query = {
                                    'table_name': table_name,
                                    'column_name': column_name,
                                    'table_type': 'database_table',
                                    'column_type': data_type,
                                    'semantic_category': category,
                                    'domain_type': category,
                                    'query_type': self._infer_query_type(query),
                                    'entity_confidence': 'high',
                                    'extraction_source': 'kg_direct_reference'
                                }
                                
                                # Add RAN-specific entities
                                ran_entities = self._extract_ran_specific_entities(query, column_name, table_name)
                                entities_in_query.update(ran_entities)
                                
                                training_data.append({
                                    'text': query,
                                    'intent': intent,
                                    'label': list(self.ran_intents.keys()).index(intent),
                                    'entities': entities_in_query,
                                    'metadata': {
                                        'category': category,
                                        'data_type': data_type,
                                        'row_count': entity['row_count'],
                                        'relationship_count': entity.get('relationship_count', 0),
                                        'source': 'kg_extraction'
                                    }
                                })
                                
                            except (ValueError, KeyError, IndexError) as e:
                                # Still create a sample with basic entities if template fails
                                basic_query = f"Show {column_name} from {table_name}"
                                basic_entities = {
                                    'table_name': table_name,
                                    'column_name': column_name,
                                    'semantic_category': category,
                                    'entity_confidence': 'medium',
                                    'extraction_source': 'kg_fallback'
                                }
                                
                                training_data.append({
                                    'text': basic_query,
                                    'intent': intent,
                                    'label': list(self.ran_intents.keys()).index(intent),
                                    'entities': basic_entities,
                                    'metadata': {
                                        'category': category,
                                        'source': 'kg_fallback'
                                    }
                                })
                                continue
                
                # Generate entity-rich general queries
                general_queries = self._generate_entity_rich_general_queries(kg_entities[:100])
                training_data.extend(general_queries)
                
                print(f"✅ Generated {len(training_data)} training samples with entities")
                
        except Exception as e:
            logging.error(f"Error generating KG training data: {e}")
            print(f"⚠️ Falling back to synthetic data generation due to: {e}")
            # Enhanced fallback with entity simulation
            training_data = self._generate_enhanced_synthetic_training_data()
        
        # Ensure we have reasonable amount of data
        if len(training_data) < 100:
            print(f"⚠️ Low training data count ({len(training_data)}), adding synthetic samples...")
            synthetic_data = self._generate_enhanced_synthetic_training_data()
            training_data.extend(synthetic_data)
        
        return training_data
    
    def _get_category_templates(self, category: str) -> List[str]:
        """Get query templates for specific RAN categories"""
        templates = {
            # Network topology templates (2.7M relationships)
            'network_topology': [
                "Show network topology data from {column} in {table}",
                "Find topology relationships for {column}",
                "What network connections are in {table}?",
                "Display topology structure using {column}",
                "Analyze network layout from {table}",
                "Get topology configuration for {column}"
            ],
            
            # Quality templates (1.6M relationships)  
            'quality': [
                "Show signal quality from {column}",
                "Analyze quality metrics in {table}",
                "What is the quality assessment for {column}?",
                "Get quality measurements from {table}",
                "Compare quality indicators in {column}",
                "Find quality issues using {table}"
            ],
            
            # Frequency templates (884K relationships)
            'frequency': [
                "Get frequency data from {column} in {table}",
                "Show frequency allocation for {column}",
                "What frequency bands are in {table}?",
                "Analyze frequency usage from {column}",
                "Find frequency conflicts in {table}",
                "Display frequency planning from {column}"
            ],
            
            # General templates (472K relationships)
            'general': [
                "Show general data from {column} in {table}",
                "Get overview information for {column}",
                "What general metrics are in {table}?",
                "Display summary data from {column}",
                "Find general information in {table}",
                "Analyze overall data from {column}"
            ],
            
            # Configuration parameters templates (437K relationships)
            'configuration_parameters': [
                "Show configuration parameters from {column} in {table}",
                "Get parameter settings for {column}",
                "What parameters are configured in {table}?",
                "Display configuration values from {column}",
                "Find parameter changes in {table}",
                "Analyze configuration data from {column}"
            ],
            
            # Topology templates (434K relationships)
            'topology': [
                "Show topology data from {column} in {table}",
                "Get topology information for {column}",
                "What topology is defined in {table}?",
                "Display network structure from {column}",
                "Find topology connections in {table}",
                "Analyze topology layout from {column}"
            ],
            
            # Traffic analysis templates (349K relationships)
            'traffic_analysis': [
                "Analyze traffic data from {column} in {table}",
                "Show traffic patterns for {column}",
                "What traffic metrics are in {table}?",
                "Get traffic volume from {column}",
                "Find traffic trends in {table}",
                "Display traffic statistics from {column}"
            ],
            
            # Frequency spectrum templates (208K relationships)
            'frequency_spectrum': [
                "Show spectrum data from {column} in {table}",
                "Get spectrum allocation for {column}",
                "What spectrum usage is in {table}?",
                "Analyze spectrum efficiency from {column}",
                "Find spectrum conflicts in {table}",
                "Display spectrum planning from {column}"
            ],
            
            # Timing synchronization templates (198K relationships)
            'timing_synchronization': [
                "Show timing data from {column} in {table}",
                "Get synchronization info for {column}",
                "What timing parameters are in {table}?",
                "Analyze timing accuracy from {column}",
                "Find timing issues in {table}",
                "Display sync status from {column}"
            ],
            
            # Traffic templates (147K relationships)
            'traffic': [
                "Show traffic data from {column} in {table}",
                "Get traffic load for {column}",
                "What traffic information is in {table}?",
                "Analyze traffic flow from {column}",
                "Find traffic bottlenecks in {table}",
                "Display traffic usage from {column}"
            ],
            
            # Quality metrics templates (104K relationships)
            'quality_metrics': [
                "Show quality metrics from {column} in {table}",
                "Get quality KPIs for {column}",
                "What quality indicators are in {table}?",
                "Analyze quality performance from {column}",
                "Find quality degradation in {table}",
                "Display quality trends from {column}"
            ],
            
            # Configuration templates (75K relationships)
            'configuration': [
                "Show configuration data from {column} in {table}",
                "Get config settings for {column}",
                "What configuration is in {table}?",
                "Display config values from {column}",
                "Find configuration changes in {table}",
                "Analyze config parameters from {column}"
            ],
            
            # Performance metrics templates (13K relationships)
            'performance_metrics': [
                "Show performance metrics from {column} in {table}",
                "Get performance data for {column}",
                "What performance indicators are in {table}?",
                "Analyze performance trends from {column}",
                "Find performance issues in {table}",
                "Display performance KPIs from {column}"
            ],
            
            # Power management templates (9K relationships)
            'power_management': [
                "Show power data from {column} in {table}",
                "Get power consumption for {column}",
                "What power metrics are in {table}?",
                "Analyze power efficiency from {column}",
                "Find power issues in {table}",
                "Display power usage from {column}"
            ],
            
            # Mobility management templates (130 relationships)
            'mobility_management': [
                "Show mobility data from {column} in {table}",
                "Get mobility metrics for {column}",
                "What mobility information is in {table}?",
                "Analyze mobility patterns from {column}",
                "Find mobility issues in {table}",
                "Display handover data from {column}"
            ],
            
            # Security features templates (76 relationships)
            'security_features': [
                "Show security data from {column} in {table}",
                "Get security settings for {column}",
                "What security features are in {table}?",
                "Analyze security status from {column}",
                "Find security vulnerabilities in {table}",
                "Display security configuration from {column}"
            ],
            
            # Legacy mappings for backward compatibility
            'power_related': [
                "Show me {column} values from {table}",
                "What is the average {column} in {table}?",
                "Find tables with power consumption data like {column}",
                "Compare {column} across different cells",
                "Analyze power efficiency using {column}",
                "Get power consumption metrics from {column}"
            ],
            'frequency_related': [
                "Get frequency data from {column} in {table}",
                "Show spectrum allocation for {column}",
                "Find all frequency-related columns similar to {column}",
                "Analyze spectrum usage in {table}",
                "What frequency bands are in {column}?",
                "Show bandwidth allocation from {table}"
            ],
            'cell_identifiers': [
                "Find cell information for {column}",
                "Show all data for cell {column}",
                "Get cell configuration from {table}",
                "List cells in {table}",
                "What cells are configured in {column}?",
                "Show cell topology from {table}"
            ]
        }
        return templates.get(category, templates['performance_metrics'])
    
    def _map_category_to_intent(self, category: str) -> str:
        """Map semantic categories to intents"""
        mapping = {
            # Direct mappings for your actual semantic categories
            'network_topology': 'cell_configuration',      # 2.7M relationships - network structure
            'quality': 'quality_assessment',                # 1.6M relationships - signal quality
            'frequency': 'spectrum_management',             # 884K relationships - frequency allocation
            'general': 'performance_analysis',              # 472K relationships - general metrics
            'configuration_parameters': 'cell_configuration', # 437K relationships - parameter settings
            'topology': 'cell_configuration',               # 434K relationships - network topology
            'traffic_analysis': 'traffic_analysis',         # 349K relationships - traffic patterns
            'frequency_spectrum': 'spectrum_management',    # 208K relationships - spectrum usage
            'timing_synchronization': 'performance_analysis', # 198K relationships - timing metrics
            'traffic': 'traffic_analysis',                  # 147K relationships - traffic data
            'quality_metrics': 'quality_assessment',        # 104K relationships - quality KPIs
            'configuration': 'cell_configuration',          # 75K relationships - config settings
            'performance_metrics': 'performance_analysis',  # 13K relationships - performance KPIs
            'power_management': 'power_optimization',       # 9K relationships - power efficiency
            'mobility_management': 'handover_optimization', # 130 relationships - mobility/handover
            'security_features': 'fault_detection',         # 76 relationships - security monitoring
            
            # Legacy mappings for backward compatibility
            'power_related': 'power_optimization',
            'frequency_related': 'spectrum_management',
            'cell_identifiers': 'cell_configuration',
            'quality_metrics_legacy': 'quality_assessment',
            'traffic_counters': 'traffic_analysis',
            'fault_indicators': 'fault_detection',
            'capacity_metrics': 'capacity_planning',
            'interference_data': 'interference_analysis',
            'mobility_data': 'handover_optimization'
        }
        return mapping.get(category, 'performance_analysis')
    
    def _extract_entities_from_query(self, query: str, table_name: str, column_name: str, 
                                   data_type: str, category: str) -> Dict[str, str]:
        """Extract entities from query text with their positions and types"""
        entities = {}
        
        # Always populate basic entities
        entities['table_name'] = table_name if table_name else 'unknown_table'
        entities['column_name'] = column_name if column_name else 'unknown_column'
        entities['table_type'] = 'database_table'
        entities['column_type'] = data_type or 'varchar'
        entities['semantic_category'] = category or 'general'
        entities['domain_type'] = category or 'general'
        entities['query_type'] = self._infer_query_type(query)
        
        # Extract RAN-specific entities based on patterns
        ran_entities = self._extract_ran_specific_entities(query, column_name, table_name)
        entities.update(ran_entities)
        
        # Ensure we always have confidence and source
        if 'entity_confidence' not in entities:
            entities['entity_confidence'] = 'high'
        if 'extraction_source' not in entities:
            entities['extraction_source'] = 'kg_based_extraction'
        
        return entities
    
    def _extract_ran_specific_entities(self, query: str, column_name: str, table_name: str) -> Dict[str, str]:
        """Extract RAN-specific entities like cell IDs, frequencies, power values"""
        entities = {}
        query_lower = query.lower()
        
        # Extract frequency-related entities
        if any(freq_term in query_lower for freq_term in ['frequency', 'freq', 'mhz', 'ghz', 'spectrum', 'band']):
            entities['domain_type'] = 'frequency_spectrum'
            if 'mhz' in query_lower or 'ghz' in query_lower:
                entities['unit_type'] = 'frequency_unit'
        
        # Extract power-related entities
        if any(power_term in query_lower for power_term in ['power', 'dbm', 'watts', 'energy', 'consumption']):
            entities['domain_type'] = 'power_management'
            if 'dbm' in query_lower or 'watts' in query_lower:
                entities['unit_type'] = 'power_unit'
        
        # Extract cell-related entities
        if any(cell_term in query_lower for cell_term in ['cell', 'site', 'antenna', 'base_station', 'enb', 'gnb']):
            entities['domain_type'] = 'cell_configuration'
            entities['network_element'] = 'cellular_infrastructure'
        
        # Extract quality-related entities
        if any(quality_term in query_lower for quality_term in ['quality', 'rsrp', 'rsrq', 'sinr', 'signal']):
            entities['domain_type'] = 'quality_assessment'
            entities['measurement_type'] = 'radio_quality'
        
        # Extract traffic-related entities
        if any(traffic_term in query_lower for traffic_term in ['traffic', 'throughput', 'volume', 'load', 'usage']):
            entities['domain_type'] = 'traffic_analysis'
            entities['measurement_type'] = 'network_traffic'
        
        # Extract topology-related entities
        if any(topo_term in query_lower for topo_term in ['topology', 'network', 'connection', 'structure']):
            entities['domain_type'] = 'network_topology'
            entities['structure_type'] = 'network_architecture'
        
        # Extract configuration-related entities
        if any(config_term in query_lower for config_term in ['config', 'parameter', 'setting', 'threshold']):
            entities['domain_type'] = 'configuration_management'
            entities['config_type'] = 'system_parameter'
        
        # Extract timing-related entities
        if any(timing_term in query_lower for timing_term in ['timing', 'sync', 'synchronization', 'clock']):
            entities['domain_type'] = 'timing_synchronization'
            entities['sync_type'] = 'network_timing'
        
        # Extract mobility-related entities
        if any(mobility_term in query_lower for mobility_term in ['handover', 'mobility', 'roaming', 'movement']):
            entities['domain_type'] = 'mobility_management'
            entities['mobility_type'] = 'user_mobility'
        
        # Extract security-related entities
        if any(security_term in query_lower for security_term in ['security', 'authentication', 'encryption']):
            entities['domain_type'] = 'security_features'
            entities['security_type'] = 'network_security'
        
        # Add entity confidence and source
        entities['entity_confidence'] = 'high' if len(entities) > 2 else 'medium'
        entities['extraction_source'] = 'kg_based_pattern_matching'
        
        return entities
    
    def _generate_entity_rich_general_queries(self, kg_entities: List[Dict]) -> List[Dict]:
        """Generate general queries enriched with real KG entity information"""
        entity_rich_queries = []
        
        # Sample entities for different types of queries
        sampled_entities = kg_entities[:50]  # Use first 50 for variety
        
        for entity in sampled_entities:
            table_name = entity['table_name']
            column_name = entity['column_name']
            category = entity['category']
            data_type = entity['data_type']
            intent = self._map_category_to_intent(category)
            
            # Generate specific queries with entity information
            queries_with_entities = [
                (f"Show me data from {table_name} table", {
                    'table_name': table_name,
                    'table_type': 'database_table',
                    'semantic_category': category,
                    'query_type': 'data_retrieval',
                    'domain_type': category,
                    'entity_confidence': 'high',
                    'extraction_source': 'kg_direct_reference'
                }),
                
                (f"What is the {column_name} value in {table_name}?", {
                    'table_name': table_name,
                    'column_name': column_name,
                    'column_type': data_type,
                    'semantic_category': category,
                    'query_type': 'specific_value_lookup',
                    'domain_type': category,
                    'entity_confidence': 'high',
                    'extraction_source': 'kg_direct_reference'
                }),
                
                (f"Analyze {column_name} patterns", {
                    'column_name': column_name,
                    'column_type': data_type,
                    'semantic_category': category,
                    'query_type': 'pattern_analysis',
                    'analysis_type': 'trend_analysis',
                    'domain_type': category,
                    'entity_confidence': 'high',
                    'extraction_source': 'kg_direct_reference'
                }),
                
                (f"Find tables similar to {table_name}", {
                    'table_name': table_name,
                    'table_type': 'database_table',
                    'semantic_category': category,
                    'query_type': 'similarity_search',
                    'search_type': 'table_similarity',
                    'domain_type': category,
                    'entity_confidence': 'medium',
                    'extraction_source': 'kg_inference'
                }),
                
                (f"Compare {column_name} across different tables", {
                    'column_name': column_name,
                    'column_type': data_type,
                    'semantic_category': category,
                    'query_type': 'cross_table_comparison',
                    'comparison_type': 'inter_table_analysis',
                    'domain_type': category,
                    'entity_confidence': 'medium',
                    'extraction_source': 'kg_inference'
                })
            ]
            
            for query_text, entities_dict in queries_with_entities:
                entity_rich_queries.append({
                    'text': query_text,
                    'intent': intent,
                    'label': list(self.ran_intents.keys()).index(intent),
                    'entities': entities_dict,
                    'metadata': {
                        'category': category,
                        'data_type': data_type,
                        'row_count': entity.get('row_count', 0),
                        'source': 'entity_rich_generation'
                    }
                })
        
        return entity_rich_queries
    
    def _generate_enhanced_synthetic_training_data(self) -> List[Dict]:
        """Generate enhanced synthetic training data with realistic entities"""
        synthetic_data = []
        
        # Realistic RAN entity examples based on common patterns
        synthetic_entities = {
            'tables': [
                'cell_performance_data', 'power_consumption_metrics', 'frequency_allocation_table',
                'signal_quality_measurements', 'network_topology_info', 'traffic_analysis_data',
                'handover_statistics', 'interference_detection_logs', 'capacity_planning_metrics',
                'configuration_parameters', 'timing_synchronization_data', 'security_audit_logs'
            ],
            'columns': {
                'power_optimization': ['power_consumption_dbm', 'energy_efficiency_ratio', 'battery_level_percent', 'power_amplifier_gain'],
                'spectrum_management': ['carrier_frequency_mhz', 'bandwidth_allocation_mhz', 'spectrum_efficiency_ratio', 'frequency_band_id'],
                'quality_assessment': ['rsrp_dbm', 'rsrq_db', 'sinr_db', 'signal_strength_indicator', 'coverage_area_km2'],
                'cell_configuration': ['cell_id', 'base_station_id', 'antenna_tilt_degrees', 'sector_azimuth_degrees'],
                'traffic_analysis': ['throughput_mbps', 'packet_loss_ratio', 'data_volume_gb', 'concurrent_users_count'],
                'performance_analysis': ['kpi_availability_percent', 'latency_ms', 'success_rate_percent', 'response_time_ms'],
                'fault_detection': ['error_count', 'fault_severity_level', 'alarm_status', 'system_health_score'],
                'capacity_planning': ['resource_utilization_percent', 'capacity_threshold_value', 'growth_projection_percent'],
                'interference_analysis': ['interference_level_db', 'noise_floor_dbm', 'interference_source_id'],
                'handover_optimization': ['handover_success_rate', 'handover_delay_ms', 'mobility_tracking_area']
            }
        }
        
        # Generate queries for each intent with realistic entities
        for intent, description in self.ran_intents.items():
            # Get relevant columns for this intent
            relevant_columns = synthetic_entities['columns'].get(intent, ['metric_value', 'data_field', 'measurement'])
            
            for i in range(10):  # Generate 10 samples per intent
                table = synthetic_entities['tables'][i % len(synthetic_entities['tables'])]
                column = relevant_columns[i % len(relevant_columns)]
                
                # Create realistic queries
                query_templates = [
                    f"Show {column} from {table}",
                    f"What is the average {column} in {table}?",
                    f"Analyze {column} trends from {table}",
                    f"Find anomalies in {column} data",
                    f"Compare {column} values across time periods"
                ]
                
                query = query_templates[i % len(query_templates)]
                
                # Generate realistic entities
                entities = {
                    'table_name': table,
                    'column_name': column,
                    'table_type': 'database_table',
                    'column_type': self._infer_column_type(column),
                    'semantic_category': intent.replace('_', '_'),
                    'domain_type': intent,
                    'query_type': self._infer_query_type(query),
                    'entity_confidence': 'medium',
                    'extraction_source': 'synthetic_generation'
                }
                
                # Add intent-specific entities
                if intent == 'power_optimization':
                    entities.update({
                        'unit_type': 'power_unit',
                        'measurement_context': 'energy_efficiency'
                    })
                elif intent == 'spectrum_management':
                    entities.update({
                        'unit_type': 'frequency_unit',
                        'spectrum_context': 'bandwidth_allocation'
                    })
                elif intent == 'quality_assessment':
                    entities.update({
                        'measurement_type': 'radio_quality',
                        'quality_context': 'signal_assessment'
                    })
                
                synthetic_data.append({
                    'text': query,
                    'intent': intent,
                    'label': list(self.ran_intents.keys()).index(intent),
                    'entities': entities,
                    'metadata': {
                        'category': intent,
                        'source': 'enhanced_synthetic'
                    }
                })
        
        return synthetic_data
    
    def _infer_column_type(self, column_name: str) -> str:
        """Infer data type from column name patterns"""
        column_lower = column_name.lower()
        
        if any(keyword in column_lower for keyword in ['id', 'identifier']):
            return 'varchar'
        elif any(keyword in column_lower for keyword in ['count', 'number', 'level']):
            return 'integer'
        elif any(keyword in column_lower for keyword in ['percent', 'ratio', 'rate', 'efficiency']):
            return 'decimal'
        elif any(keyword in column_lower for keyword in ['dbm', 'db', 'mhz', 'ghz', 'ms', 'mbps']):
            return 'float'
        elif any(keyword in column_lower for keyword in ['time', 'date', 'timestamp']):
            return 'timestamp'
        elif any(keyword in column_lower for keyword in ['status', 'name', 'type']):
            return 'varchar'
        else:
            return 'varchar'
    
    def _infer_query_type(self, query: str) -> str:
        """Infer query type from query text"""
        query_lower = query.lower()
        
        if query_lower.startswith('show') or query_lower.startswith('display'):
            return 'data_retrieval'
        elif 'what is' in query_lower or 'what are' in query_lower:
            return 'information_lookup'
        elif 'analyze' in query_lower or 'analysis' in query_lower:
            return 'analytical_query'
        elif 'compare' in query_lower or 'comparison' in query_lower:
            return 'comparative_analysis'
        elif 'find' in query_lower or 'search' in query_lower:
            return 'search_query'
        elif 'trend' in query_lower or 'pattern' in query_lower:
            return 'trend_analysis'
        else:
            return 'general_query'
    
    def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data as fallback - now enhanced with entities"""
        print("🔄 Generating enhanced synthetic training data as fallback...")
        return self._generate_enhanced_synthetic_training_data()
    
    def prepare_training_dataset(self, training_data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        texts = [item['text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
    
    def train_ran_model(self, output_dir: str = "./ran_domain_model"):
        """Train the RAN domain-specific model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for training")
        
        # Check compatibility
        compatible, message = check_transformers_compatibility()
        print(f"Compatibility check: {message}")
        if not compatible:
            raise ImportError(message)
            
        try:
            print("Generating training data...")
            training_data = self.generate_training_data()
            print(f"Generated {len(training_data)} training samples")
            
            if len(training_data) < 10:
                raise ValueError(f"Not enough training data generated: {len(training_data)} samples")
            
            dataset = self.prepare_training_dataset(training_data)
            
            # Initialize model and tokenizer
            print("Loading base model...")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Add padding token if it doesn't exist
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=len(self.ran_intents)
            )
            
            # Resize token embeddings if necessary
            model.resize_token_embeddings(len(tokenizer))
            
            # Tokenize data
            def tokenize_function(examples):
                return tokenizer(
                    examples['text'], 
                    truncation=True, 
                    padding=True, 
                    max_length=512,
                    return_tensors=None
                )
            
            print("Tokenizing dataset...")
            tokenized_dataset = dataset.map(tokenize_function, batched=True)
            
            # Create output directory if it doesn't exist
            import os
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs('./logs', exist_ok=True)
            
            # Training arguments with memory optimization
            training_args_dict = {
                'output_dir': output_dir,
                'num_train_epochs': 2,  # Reduced epochs for faster training
                'per_device_train_batch_size': 4,  # Smaller batch size for memory
                'per_device_eval_batch_size': 8,   # Smaller eval batch
                'warmup_steps': 100,    # Reduced warmup steps
                'weight_decay': 0.01,
                'logging_dir': './logs',
                'logging_steps': 50,    # More frequent logging for monitoring
                'save_steps': 500,      # More frequent saves to prevent loss
                'save_total_limit': 1,  # Keep only 1 checkpoint to save disk space
                'dataloader_drop_last': True,  # Drop last incomplete batch
                'push_to_hub': False,
                'report_to': None,  # Disable wandb/tensorboard logging
                'remove_unused_columns': True,  # Remove unused columns to save memory
                'load_best_model_at_end': False,
                'dataloader_num_workers': 0,  # Disable multiprocessing to save memory
                'gradient_accumulation_steps': 2,  # Accumulate gradients to simulate larger batch
                'fp16': torch.cuda.is_available(),  # Use mixed precision if GPU available
                'gradient_checkpointing': True,  # Trade compute for memory
            }
            
            # Add version-specific parameters
            try:
                # Try modern parameter names first
                training_args = TrainingArguments(**training_args_dict)
            except TypeError as e:
                print(f"Adjusting training arguments for compatibility: {e}")
                # Remove problematic parameters for older versions
                training_args_dict.pop('report_to', None)
                training_args_dict.pop('remove_unused_columns', None)
                training_args_dict.pop('load_best_model_at_end', None)
                training_args = TrainingArguments(**training_args_dict)
            
            # Train model
            print("Starting training...")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
            )
            
            # Start training with error handling and memory optimization
            try:
                # Clear GPU cache before training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"🖥️ GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB")
                
                # Monitor memory during training
                print("🚀 Starting training with memory optimization...")
                trainer.train()
                
                # Clear cache after training
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
                    print("💥 GPU out of memory. Trying CPU training with minimal settings...")
                    
                    # Clear GPU memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        del model, trainer
                    
                    # Force CPU training with minimal settings
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name, 
                        num_labels=len(self.ran_intents)
                    )
                    model.resize_token_embeddings(len(tokenizer))
                    
                    # Ultra-minimal CPU training args
                    cpu_training_args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=1,  # Single epoch for CPU
                        per_device_train_batch_size=2,  # Minimal batch size
                        warmup_steps=10,
                        weight_decay=0.01,
                        logging_steps=10,
                        save_steps=100,
                        save_total_limit=1,
                        report_to=None,
                        use_cpu=True,
                        dataloader_num_workers=0,
                        gradient_accumulation_steps=4,  # Larger accumulation for CPU
                    )
                    
                    trainer = Trainer(
                        model=model,
                        args=cpu_training_args,
                        train_dataset=tokenized_dataset,
                        tokenizer=tokenizer,
                    )
                    
                    print("🔄 Starting CPU training with minimal configuration...")
                    trainer.train()
                else:
                    raise e
            
            # Save model
            print(f"Saving model to {output_dir}")
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            
            # Save intent labels
            with open(f"{output_dir}/intent_labels.json", 'w') as f:
                json.dump(list(self.ran_intents.keys()), f)
            
            # Save training info
            training_info = {
                'model_name': self.model_name,
                'num_labels': len(self.ran_intents),
                'training_samples': len(training_data),
                'intents': list(self.ran_intents.keys()),
                'training_date': datetime.now().isoformat(),
                'transformers_version': TRANSFORMERS_VERSION
            }
            
            with open(f"{output_dir}/training_info.json", 'w') as f:
                json.dump(training_info, f, indent=2)
            
            print("Training completed successfully!")
            return model, tokenizer
            
        except Exception as e:
            print(f"Training failed: {e}")
            print("Trying fallback training with reduced parameters...")
            
            # Fallback training with minimal settings
            try:
                return self._fallback_training(output_dir)
            except Exception as fallback_error:
                print(f"Fallback training also failed: {fallback_error}")
                raise e
    
    def _fallback_training(self, output_dir: str):
        """Fallback training with minimal settings"""
        print("Starting fallback training with synthetic data...")
        
        # Generate minimal synthetic data
        synthetic_data = self._generate_synthetic_training_data()
        dataset = self.prepare_training_dataset(synthetic_data)
        
        # Use a smaller model for fallback
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=len(self.ran_intents)
        )
        
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=256)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=1,  # Reduced epochs
            per_device_train_batch_size=4,  # Small batch size
            warmup_steps=50,  # Reduced warmup
            weight_decay=0.01,
            logging_steps=50,
            save_steps=500,
            save_total_limit=1,
            report_to=None,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save intent labels
        with open(f"{output_dir}/intent_labels.json", 'w') as f:
            json.dump(list(self.ran_intents.keys()), f)
        
        print("Fallback training completed!")
        return model, tokenizer
    
    def evaluate_model(self, model_path: str = "./ran_domain_model"):
        """Evaluate the trained model with test queries"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for evaluation")
            
        try:
            # Load the trained model
            classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                return_all_scores=False
            )
            
            # Load intent labels
            intent_labels_path = f"{model_path}/intent_labels.json"
            if os.path.exists(intent_labels_path):
                with open(intent_labels_path, 'r') as f:
                    intent_labels = json.load(f)
            else:
                intent_labels = list(self.ran_intents.keys())
                print("Warning: intent_labels.json not found, using default labels")
            
            # Test queries covering all semantic categories
            test_queries = [
                # Network topology
                "Show network topology structure",
                "Display network connections",
                
                # Quality
                "Show signal quality measurements", 
                "Analyze quality metrics",
                
                # Frequency
                "What are the frequency allocations?",
                "Show spectrum usage",
                
                # Configuration
                "Show configuration parameters",
                "Display parameter settings",
                
                # Traffic
                "Show network traffic patterns",
                "Analyze traffic data",
                
                # Performance
                "Analyze cell performance metrics",
                "Show KPI dashboard",
                
                # Power
                "Show me power consumption data",
                "Analyze power efficiency",
                
                # Security
                "Show security features",
                "Check security status",
                
                # Mobility
                "Optimize handover parameters",
                "Show mobility patterns",
                
                # General
                "Show general system information",
                "Get overview data"
            ]
            
            print("\n=== Model Evaluation ===")
            results = []
            
            for query in test_queries:
                try:
                    result = classifier(query)
                    
                    # Handle different label formats
                    label = result[0]['label']
                    if 'LABEL_' in label:
                        # Format: LABEL_0, LABEL_1, etc.
                        label_idx = int(label.split('_')[-1])
                    else:
                        # Try to find the label index directly
                        try:
                            label_idx = int(label)
                        except ValueError:
                            # If label is already the intent name
                            predicted_intent = label
                            confidence = result[0]['score']
                            label_idx = None
                    
                    if label_idx is not None and label_idx < len(intent_labels):
                        predicted_intent = intent_labels[label_idx]
                    elif label_idx is not None:
                        predicted_intent = f"Unknown_Label_{label_idx}"
                    
                    confidence = result[0]['score']
                    
                    print(f"Query: '{query}'")
                    print(f"Predicted: {predicted_intent} (confidence: {confidence:.3f})")
                    
                    # Add confidence assessment
                    if confidence > 0.8:
                        print("   ✅ High confidence")
                    elif confidence > 0.6:
                        print("   ⚠️ Medium confidence")
                    else:
                        print("   ❓ Low confidence")
                    
                    print("-" * 50)
                    
                    results.append({
                        'query': query,
                        'predicted_intent': predicted_intent,
                        'confidence': confidence
                    })
                    
                except Exception as query_error:
                    print(f"Error processing query '{query}': {query_error}")
                    continue
            
            # Print summary statistics
            if results:
                confidences = [r['confidence'] for r in results]
                avg_confidence = sum(confidences) / len(confidences)
                print(f"\n📊 Evaluation Summary:")
                print(f"   Total queries tested: {len(results)}")
                print(f"   Average confidence: {avg_confidence:.3f}")
                print(f"   High confidence queries (>0.8): {sum(1 for c in confidences if c > 0.8)}")
                print(f"   Medium confidence queries (0.6-0.8): {sum(1 for c in confidences if 0.6 <= c <= 0.8)}")
                print(f"   Low confidence queries (<0.6): {sum(1 for c in confidences if c < 0.6)}")
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            print("This might indicate the model wasn't trained properly or files are missing.")
            print(f"Error during evaluation: {e}")
    
    def export_training_data(self, filepath: str = "./ran_training_data.json"):
        """Export training data for manual review or external training"""
        training_data = self.generate_training_data()
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Training data exported to {filepath}")
        print(f"Total samples: {len(training_data)}")
        
        # Print statistics
        intent_counts = {}
        for item in training_data:
            intent = item['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("\nIntent distribution:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  {intent}: {count} samples")

class RANEntityRecognitionTrainer:
    """Train NER model for RAN-specific entities"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.entity_types = {
            'TABLE_NAME': 'B-TAB',
            'COLUMN_NAME': 'B-COL',
            'CELL_ID': 'B-CELL',
            'FREQUENCY': 'B-FREQ',
            'POWER_VALUE': 'B-PWR',
            'METRIC_NAME': 'B-MET',
            'TIME_VALUE': 'B-TIME'
        }
    
    def generate_ner_training_data(self) -> List[Tuple[str, List[Tuple]]]:
        """Generate NER training data in spaCy format"""
        training_data = []
        
        try:
            with self.integrator.driver.session() as session:
                # Get entity examples from the graph
                result = session.run("""
                    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, collect(c.name)[0..10] as column_names
                    LIMIT 50
                """)
                
                for record in result:
                    table_name = record['table_name']
                    column_names = record['column_names'] or []
                    
                    # Generate sentences with entities
                    sentences = [
                        f"Show data from {table_name} table",
                        f"Get {column_names[0]} from {table_name}" if column_names else f"Query {table_name}",
                        f"Find tables related to {table_name}",
                        f"What columns are in {table_name}?"
                    ]
                    
                    for sentence in sentences:
                        entities = []
                        
                        # Find table name position
                        table_start = sentence.find(table_name)
                        if table_start != -1:
                            entities.append((table_start, table_start + len(table_name), 'TABLE_NAME'))
                        
                        # Find column name position
                        for col in column_names[:2]:  # Limit to first 2 columns
                            col_start = sentence.find(col)
                            if col_start != -1:
                                entities.append((col_start, col_start + len(col), 'COLUMN_NAME'))
                        
                        if entities:
                            training_data.append((sentence, entities))
                            
        except Exception as e:
            logging.error(f"Error generating NER training data: {e}")
        
        return training_data
    
    def export_ner_training_data(self, filepath: str = "./ran_ner_training.json"):
        """Export NER training data"""
        training_data = self.generate_ner_training_data()
        
        # Convert to a more readable format
        export_data = []
        for text, entities in training_data:
            export_data.append({
                'text': text,
                'entities': [{'start': start, 'end': end, 'label': label} for start, end, label in entities]
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"NER training data exported to {filepath}")
        print(f"Total samples: {len(export_data)}")

# Example usage function
def train_ran_models(neo4j_integrator):
    """Train both intent classification and NER models"""
    print("Starting RAN domain model training...")
    
    # Check compatibility first
    compatible, message = check_transformers_compatibility()
    print(f"Compatibility check: {message}")
    
    if not compatible:
        print("Please upgrade transformers: pip install --upgrade transformers")
        return False
    
    # Train intent classification model
    intent_trainer = RANDomainModelTrainer(neo4j_integrator)
    
    try:
        model, tokenizer = intent_trainer.train_ran_model()
        print("Intent classification model trained successfully!")
        
        # Evaluate the model
        intent_trainer.evaluate_model()
        
        return True
        
    except Exception as e:
        print(f"Error training intent model: {e}")
        print("Exporting training data for manual review...")
        try:
            intent_trainer.export_training_data()
            print("Training data exported successfully")
        except Exception as export_error:
            print(f"Error exporting training data: {export_error}")
        return False

def test_training_setup(neo4j_integrator=None):
    """Test the training setup without full training"""
    print("Testing RAN fine-tuning setup...")
    
    # Test imports
    if not TRANSFORMERS_AVAILABLE:
        print("❌ Transformers not available")
        return False
    
    # Test compatibility
    compatible, message = check_transformers_compatibility()
    print(f"Transformers: {message}")
    
    if not compatible:
        print("❌ Compatibility issues detected")
        return False
    
    # Test trainer initialization
    if neo4j_integrator:
        try:
            trainer = RANDomainModelTrainer(neo4j_integrator)
            print("✅ Trainer initialized successfully")
            
            # Test synthetic data generation
            synthetic_data = trainer._generate_synthetic_training_data()
            print(f"✅ Generated {len(synthetic_data)} synthetic training samples")
            
            return True
            
        except Exception as e:
            print(f"❌ Trainer initialization failed: {e}")
            return False
    else:
        print("⚠️ No Neo4j integrator provided - limited testing")
        return True

if __name__ == "__main__":
    # Example usage - you would need to provide your neo4j_integrator
    print("RAN Fine-tuning Module")
    print("This module requires a Neo4j integrator instance to run.")
    print("Example usage:")
    print("  from ran_finetuning import train_ran_models")
    print("  train_ran_models(your_neo4j_integrator)")
