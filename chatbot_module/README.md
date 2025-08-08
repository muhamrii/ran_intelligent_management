# Enhanced RAN Chatbot Module

This module provides an advanced chatbot system specifically designed for querying and analyzing RAN (Radio Access Network) knowledge graphs with over 273 tables, 5,584 columns, and 4M+ relationships.

## Features

### 🚀 Performance Optimizations
- **Query Caching**: Intelligent caching system with TTL for repeated queries
- **Optimized Graph Traversal**: Multi-hop relationship analysis with path scoring
- **Efficient Search**: Enhanced semantic search with relevance scoring
- **Memory Management**: Optimized for large-scale knowledge graphs

### 🧠 Enhanced NLP Capabilities
- **RAN Entity Extraction**: Specialized extraction for technical terms, measurements, identifiers
- **Intent Classification**: Domain-specific intent detection for RAN queries
- **Contextualized Search**: Query generation based on extracted entities
- **Technical Pattern Recognition**: Recognition of RAN-specific patterns (cell IDs, frequencies, power values)

### 🎯 Domain-Specific Features
- **Multi-hop Relationship Discovery**: Find complex relationships across multiple nodes
- **Semantic Clustering**: Identify conceptual groups in the knowledge graph
- **Performance Analysis**: Specialized queries for network performance metrics
- **Technical Troubleshooting**: Support for fault detection and diagnostics

### 🔧 Fine-tuning Capabilities
- **Custom Model Training**: Train domain-specific models on your KG data
- **Intent Classification**: Specialized classification for RAN use cases
- **Entity Recognition**: NER training for RAN-specific entities
- **Continuous Learning**: Model adaptation based on query patterns

## File Structure

```
chatbot_module/
├── chatbot.py              # Enhanced chatbot with optimizations
├── ran_finetuning.py       # Fine-tuning module for domain models
├── chatbot_example.py      # Example usage and interactive demo
├── chatbot_ui.py          # UI components (if applicable)
└── README.md              # This file
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- `transformers` - For fine-tuning language models
- `datasets` - For training data management
- `torch` - Deep learning framework
- `neo4j` - Graph database connectivity
- `sentence-transformers` - Semantic similarity
- `scikit-learn` - ML utilities

### 2. Basic Usage

```python
from chatbot_module.chatbot import EnhancedRANChatbot
from knowledge_graph_module.kg_builder import Neo4jIntegrator

# Initialize Neo4j connection
neo4j_integrator = Neo4jIntegrator("bolt://localhost:7687", "neo4j", "password")

# Create enhanced chatbot
chatbot = EnhancedRANChatbot(neo4j_integrator)

# Process queries
result = chatbot.enhanced_process_query("Show me power consumption data")
print(result['response'])
```

### 3. Fine-tuning Domain Model

```python
from chatbot_module.ran_finetuning import train_ran_models

# Train RAN-specific models
train_ran_models(neo4j_integrator)

# Use fine-tuned model
chatbot = EnhancedRANChatbot(neo4j_integrator, use_domain_model=True)
```

## Enhanced Query Types

### 1. Technical Entity Queries
Automatically extract and search for RAN-specific entities:

```python
# Query: "Show cell_id 12345 with frequency 2.4 GHz"
# Extracts: identifiers=['cell_id'], measurements=['2.4 ghz']
result = chatbot.enhanced_process_query("Show cell_id 12345 with frequency 2.4 GHz")
```

### 2. Multi-hop Relationship Analysis
Find complex relationships across multiple nodes:

```python
# Query: "What tables are connected to power_metrics?"
result = chatbot.enhanced_process_query("Find tables related to power_metrics")
# Returns: relationship strength, connection paths, related tables
```

### 3. Semantic Clustering
Discover conceptual groups in your knowledge graph:

```python
# Query: "Show me conceptual groups related to performance"
result = chatbot.enhanced_process_query("Show conceptual groups for performance")
# Returns: semantic clusters, confidence scores, related tables
```

### 4. Performance-Optimized Search
Fast search with caching and relevance scoring:

```python
# Cached and optimized search
result = chatbot.enhanced_process_query("Find throughput metrics")
# Returns: relevance-scored results with relationship context
```

## Intent Classification

The system recognizes 10 RAN-specific intents:

1. **Performance Analysis** - KPIs, metrics, benchmarks
2. **Power Optimization** - Energy consumption, efficiency
3. **Spectrum Management** - Frequency allocation, bandwidth
4. **Cell Configuration** - Cell parameters, settings
5. **Quality Assessment** - Signal quality, coverage
6. **Traffic Analysis** - Network load, usage patterns
7. **Fault Detection** - Error diagnosis, anomalies
8. **Capacity Planning** - Resource allocation, scaling
9. **Interference Analysis** - Signal interference, mitigation
10. **Handover Optimization** - Mobility procedures, optimization

## Entity Recognition

Specialized recognition for RAN entities:

- **Cell Identifiers**: cell_id, enb_id, gnb_id
- **Frequency Values**: MHz, GHz, frequency bands
- **Power Measurements**: dBm, watts, power levels
- **Quality Metrics**: RSRP, RSRQ, SINR
- **Time References**: timestamps, time periods
- **Spatial Data**: coordinates, locations

## API Reference

### EnhancedRANChatbot

```python
class EnhancedRANChatbot(RANChatbot):
    def __init__(self, neo4j_integrator, use_domain_model=False):
        """
        Initialize enhanced chatbot
        
        Args:
            neo4j_integrator: Neo4j database connection
            use_domain_model: Whether to use fine-tuned domain model
        """
    
    def enhanced_process_query(self, user_query: str) -> Dict:
        """
        Process query with enhanced capabilities
        
        Args:
            user_query: Natural language query
            
        Returns:
            Dict with type, intent, entities, results, and response
        """
```

### Key Methods

- `enhanced_process_query()`: Main query processing with optimizations
- `_predict_intent_with_model()`: Domain-specific intent prediction
- `_extract_table_name()`: Extract table names from queries
- `_format_*_response()`: Specialized response formatting

## Fine-tuning Module

### RANDomainModelTrainer

```python
class RANDomainModelTrainer:
    def train_ran_model(self, output_dir="./ran_domain_model"):
        """Train domain-specific intent classification model"""
    
    def generate_training_data(self) -> List[Dict]:
        """Generate training data from knowledge graph"""
    
    def evaluate_model(self, model_path="./ran_domain_model"):
        """Evaluate trained model performance"""
```

### Training Process

1. **Data Generation**: Extract real column/table names from KG
2. **Template Application**: Generate realistic queries using templates
3. **Intent Mapping**: Map semantic categories to RAN intents
4. **Model Training**: Fine-tune DistilBERT for intent classification
5. **Evaluation**: Test model on RAN-specific queries

## Performance Considerations

### Large Knowledge Graph Optimization

For your KG with 273 tables and 4M+ relationships:

1. **Caching Strategy**: 1-hour TTL for query results
2. **Pagination**: Limit results to prevent memory issues
3. **Relevance Scoring**: Prioritize most relevant results
4. **Index Usage**: Leverage Neo4j indices for faster queries
5. **Connection Pooling**: Efficient database connections

### Memory Management

- Query result caching with automatic cleanup
- Efficient relationship traversal algorithms
- Optimized data structures for large result sets
- Background processing for complex queries

## Example Interactions

### Performance Analysis
```
User: "Show me KPI metrics for cell performance"
System: 🔍 Technical Search Results (Found 15 matches)
        Detected entities: measurements=['kpi']
        📋 cell_performance_kpis
        • Matching columns: throughput_mbps, latency_ms, success_rate
        • Conceptual relationships: 45
```

### Power Optimization
```
User: "Find power consumption patterns"
System: ⚡ Optimized Search Results (Found 8 matches)
        📋 power_consumption_hourly (Relevance: 2.0)
        • Top columns: power_dbm, energy_kwh, efficiency_ratio
        • Relationships: 23
        • Related to: cell_config, power_settings
```

### Relationship Discovery
```
User: "What tables are connected to frequency_allocation?"
System: 🔗 Multi-hop Relationships for frequency_allocation
        📋 spectrum_usage
        • Relationship strength: 0.85
        • Connection count: 12
        • Shortest path: 2 hops
```

## Troubleshooting

### Common Issues

1. **Model Not Found**: Fine-tuned model not available
   - Solution: Train model first or use `use_domain_model=False`

2. **Slow Queries**: Large result sets
   - Solution: Add more specific filters or increase cache TTL

3. **Memory Issues**: Large knowledge graph
   - Solution: Increase pagination limits, optimize queries

4. **Connection Errors**: Neo4j connectivity
   - Solution: Check Neo4j status and connection parameters

### Performance Tuning

1. **Query Optimization**:
   ```python
   # Use more specific queries
   chatbot.enhanced_process_query("Show power data for cell_id 123")
   # Instead of
   chatbot.enhanced_process_query("Show all data")
   ```

2. **Caching Configuration**:
   ```python
   # Adjust cache TTL for your use case
   chatbot.optimized_query.cache_ttl = 7200  # 2 hours
   ```

3. **Result Limiting**:
   ```python
   # Limit results for faster responses
   results = chatbot.optimized_query.optimized_semantic_search(query, limit=5)
   ```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

## License

[Your License Here]

## Support

For questions or issues:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the example usage in `chatbot_example.py`
