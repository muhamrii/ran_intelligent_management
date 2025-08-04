# RAN Intelligent Management System

This project transforms Radio Access Network (RAN) configuration data into a structured knowledge base using Neo4j, enabling automated understanding, navigation, and retrieval for AI-powered chatbots or digital assistants.

## Project Overview

The system organizes RAN tables into a graph structure, discovers semantic relationships, and generates Named Entity Recognition (NER) training data for natural language processing.

## Architecture

### Modules

1. **Knowledge Graph Module** (`knowledge_graph_module/`)
   - `kg_builder.py`: Core functions for knowledge graph generation
   - `kg_builder.ipynb`: Step-by-step notebook for graph creation

2. **Chatbot Module** (`chatbot_module/`)
   - `chatbot.py`: Information retrieval and answer generation functions
   - `chatbot.ipynb`: Step-by-step notebook for chatbot testing
   - `chatbot_ui.py`: Streamlit-based UI for user interaction

3. **Parser Module** (`parser_module/`)
   - Extracts metadata from RAN tables (column names, data types, statistics)

## Key Features

### Knowledge Graph Components
- **Nodes**: Table, Column, Concept
- **Relationships**: 
  - `NAME_SIMILARITY`: Semantic similarity between column names
  - `VALUE_OVERLAP`: Shared values between columns
  - `PATTERN_MATCH`: Pattern-based relationships
  - `REFERENCES`: Foreign key relationships
  - `CONCEPTUAL_GROUP`: Clustering-based semantic groups

### Chatbot Capabilities
- Semantic search across the graph
- Related table discovery
- Natural language query processing
- NER training data generation
- Schema overview and analysis

## Installation

### Prerequisites
- Python 3.8+
- Neo4j Database 4.0+
- Required Python packages (see requirements.txt)

### Setup
1. Install dependencies:
   ```bash
   pip install pandas numpy neo4j sentence-transformers scikit-learn streamlit matplotlib seaborn
   ```

2. Start Neo4j database:
   ```bash
   # Default connection: bolt://localhost:7687
   # Default credentials: neo4j/password
   ```

3. Update connection parameters in the scripts as needed.

## Usage

### Quick Start
Run the complete demo:
```bash
python main_example.py
```

### Step-by-Step Execution

1. **Create Knowledge Graph**:
   - Open `knowledge_graph_module/kg_builder.ipynb`
   - Execute cells to build the graph from sample data

2. **Test Chatbot**:
   - Open `chatbot_module/chatbot.ipynb`
   - Execute cells to test NER and query functionality

3. **Launch UI**:
   ```bash
   cd chatbot_module
   streamlit run chatbot_ui.py
   ```

### Manual Usage

#### Knowledge Graph Creation
```python
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator

# Initialize
integrator = RANNeo4jIntegrator(
    neo4j_uri="bolt://localhost:7687",
    neo4j_user="neo4j", 
    neo4j_password="password"
)

# Create graph from DataFrames
integrator.create_nodes_and_relationships(dataframes_dict)
```

#### Chatbot Usage
```python
from chatbot_module.chatbot import RANChatbot

# Initialize chatbot
chatbot = RANChatbot(integrator)

# Process queries
result = chatbot.process_query("Show me tables related to cell configuration")
response = chatbot.generate_response(result)
print(response)
```

## Sample Data Structure

The system works with RAN configuration tables such as:

- **cell_config**: Cell ID, frequency, power level, status, technology
- **neighbor_relations**: Source/target cells, handover priority, distance
- **performance_counters**: Throughput, latency, packet loss, utilization
- **site_info**: Site details, location, installation dates
- **alarm_data**: Alarm types, severity, timestamps

## API Reference

### Knowledge Graph Builder
- `extract_table_metadata()`: Extract metadata from DataFrames
- `discover_semantic_relationships()`: Find relationships between columns
- `create_nodes_and_relationships()`: Build Neo4j graph
- `generate_conceptual_groups()`: Create semantic clusters

### Chatbot Interface
- `semantic_search()`: Search across the graph
- `find_related_tables()`: Discover table relationships
- `get_table_details()`: Retrieve table information
- `process_query()`: Handle natural language queries

## Configuration

### Neo4j Settings
Update connection parameters in each module:
```python
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
```

### Similarity Thresholds
Adjust in `kg_builder.py`:
```python
self.similarity_threshold = 0.7  # Name similarity threshold
```

## Examples

### Query Examples
- "Show me all tables"
- "Find tables related to cell_config"
- "What columns contain frequency information?"
- "Get schema overview"
- "Show performance metrics tables"

### Expected Relationships
- cell_config.cell_id ↔ neighbor_relations.source_cell_id (REFERENCES)
- cell_config.frequency ↔ performance_counters.frequency_band (NAME_SIMILARITY)
- Multiple status fields across tables (PATTERN_MATCH)

## Troubleshooting

### Common Issues
1. **Neo4j Connection Failed**: Ensure Neo4j is running and credentials are correct
2. **No Relationships Found**: Check data quality and similarity thresholds
3. **Import Errors**: Verify all required packages are installed

### Debugging
Enable logging in scripts:
```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For questions and support, please open an issue in the repository.