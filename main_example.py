"""
Main Example Script for RAN Intelligent Management
This script demonstrates the complete workflow from knowledge graph creation to chatbot interaction.
"""

import pandas as pd
import numpy as np
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import RANChatbot

def main():
    print("🚀 RAN Intelligent Management System Demo")
    print("=" * 50)
    
    # Step 1: Initialize Neo4j connection
    print("🔗 Connecting to Neo4j...")
    integrator = RANNeo4jIntegrator(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user="neo4j",
        neo4j_password="password"
    )
    
    # Step 2: Create sample data (as shown in original sample.py)
    print("📊 Creating sample RAN data...")
    np.random.seed(42)
    
    sample_dataframes = {
        'cell_config': pd.DataFrame({
            'cell_id': range(100),
            'frequency': np.random.choice([1800, 2100, 2600], 100),
            'power_level': np.random.uniform(20, 40, 100),
            'status': np.random.choice(['active', 'inactive'], 100)
        }),
        'neighbor_list': pd.DataFrame({
            'source_cell': range(50),
            'target_cell': np.random.randint(0, 100, 50),
            'handover_priority': np.random.randint(1, 10, 50)
        })
    }
    
    # Step 3: Create knowledge graph
    print("🏗️ Building knowledge graph...")
    integrator.create_nodes_and_relationships(sample_dataframes)
    
    # Step 4: Initialize chatbot
    print("🤖 Initializing chatbot...")
    chatbot = RANChatbot(integrator)
    
    # Step 5: Generate NER training data
    print("🧠 Generating NER training data...")
    training_data = chatbot.ner_generator.generate_ner_training_data()
    print(f"Generated {len(training_data)} NER training examples")
    
    # Step 6: Test query interface
    print("🔍 Testing query interface...")
    results = chatbot.query_interface.find_related_tables('cell_config')
    print("Related tables:", results)
    
    # Step 7: Test chatbot conversation
    print("💬 Testing chatbot conversation...")
    test_queries = [
        "Show me all tables",
        "Find tables related to cell_config",
        "What columns are in neighbor_list?"
    ]
    
    for query in test_queries:
        print(f"\n👤 User: {query}")
        result = chatbot.process_query(query)
        response = chatbot.generate_response(result)
        print(f"🤖 Assistant: {response}")
    
    print("\n✅ Demo completed successfully!")
    integrator.close()

if __name__ == "__main__":
    main()
