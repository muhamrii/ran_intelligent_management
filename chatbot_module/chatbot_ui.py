"""
Streamlit-based Chatbot UI for RAN Intelligent Management
"""

import streamlit as st
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import RANChatbot
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="RAN Intelligent Management Chatbot",
    page_icon="📡",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'neo4j_connected' not in st.session_state:
        st.session_state.neo4j_connected = False

def connect_to_neo4j(uri, user, password):
    """Connect to Neo4j and initialize chatbot"""
    try:
        integrator = RANNeo4jIntegrator(uri, user, password)
        chatbot = RANChatbot(integrator)
        st.session_state.chatbot = chatbot
        st.session_state.neo4j_connected = True
        return True, "Successfully connected to Neo4j!"
    except Exception as e:
        return False, f"Failed to connect to Neo4j: {str(e)}"

def display_chat_history():
    """Display chat history"""
    for i, (query, response) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**🧑 User:** {query}")
            st.markdown(f"**🤖 Assistant:** {response}")
            st.markdown("---")

def main():
    initialize_session_state()
    
    st.title("📡 RAN Intelligent Management Chatbot")
    st.markdown("Welcome to the RAN Configuration Knowledge Graph Chatbot!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("Configuration")
        
        # Neo4j Connection
        st.subheader("Neo4j Connection")
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Username", value="neo4j")
        neo4j_password = st.text_input("Password", type="password", value="password")
        
        if st.button("Connect to Neo4j"):
            with st.spinner("Connecting to Neo4j..."):
                success, message = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
                if success:
                    st.success(message)
                else:
                    st.error(message)
        
        # Connection status
        if st.session_state.neo4j_connected:
            st.success("✅ Connected to Neo4j")
        else:
            st.warning("⚠️ Not connected to Neo4j")
        
        # Quick actions
        st.subheader("Quick Actions")
        if st.button("Schema Overview"):
            if st.session_state.chatbot:
                result = st.session_state.chatbot.process_query("show schema overview")
                response = st.session_state.chatbot.generate_response(result)
                st.session_state.chat_history.append(("show schema overview", response))
                st.rerun()
        
        if st.button("List All Tables"):
            if st.session_state.chatbot:
                result = st.session_state.chatbot.process_query("show all tables")
                response = st.session_state.chatbot.generate_response(result)
                st.session_state.chat_history.append(("show all tables", response))
                st.rerun()
        
        # Sample data loader
        st.subheader("Sample Data")
        if st.button("Load Sample Data"):
            if st.session_state.chatbot:
                with st.spinner("Loading sample data..."):
                    try:
                        # Create sample data
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
                            }),
                            'performance_metrics': pd.DataFrame({
                                'cell_id': np.random.randint(0, 100, 200),
                                'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
                                'throughput': np.random.uniform(10, 100, 200),
                                'latency': np.random.uniform(1, 50, 200)
                            })
                        }
                        
                        # Load data into knowledge graph
                        st.session_state.chatbot.integrator.create_nodes_and_relationships(sample_dataframes)
                        st.success("Sample data loaded successfully!")
                        
                    except Exception as e:
                        st.error(f"Failed to load sample data: {str(e)}")
    
    # Main chat interface
    if not st.session_state.neo4j_connected:
        st.warning("Please connect to Neo4j first using the sidebar.")
        return
    
    # Chat input
    st.subheader("Chat with the RAN Knowledge Graph")
    
    # Example queries
    with st.expander("💡 Example Queries"):
        st.markdown("""
        Try these example queries:
        - "Show me all tables"
        - "Find tables related to cell_config"
        - "What columns are in cell_config?"
        - "Show schema overview"
        - "Get details about neighbor_list table"
        - "Find columns similar to frequency"
        - "Show performance metrics"
        """)
    
    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_query = st.text_input("Enter your question:", placeholder="e.g., Show me tables related to cell configuration")
        submitted = st.form_submit_button("Send", use_container_width=True)
        
        if submitted and user_query:
            if st.session_state.chatbot:
                with st.spinner("Processing your query..."):
                    try:
                        # Process the query
                        result = st.session_state.chatbot.process_query(user_query)
                        response = st.session_state.chatbot.generate_response(result)
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_query, response))
                        
                    except Exception as e:
                        error_response = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append((user_query, error_response))
                
                st.rerun()
            else:
                st.error("Chatbot not initialized. Please check your Neo4j connection.")
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        display_chat_history()
        
        # Clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Additional features
    st.subheader("Additional Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Generate NER Training Data"):
            if st.session_state.chatbot:
                with st.spinner("Generating NER training data..."):
                    try:
                        training_data = st.session_state.chatbot.ner_generator.generate_ner_training_data()
                        st.success(f"Generated {len(training_data)} NER training examples!")
                        
                        # Show sample training data
                        if training_data:
                            st.subheader("Sample Training Data")
                            for i, (text, entities) in enumerate(training_data[:5]):
                                st.write(f"**Example {i+1}:**")
                                st.write(f"Text: {text}")
                                st.write(f"Entities: {entities}")
                                st.write("---")
                    except Exception as e:
                        st.error(f"Failed to generate NER data: {str(e)}")
    
    with col2:
        if st.button("Export Schema"):
            if st.session_state.chatbot:
                with st.spinner("Exporting schema..."):
                    try:
                        schema = st.session_state.chatbot.query_interface.get_schema_overview()
                        st.json(schema)
                        st.success("Schema exported!")
                    except Exception as e:
                        st.error(f"Failed to export schema: {str(e)}")

if __name__ == "__main__":
    main()
