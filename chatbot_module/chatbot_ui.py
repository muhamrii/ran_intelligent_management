"""
Streamlit-based Chatbot UI for RAN Intelligent Management
Enhanced with RAN domain intelligence and advanced query capabilities
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
import matplotlib.pyplot as plt

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
    """Enhanced main Streamlit application"""
    initialize_session_state()
    
    # Header
    st.title("📡 RAN Intelligent Management Chatbot")
    st.markdown("Interact with your RAN knowledge graph using natural language and domain intelligence")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Neo4j connection
        st.subheader("🔗 Neo4j Connection")
        neo4j_uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
        neo4j_user = st.text_input("Username", value="neo4j")
        neo4j_password = st.text_input("Password", type="password", value="ranqarag#1")
        
        if st.button("Connect to Neo4j"):
            success, message = connect_to_neo4j(neo4j_uri, neo4j_user, neo4j_password)
            if success:
                st.success(message)
                st.rerun()
            else:
                st.error(message)
        
        # Connection status
        if st.session_state.neo4j_connected:
            st.success("✅ Connected to Neo4j")
        else:
            st.warning("⚠️ Not connected to Neo4j")
        
        # Enhanced Query Examples
        st.subheader("💡 Enhanced Query Examples")
        query_categories = {
            "Basic Queries": [
                "Show me all available tables",
                "What is the schema overview?",
                "Get details about cell_config"
            ],
            "RAN Domain Insights": [
                "Show me power domain insights",
                "Find frequency related patterns",
                "What timing synchronization data exists?",
                "Show me traffic analysis tables"
            ],
            "Advanced Searches": [
                "Search for performance concepts",
                "Find tables related to neighbor_relations",
                "Show configuration patterns",
                "Find mobility management data"
            ]
        }
        
        selected_category = st.selectbox("Query Category:", list(query_categories.keys()))
        selected_example = st.selectbox("Choose an example:", [""] + query_categories[selected_category])
        
        # RAN Domain Quick Access
        st.subheader("🎯 RAN Domain Analysis")
        ran_domains = ['performance', 'power', 'frequency', 'topology', 'quality', 'traffic', 'mobility', 'configuration', 'security', 'timing']
        selected_domain = st.selectbox("Analyze RAN Domain:", [""] + ran_domains)
        
        if selected_domain and st.button("Analyze Domain"):
            if st.session_state.chatbot:
                domain_query = f"Show me {selected_domain} domain insights"
                process_query(domain_query)
        
        # Pattern Analysis
        st.subheader("🔍 Pattern Analysis")
        pattern_types = ['id_field', 'timestamp', 'status_field', 'power_related', 'frequency_related', 'measurement', 'configuration']
        selected_pattern = st.selectbox("Analyze Pattern:", [""] + pattern_types)
        
        if selected_pattern and st.button("Find Patterns"):
            if st.session_state.chatbot:
                pattern_query = f"Find {selected_pattern} patterns"
                process_query(pattern_query)
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Main chat interface
    if st.session_state.neo4j_connected and st.session_state.chatbot:
        # Display enhanced knowledge graph stats
        try:
            schema = st.session_state.chatbot.query_interface.get_schema_overview()
            if schema:
                st.subheader("📊 Knowledge Graph Overview")
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Tables", schema.get('tables', {}).get('total_tables', 0))
                with col2:
                    st.metric("Columns", schema.get('columns', {}).get('total_columns', 0))
                with col3:
                    total_rels = sum(r['count'] for r in schema.get('relationships', []))
                    st.metric("Relationships", f"{total_rels:,}")
                with col4:
                    st.metric("Concepts", len(schema.get('concepts', [])))
                with col5:
                    avg_rows = schema.get('tables', {}).get('avg_row_count', 0)
                    st.metric("Avg Rows", f"{avg_rows:.0f}" if avg_rows else "N/A")
        except:
            pass
        
        # Enhanced chat input
        st.subheader("💬 Intelligent RAN Chat")
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input("Ask me about your RAN data:", 
                                         value=selected_example if selected_example else "",
                                         placeholder="e.g., Show me power domain insights or Find frequency patterns")
            
            with col2:
                send_button = st.button("Send 📤", use_container_width=True)
        
        # Process query
        if send_button and user_input:
            process_query(user_input)
        
        # Display enhanced chat history
        if st.session_state.chat_history:
            st.subheader("💬 Conversation History")
            display_enhanced_chat_history()
        
        # Enhanced Analytics Dashboard
        with st.expander("📊 Advanced Analytics Dashboard", expanded=False):
            display_enhanced_analytics_dashboard()
    
    else:
        st.warning("Please connect to Neo4j to start chatting with your RAN knowledge graph.")
        
        # Enhanced sample data loading section
        st.subheader("📊 Sample RAN Data")
        display_sample_data_info()

def process_query(user_input: str):
    """Process user query with enhanced functionality and error handling"""
    if st.session_state.chatbot:
        try:
            with st.spinner("🧠 Processing your query with RAN intelligence..."):
                # Process query
                result = st.session_state.chatbot.process_query(user_input)
                response = st.session_state.chatbot.generate_response(result)
                
                # Add metadata for analytics
                query_metadata = {
                    'intent': st.session_state.chatbot.detect_intent(user_input),
                    'type': result.get('type'),
                    'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'domain': result.get('domain'),
                    'pattern_type': result.get('pattern_type')
                }
                
                st.session_state.chat_history.append((user_input, response, query_metadata))
                
                # Show success message
                st.success("✅ Query processed successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            # Add error to chat history for debugging
            error_response = f"I encountered an error while processing your query: {str(e)}\n\nPlease try rephrasing your question or check the connection to Neo4j."
            st.session_state.chat_history.append((user_input, error_response, {'error': True, 'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}))

def display_enhanced_chat_history():
    """Display enhanced chat history with rich formatting and metadata"""
    for i, chat_item in enumerate(reversed(st.session_state.chat_history)):
        if len(chat_item) == 3:
            query, response, metadata = chat_item
        else:
            query, response = chat_item
            metadata = {}
        
        # Create expandable chat items
        is_expanded = i == 0  # Expand most recent
        chat_title = f"💬 {query[:60]}..." if len(query) > 60 else f"💬 {query}"
        
        with st.expander(chat_title, expanded=is_expanded):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.markdown(f"**🧑 User:** {query}")
                st.markdown("**🤖 Assistant:**")
                
                # Enhanced response formatting
                if metadata.get('error'):
                    st.error(response)
                else:
                    st.markdown(response)
            
            with col2:
                if metadata:
                    st.markdown("**📊 Query Analysis:**")
                    
                    if metadata.get('intent'):
                        st.caption(f"🎯 Intent: {metadata['intent']}")
                    if metadata.get('type'):
                        st.caption(f"📋 Type: {metadata['type']}")
                    if metadata.get('domain'):
                        st.caption(f"🏢 Domain: {metadata['domain']}")
                    if metadata.get('pattern_type'):
                        st.caption(f"🔍 Pattern: {metadata['pattern_type']}")
                    if metadata.get('timestamp'):
                        st.caption(f"⏰ Time: {metadata['timestamp']}")
                    
                    if metadata.get('error'):
                        st.caption("❌ Error occurred")

def display_enhanced_analytics_dashboard():
    """Display comprehensive analytics dashboard"""
    if not st.session_state.chatbot:
        st.warning("Connect to Neo4j to view analytics.")
        return
    
    try:
        # Query usage analytics
        if st.session_state.chat_history:
            st.subheader("🔍 Query Analytics")
            
            # Prepare data
            query_data = []
            for chat_item in st.session_state.chat_history:
                if len(chat_item) == 3:
                    _, _, metadata = chat_item
                    query_data.append(metadata)
            
            if query_data:
                df = pd.DataFrame(query_data)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'type' in df.columns:
                        type_counts = df['type'].value_counts()
                        st.bar_chart(type_counts)
                        st.caption("Query Types Distribution")
                
                with col2:
                    if 'intent' in df.columns:
                        intent_counts = df['intent'].value_counts()
                        st.bar_chart(intent_counts)
                        st.caption("Intent Distribution")
                
                with col3:
                    if 'domain' in df.columns:
                        domain_counts = df['domain'].dropna().value_counts()
                        if not domain_counts.empty:
                            st.bar_chart(domain_counts)
                            st.caption("RAN Domain Usage")
        
        # Schema analytics
        st.subheader("📊 Knowledge Graph Insights")
        schema = st.session_state.chatbot.query_interface.get_schema_overview()
        
        if schema and schema.get('relationships'):
            col1, col2 = st.columns(2)
            
            with col1:
                # Relationship distribution
                rel_data = schema['relationships']
                rel_df = pd.DataFrame(rel_data)
                if not rel_df.empty:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(rel_df['relationship_type'], rel_df['count'], color='lightcoral')
                    ax.set_title('Relationship Types Distribution')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            with col2:
                # Conceptual groups
                if schema.get('concepts'):
                    concepts = schema['concepts'][:10]  # Top 10
                    concept_df = pd.DataFrame(concepts)
                    if not concept_df.empty:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(concept_df['concept'], concept_df['usage_count'], color='lightblue')
                        ax.set_title('Top Conceptual Groups')
                        ax.set_xlabel('Usage Count')
                        plt.tight_layout()
                        st.pyplot(fig)
        
        # Export functionality
        st.subheader("📤 Export Options")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Export Chat History"):
                if st.session_state.chat_history:
                    chat_df = pd.DataFrame([
                        {'query': item[0], 'response': item[1], 'metadata': item[2] if len(item) > 2 else {}}
                        for item in st.session_state.chat_history
                    ])
                    st.download_button(
                        label="Download Chat History CSV",
                        data=chat_df.to_csv(index=False),
                        file_name="ran_chat_history.csv",
                        mime="text/csv"
                    )
        
        with col2:
            if st.button("Export Schema"):
                schema = st.session_state.chatbot.query_interface.get_schema_overview()
                if schema:
                    st.download_button(
                        label="Download Schema JSON",
                        data=pd.Series(schema).to_json(),
                        file_name="ran_schema.json",
                        mime="application/json"
                    )
        
        with col3:
            if st.button("Generate NER Data"):
                try:
                    training_data = st.session_state.chatbot.ner_generator.generate_ner_training_data()
                    st.success(f"Generated {len(training_data)} NER training examples!")
                    
                    # Show sample
                    if training_data:
                        st.subheader("Sample NER Training Data")
                        for i, (text, entities) in enumerate(training_data[:3]):
                            st.code(f"Text: {text}\nEntities: {entities}")
                except Exception as e:
                    st.error(f"Error generating NER data: {e}")
        
    except Exception as e:
        st.error(f"Error in analytics dashboard: {str(e)}")

def display_sample_data_info():
    """Display information about sample RAN data"""
    st.markdown("""
    ### 🎯 RAN Domain Sample Data
    
    This system can work with various types of RAN (Radio Access Network) data including:
    """)
    
    sample_categories = {
        "📊 Performance Data": [
            "cell_performance_counters - KPI metrics and throughput data",
            "handover_statistics - Mobility and handover success rates",
            "traffic_volume_stats - Data usage and capacity metrics"
        ],
        "⚙️ Configuration Data": [
            "cell_config_parameters - Cell settings and thresholds",
            "antenna_configuration - Antenna tilt and azimuth settings",
            "power_control_settings - Transmission power parameters"
        ],
        "🌐 Network Topology": [
            "neighbor_relations - Cell adjacency and relationships",
            "site_information - Physical location and equipment data",
            "frequency_planning - Spectrum allocation and usage"
        ],
        "🔍 Quality Metrics": [
            "signal_quality_measurements - RSRP, RSRQ, SINR data",
            "interference_analysis - Noise and interference levels",
            "coverage_statistics - Signal strength and coverage areas"
        ]
    }
    
    for category, items in sample_categories.items():
        st.markdown(f"**{category}**")
        for item in items:
            st.markdown(f"• {item}")
        st.markdown("")
    
    st.info("💡 To get started, connect to Neo4j and load your actual RAN data using the kg_builder module, or use the sample data loading functionality above.")

if __name__ == "__main__":
    main()
