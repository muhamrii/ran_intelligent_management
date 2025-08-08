"""
Example usage of the Enhanced RAN Chatbot with Fine-tuning capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from chatbot import EnhancedRANChatbot
from ran_finetuning import train_ran_models, RANDomainModelTrainer

# Assuming you have your Neo4j integrator from the kg_builder module
# from knowledge_graph_module.kg_builder import Neo4jIntegrator

class RANChatbotExample:
    """Example implementation showing how to use the enhanced chatbot"""
    
    def __init__(self, neo4j_uri="bolt://localhost:7687", username="neo4j", password="password"):
        """Initialize with Neo4j connection"""
        # You would initialize your Neo4j integrator here
        # self.neo4j_integrator = Neo4jIntegrator(neo4j_uri, username, password)
        self.neo4j_integrator = None  # Placeholder
        
        # Initialize the enhanced chatbot
        self.chatbot = None
    
    def setup_chatbot(self, use_finetuned_model=False):
        """Setup the chatbot with or without fine-tuned model"""
        if self.neo4j_integrator is None:
            print("Warning: Neo4j integrator not initialized. Using mock setup.")
            return
            
        self.chatbot = EnhancedRANChatbot(
            self.neo4j_integrator, 
            use_domain_model=use_finetuned_model
        )
        print(f"Enhanced RAN Chatbot initialized (Fine-tuned model: {use_finetuned_model})")
    
    def train_domain_model(self):
        """Train the RAN domain-specific model"""
        if self.neo4j_integrator is None:
            print("Error: Neo4j integrator required for training")
            return
            
        print("Starting domain model training...")
        try:
            train_ran_models(self.neo4j_integrator)
            print("Training completed! You can now use the fine-tuned model.")
        except Exception as e:
            print(f"Training failed: {e}")
            print("You can still use the chatbot without the fine-tuned model.")
    
    def demo_queries(self):
        """Demonstrate various types of queries"""
        if self.chatbot is None:
            print("Please setup the chatbot first")
            return
            
        # Example queries that showcase different capabilities
        demo_queries = [
            # Performance analysis
            "Show me performance metrics for cell throughput",
            
            # Power optimization  
            "Find tables with power consumption data",
            
            # Frequency/spectrum management
            "What frequency bands are configured?",
            
            # Quality assessment
            "Show signal quality measurements like RSRP",
            
            # Relationship discovery
            "What tables are related to cell_performance?",
            
            # Multi-hop relationships
            "Find all tables connected to power_metrics through relationships",
            
            # Concept-based search
            "Show me all conceptual groups related to performance",
            
            # Technical entity extraction
            "Get data for cell_id 12345 with frequency 2.4 GHz",
            
            # Schema overview
            "Give me an overview of the database schema"
        ]
        
        print("\n" + "="*60)
        print("RAN CHATBOT DEMO - VARIOUS QUERY TYPES")
        print("="*60)
        
        for i, query in enumerate(demo_queries, 1):
            print(f"\n--- Query {i}: {query} ---")
            
            try:
                # Use enhanced processing
                result = self.chatbot.enhanced_process_query(query)
                
                print(f"Query Type: {result.get('type', 'unknown')}")
                print(f"Detected Intent: {result.get('intent', 'unknown')}")
                
                if result.get('entities'):
                    print(f"Extracted Entities: {result['entities']}")
                
                print("\nResponse:")
                print(result.get('response', 'No response generated'))
                
            except Exception as e:
                print(f"Error processing query: {e}")
            
            print("-" * 50)
    
    def interactive_mode(self):
        """Run interactive chatbot session"""
        if self.chatbot is None:
            print("Please setup the chatbot first")
            return
            
        print("\n" + "="*60)
        print("RAN INTELLIGENT CHATBOT - INTERACTIVE MODE")
        print("="*60)
        print("Type your questions about the RAN data.")
        print("Examples:")
        print("  - 'Show me power consumption data'")
        print("  - 'What tables contain frequency information?'")
        print("  - 'Find relationships between cell performance tables'")
        print("Type 'quit' to exit, 'help' for more examples")
        print("-" * 60)
        
        while True:
            try:
                user_query = input("\n🤖 Your question: ").strip()
                
                if user_query.lower() in ['quit', 'exit', 'bye']:
                    print("Thanks for using RAN Chatbot! Goodbye!")
                    break
                
                if user_query.lower() == 'help':
                    self._show_help()
                    continue
                
                if not user_query:
                    continue
                
                print("\n🔍 Processing your query...")
                
                # Process the query
                result = self.chatbot.enhanced_process_query(user_query)
                
                # Show analysis
                print(f"\n📊 Analysis:")
                print(f"   Query Type: {result.get('type', 'semantic_search')}")
                print(f"   Intent: {result.get('intent', 'general_query')}")
                
                if result.get('entities'):
                    entities_summary = []
                    for entity_type, values in result['entities'].items():
                        if values:
                            entities_summary.append(f"{entity_type}: {', '.join(values[:3])}")
                    if entities_summary:
                        print(f"   Entities: {'; '.join(entities_summary)}")
                
                # Show response
                print(f"\n💬 Response:")
                print(result.get('response', 'Sorry, I could not process your query.'))
                
            except KeyboardInterrupt:
                print("\n\nChatbot session interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try rephrasing your question.")
    
    def _show_help(self):
        """Show help examples"""
        help_examples = {
            "Performance Analysis": [
                "Show performance metrics",
                "Analyze KPI data", 
                "Get throughput statistics"
            ],
            "Power Optimization": [
                "Find power consumption data",
                "Show energy efficiency metrics",
                "Analyze power usage patterns"
            ],
            "Frequency Management": [
                "Show frequency allocations",
                "Find spectrum usage data",
                "Get bandwidth information"
            ],
            "Quality Assessment": [
                "Show signal quality data",
                "Find RSRP measurements",
                "Analyze coverage quality"
            ],
            "Relationship Discovery": [
                "What tables are related to X?",
                "Find connections between tables",
                "Show multi-hop relationships"
            ],
            "Schema Exploration": [
                "Show database overview",
                "List all tables",
                "Get schema structure"
            ]
        }
        
        print("\n📖 Query Examples by Category:")
        print("-" * 40)
        
        for category, examples in help_examples.items():
            print(f"\n{category}:")
            for example in examples:
                print(f"  • {example}")

def main():
    """Main function to run the example"""
    print("RAN Intelligent Management - Enhanced Chatbot Example")
    print("=" * 55)
    
    # Create example instance
    example = RANChatbotExample()
    
    while True:
        print("\nWhat would you like to do?")
        print("1. Setup chatbot (without fine-tuned model)")
        print("2. Setup chatbot (with fine-tuned model)")
        print("3. Train domain-specific model")
        print("4. Run demo queries")
        print("5. Interactive chatbot mode")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        try:
            if choice == '1':
                example.setup_chatbot(use_finetuned_model=False)
            
            elif choice == '2':
                example.setup_chatbot(use_finetuned_model=True)
            
            elif choice == '3':
                example.train_domain_model()
            
            elif choice == '4':
                example.demo_queries()
            
            elif choice == '5':
                example.interactive_mode()
            
            elif choice == '6':
                print("Goodbye!")
                break
            
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
