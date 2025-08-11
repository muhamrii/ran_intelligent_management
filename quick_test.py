#!/usr/bin/env python3
"""
Quick test of enhanced chatbot functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot

def quick_test():
    try:
        # Test connection
        integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'neo4j')
        chatbot = EnhancedRANChatbot(integrator, use_caching=True)
        print("✅ Connection successful")
        
        # Test entity extraction
        entities = chatbot.entity_extractor.extract_technical_entities("Show power and frequency data")
        print(f"✅ Entity extraction works: {entities}")
        
        # Test simple query
        result = chatbot.enhanced_process_query("Show me BoundaryOrdinaryClock table")
        print(f"✅ Query processing works: {len(result.get('top_tables', []))} tables found")
        
        # Test cache stats
        if hasattr(chatbot, 'get_cache_stats'):
            stats = chatbot.get_cache_stats()
            print(f"✅ Cache stats: {stats}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"Overall: {'✅ SUCCESS' if success else '❌ FAILED'}")
