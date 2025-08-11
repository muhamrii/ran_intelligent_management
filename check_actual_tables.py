#!/usr/bin/env python3
"""Check what tables are actually in the knowledge graph"""

import sys
import os
sys.path.append('/workspaces/ran_intelligent_management')

from knowledge_graph_module.kg_builder import RANNeo4jIntegrator

def check_actual_tables():
    """Check what tables are actually in the KG"""
    integrator = RANNeo4jIntegrator('bolt://localhost:7687', 'neo4j', 'ranqarag#1')
    
    print("Tables in Knowledge Graph:")
    print("=" * 30)
    
    with integrator.driver.session() as session:
        # Get all table names
        result = session.run("MATCH (t:Table) RETURN t.name as name ORDER BY t.name")
        tables = [record['name'] for record in result]
        
        print(f"Total tables: {len(tables)}")
        print("\nTable names:")
        for i, table in enumerate(tables):
            print(f"{i+1:3d}. {table}")
            
        # Check for our test tables
        test_tables = ['TDD_FRAME_STRUCT', 'CELL_LOCAL_RELATED', 'SectorEquipmentFunction', 
                      'TDD_FRAME_STRUCT_RELATED', 'CELL_EQUIPMENT_RELATED']
        
        print(f"\nTest table status:")
        for table in test_tables:
            status = "✓ Found" if table in tables else "✗ Missing"
            print(f"{table}: {status}")
            
            # Look for similar names
            if table not in tables:
                similar = [t for t in tables if 
                          table.lower() in t.lower() or 
                          t.lower() in table.lower() or
                          any(word in t.upper() for word in table.split('_'))]
                if similar:
                    print(f"  Similar: {similar[:3]}")

if __name__ == "__main__":
    check_actual_tables()
