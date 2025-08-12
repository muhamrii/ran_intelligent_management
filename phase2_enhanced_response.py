#!/usr/bin/env python3
"""
Phase 2: Enhanced Response Generation for NLU
============================================
Improves response quality using enhanced processing results
"""

def get_enhanced_nlu_response_generator():
    """Generate enhanced response generation code for NLU benchmarking"""
    
    enhanced_response_code = '''
def generate_enhanced_nlu_response(query_result, query, bot):
    """Generate high-quality response using enhanced processing results"""
    result_type = query_result.get('type', 'unknown')
    
    # Use response from enhanced processing if available
    if 'response' in query_result and query_result['response']:
        base_response = query_result['response']
        
        # Enhance with structured formatting if not already present
        if not any(marker in base_response for marker in ['📋', '🔍', '📊', '⚡', '📡', '⏱️']):
            base_response = enhance_response_formatting(base_response, result_type)
        
        return base_response
    
    # Generate enhanced response based on result type
    if result_type == 'explicit_table':
        return generate_table_description_response(query_result)
    elif result_type == 'parallel_aggregated':
        return generate_domain_analysis_response(query_result)
    elif result_type == 'semantic_search':
        return generate_contextual_response(query_result)
    elif result_type == 'domain_inquiry':
        return generate_domain_inquiry_response(query_result)
    else:
        # Fallback with enhancement
        fallback_response = bot.generate_response(query_result)
        return enhance_response_formatting(fallback_response, result_type)

def enhance_response_formatting(response, result_type):
    """Add structured formatting to response"""
    # Add appropriate emoji based on content
    if any(term in response.lower() for term in ['table', 'structure', 'schema']):
        return f"📋 {response}"
    elif any(term in response.lower() for term in ['power', 'energy', 'consumption']):
        return f"⚡ {response}"
    elif any(term in response.lower() for term in ['frequency', 'spectrum', 'carrier']):
        return f"📡 {response}"
    elif any(term in response.lower() for term in ['performance', 'metric', 'kpi']):
        return f"📊 {response}"
    elif any(term in response.lower() for term in ['sync', 'timing', 'clock']):
        return f"⏱️ {response}"
    else:
        return f"🔍 {response}"

def generate_table_description_response(query_result):
    """Generate structured table description response"""
    results = query_result.get('results', {})
    table_name = results.get('table_name', 'Unknown')
    
    # Extract table information
    columns = results.get('columns', [])
    description = results.get('description', '')
    
    response = f"📋 {table_name} table contains network configuration data"
    
    if description:
        response += f" for {description.lower()}"
    
    if columns:
        column_names = [col.get('name', '') for col in columns[:5] if col.get('name')]
        if column_names:
            response += f" with key columns: {', '.join(column_names)}"
    
    response += f". This table manages {get_table_functionality(table_name)} parameters for network optimization."
    
    return response

def generate_domain_analysis_response(query_result):
    """Generate domain analysis response from parallel aggregated results"""
    top_tables = query_result.get('top_tables', [])[:5]
    key_results = query_result.get('key_results', {})
    
    if top_tables:
        table_names = [t.get('table_name', '') for t in top_tables if t.get('table_name')]
        domain = determine_domain_from_tables(table_names)
        
        response = f"🔍 {domain.title()} analysis involves tables: {', '.join(table_names[:3])}"
        
        # Add domain insights if available
        domain_insights = key_results.get('domain_insights', {})
        if domain_insights:
            aspects = domain_insights.get('key_aspects', [])
            if aspects:
                response += f" with focus on {', '.join(aspects[:3])}"
        
        response += " for network performance optimization."
        return response
    
    return "🔍 Network analysis involves relevant configuration tables for system optimization."

def generate_contextual_response(query_result):
    """Generate contextual response from semantic search results"""
    results = query_result.get('results', [])[:3]
    
    if results:
        contexts = [r.get('context', '') for r in results if r.get('context')]
        table_names = [r.get('table_name', '') for r in results if r.get('table_name')]
        
        if table_names:
            response = f"📊 Analysis from {', '.join(table_names)} shows relevant configuration data"
            
            if contexts:
                # Extract key terms from contexts
                key_terms = extract_key_terms_from_contexts(contexts)
                if key_terms:
                    response += f" including {', '.join(key_terms[:4])}"
            
            response += " for network optimization."
            return response
    
    return "📊 Network configuration analysis provides relevant system parameters."

def generate_domain_inquiry_response(query_result):
    """Generate domain inquiry response"""
    results = query_result.get('results', {})
    related_tables = results.get('related_tables', [])
    
    if related_tables:
        table_names = [t.get('table_name', '') for t in related_tables[:4] if t.get('table_name')]
        domain = determine_domain_from_tables(table_names)
        
        response = f"🔍 {domain.title()} inquiry involves tables: {', '.join(table_names)}"
        response += f" with parameters for {domain} optimization and monitoring."
        return response
    
    return "🔍 Domain analysis involves relevant network configuration tables."

def determine_domain_from_tables(table_names):
    """Determine domain based on table names"""
    tables_str = ' '.join(table_names).lower()
    
    if any(term in tables_str for term in ['energy', 'power', 'consumption']):
        return 'power'
    elif any(term in tables_str for term in ['frequency', 'spectrum', 'carrier']):
        return 'frequency'
    elif any(term in tables_str for term in ['cell', 'sector', 'coverage']):
        return 'cell'
    elif any(term in tables_str for term in ['performance', 'counter', 'measurement']):
        return 'performance'
    else:
        return 'network'

def extract_key_terms_from_contexts(contexts):
    """Extract key terms from context strings"""
    import re
    
    all_terms = []
    for context in contexts:
        # Extract technical terms (capitalized words, identifiers)
        terms = re.findall(r'[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*', context)
        all_terms.extend(terms)
    
    # Remove duplicates and return most relevant
    unique_terms = list(dict.fromkeys(all_terms))
    return unique_terms[:4]

def get_table_functionality(table_name):
    """Get functionality description for table (same as Phase 1)"""
    if not table_name:
        return 'network'
    
    name_lower = table_name.lower()
    
    if 'function' in name_lower:
        return 'functional control'
    elif 'profile' in name_lower:
        return 'configuration profile'
    elif 'management' in name_lower:
        return 'resource management'
    elif 'control' in name_lower:
        return 'operational control'
    elif 'config' in name_lower:
        return 'configuration'
    elif 'meter' in name_lower:
        return 'measurement'
    else:
        return 'network element'
'''
    
    return enhanced_response_code

if __name__ == "__main__":
    print("🚀 Phase 2: Enhanced Response Generation")
    print("=" * 50)
    print("\n📋 Key Improvements:")
    print("   • Leverage enhanced processing results")
    print("   • Structured response formatting with emojis")
    print("   • Domain-aware response generation")
    print("   • Context extraction from query results")
    print("   • Fallback enhancement for existing responses")
    print("\n💡 Ready to integrate into NLU benchmarking!")
    
    code = get_enhanced_nlu_response_generator()
    print(f"\n📝 Generated {len(code.split('def '))-1} enhanced functions")
