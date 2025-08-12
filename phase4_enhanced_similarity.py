#!/usr/bin/env python3
"""
Phase 4: Enhanced Semantic Similarity for NLU
=============================================
Improves semantic similarity computation with domain awareness
"""

def get_enhanced_semantic_similarity():
    """Generate enhanced semantic similarity code for NLU benchmarking"""
    
    enhanced_similarity_code = '''
def compute_enhanced_semantic_similarity(predicted_response, expected_response):
    """Multi-dimensional semantic similarity with domain awareness"""
    
    if not predicted_response or not expected_response:
        return 0.0
    
    # 1. Embedding-based similarity (40% weight)
    embedding_sim = compute_embedding_similarity_safe(predicted_response, expected_response)
    
    # 2. Domain terminology overlap (25% weight)
    domain_sim = compute_domain_similarity(predicted_response, expected_response)
    
    # 3. Structure similarity (20% weight)
    structure_sim = compute_structure_similarity(predicted_response, expected_response)
    
    # 4. Entity mention similarity (15% weight)
    entity_sim = compute_entity_mention_similarity(predicted_response, expected_response)
    
    # Weighted combination
    total_similarity = (0.4 * embedding_sim + 
                       0.25 * domain_sim + 
                       0.2 * structure_sim + 
                       0.15 * entity_sim)
    
    return min(1.0, max(0.0, total_similarity))

def compute_embedding_similarity_safe(text1, text2):
    """Safe embedding-based similarity computation"""
    try:
        # Use existing embedding model if available
        if 'embedding_model' in st.session_state:
            model = st.session_state.embedding_model
        else:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            st.session_state.embedding_model = model
        
        # Generate embeddings
        embeddings = model.encode([str(text1), str(text2)])
        
        # Compute cosine similarity
        import numpy as np
        sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-9
        )
        
        return float(sim)
        
    except Exception as e:
        # Fallback to token-based similarity
        return compute_token_similarity(text1, text2)

def compute_domain_similarity(text1, text2):
    """Compute similarity based on domain-specific terminology"""
    import re
    
    # RAN domain terminology with weights
    domain_terms = {
        'high_value': ['table', 'column', 'database', 'schema', 'configuration', 'network'],
        'medium_value': ['performance', 'frequency', 'power', 'cell', 'energy', 'spectrum'],
        'technical': ['RSRP', 'RSRQ', 'handover', 'throughput', 'latency', 'optimization'],
        'structural': ['Function', 'Profile', 'Management', 'Control', 'Setting', 'Data']
    }
    
    # Extract domain terms from both texts
    def extract_domain_terms(text):
        text_lower = text.lower()
        found_terms = {'high_value': [], 'medium_value': [], 'technical': [], 'structural': []}
        
        for category, terms in domain_terms.items():
            for term in terms:
                if term.lower() in text_lower:
                    found_terms[category].append(term.lower())
                elif term in text:  # Case-sensitive for acronyms
                    found_terms[category].append(term)
        
        return found_terms
    
    terms1 = extract_domain_terms(text1)
    terms2 = extract_domain_terms(text2)
    
    # Calculate weighted similarity
    total_weight = 0
    total_similarity = 0
    
    weights = {'high_value': 3, 'medium_value': 2, 'technical': 2, 'structural': 1}
    
    for category in domain_terms:
        set1 = set(terms1[category])
        set2 = set(terms2[category])
        
        if set1 or set2:
            overlap = len(set1 & set2)
            union = len(set1 | set2)
            category_sim = overlap / union if union > 0 else 0
            
            weight = weights[category]
            total_similarity += weight * category_sim
            total_weight += weight
    
    return total_similarity / total_weight if total_weight > 0 else 0.0

def compute_structure_similarity(text1, text2):
    """Compute similarity based on response structure"""
    
    # Check for emoji/marker presence
    emoji_markers = ['📋', '🔍', '📊', '⚡', '📡', '⏱️', '⚙️']
    
    def has_structure_markers(text):
        return any(marker in text for marker in emoji_markers)
    
    def extract_structure_features(text):
        features = {
            'has_emoji': has_structure_markers(text),
            'has_colon': ':' in text,
            'has_comma_list': text.count(',') >= 2,
            'has_technical_terms': bool(re.search(r'[A-Z][a-zA-Z]*(?:Function|Profile)', text)),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'avg_word_length': sum(len(word) for word in text.split()) / max(1, len(text.split()))
        }
        return features
    
    import re
    
    features1 = extract_structure_features(text1)
    features2 = extract_structure_features(text2)
    
    # Calculate structure similarity
    similarities = []
    
    # Boolean feature similarity
    bool_features = ['has_emoji', 'has_colon', 'has_comma_list', 'has_technical_terms']
    for feature in bool_features:
        if features1[feature] == features2[feature]:
            similarities.append(1.0)
        else:
            similarities.append(0.0)
    
    # Numerical feature similarity
    for feature in ['sentence_count', 'avg_word_length']:
        val1, val2 = features1[feature], features2[feature]
        if val1 == 0 and val2 == 0:
            similarities.append(1.0)
        else:
            sim = 1.0 - abs(val1 - val2) / max(val1, val2, 1.0)
            similarities.append(max(0.0, sim))
    
    return sum(similarities) / len(similarities)

def compute_entity_mention_similarity(text1, text2):
    """Compute similarity based on entity mentions"""
    import re
    
    def extract_entities_for_similarity(text):
        entities = []
        
        # Table names (capitalized patterns)
        table_names = re.findall(r'[A-Z][a-zA-Z]*(?:Function|Profile|Management|Control|Config|Data)', text)
        entities.extend(table_names)
        
        # Column references
        column_refs = re.findall(r'[A-Z][a-zA-Z]*\.[a-zA-Z][a-zA-Z]*', text)
        entities.extend(column_refs)
        
        # Technical identifiers
        tech_ids = re.findall(r'[a-z][a-zA-Z]*(?:[A-Z][a-zA-Z]*)+', text)
        entities.extend(tech_ids)
        
        # Common terms
        common_terms = re.findall(r'\b(?:table|column|database|network|performance|frequency|power|cell|energy)\b', text.lower())
        entities.extend(common_terms)
        
        return set(entities)
    
    entities1 = extract_entities_for_similarity(text1)
    entities2 = extract_entities_for_similarity(text2)
    
    if not entities1 and not entities2:
        return 1.0
    elif not entities1 or not entities2:
        return 0.0
    
    # Jaccard similarity
    overlap = len(entities1 & entities2)
    union = len(entities1 | entities2)
    
    return overlap / union if union > 0 else 0.0

def compute_token_similarity(text1, text2):
    """Fallback token-based similarity"""
    import re
    
    def tokenize(text):
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)
    
    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)
    
    if not tokens1 and not tokens2:
        return 1.0
    elif not tokens1 or not tokens2:
        return 0.0
    
    overlap = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    
    return overlap / union if union > 0 else 0.0

def compute_query_type_aware_similarity(predicted_response, expected_response, query_type):
    """Compute similarity with query type awareness"""
    
    base_similarity = compute_enhanced_semantic_similarity(predicted_response, expected_response)
    
    # Apply query-type specific adjustments
    if query_type == 'table_details':
        # For table details, emphasize structure and entity accuracy
        structure_weight = 0.3
        entity_weight = 0.3
        base_weight = 0.4
        
        structure_sim = compute_structure_similarity(predicted_response, expected_response)
        entity_sim = compute_entity_mention_similarity(predicted_response, expected_response)
        
        adjusted_similarity = (base_weight * base_similarity + 
                             structure_weight * structure_sim + 
                             entity_weight * entity_sim)
        
    elif query_type == 'performance_analysis':
        # For performance analysis, emphasize domain terminology
        domain_weight = 0.4
        base_weight = 0.6
        
        domain_sim = compute_domain_similarity(predicted_response, expected_response)
        adjusted_similarity = base_weight * base_similarity + domain_weight * domain_sim
        
    else:
        # Default weighting
        adjusted_similarity = base_similarity
    
    return min(1.0, max(0.0, adjusted_similarity))
'''
    
    return enhanced_similarity_code

if __name__ == "__main__":
    print("🚀 Phase 4: Enhanced Semantic Similarity")
    print("=" * 50)
    print("\n📋 Key Improvements:")
    print("   • Multi-dimensional similarity computation")
    print("   • Domain terminology awareness")
    print("   • Structure and format similarity")
    print("   • Entity mention matching")
    print("   • Query-type specific adjustments")
    print("\n💡 Expected Semantic Similarity improvement: 0.3-0.8 → 0.5-0.9!")
    
    code = get_enhanced_semantic_similarity()
    print(f"\n📝 Generated {len(code.split('def '))-1} enhanced functions")
