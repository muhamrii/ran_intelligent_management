"""
Enhanced NLU Functions for RANChatbot
=====================================
Integrated enhanced NLU capabilities from Phases 2-4
"""

import re
import numpy as np
from typing import Dict, List, Any


class EnhancedNLUMixin:
    """Mixin class containing enhanced NLU capabilities"""
    
    def generate_enhanced_nlu_response(self, query_result, query, bot=None):
        """Generate high-quality response using enhanced processing results"""
        result_type = query_result.get('type', 'unknown')
        
        # Use response from enhanced processing if available
        if 'response' in query_result and query_result['response']:
            base_response = query_result['response']
            
            # Enhance with structured formatting if not already present
            if not any(marker in base_response for marker in ['📋', '🔍', '📊', '⚡', '📡', '⏱️']):
                base_response = self.enhance_response_formatting(base_response, result_type)
            
            return base_response
        
        # Generate enhanced response based on result type
        if result_type == 'explicit_table':
            return self.generate_table_description_response(query_result)
        elif result_type == 'parallel_aggregated':
            return self.generate_domain_analysis_response(query_result)
        elif result_type == 'semantic_search':
            return self.generate_contextual_response(query_result)
        elif result_type == 'domain_inquiry':
            return self.generate_domain_inquiry_response(query_result)
        else:
            # Fallback with enhancement
            if bot:
                fallback_response = bot.generate_response(query_result)
            else:
                fallback_response = self.generate_response(query_result)
            return self.enhance_response_formatting(fallback_response, result_type)

    def enhance_response_formatting(self, response, result_type):
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

    def generate_table_description_response(self, query_result):
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
        
        response += f". This table manages {self.get_table_functionality(table_name)} parameters for network optimization."
        
        return response

    def generate_domain_analysis_response(self, query_result):
        """Generate domain analysis response from parallel aggregated results"""
        top_tables = query_result.get('top_tables', [])[:5]
        key_results = query_result.get('key_results', {})
        
        if top_tables:
            table_names = [t.get('table_name', '') for t in top_tables if t.get('table_name')]
            domain = self.determine_domain_from_tables(table_names)
            
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

    def generate_contextual_response(self, query_result):
        """Generate contextual response from semantic search results"""
        results = query_result.get('results', [])[:3]
        
        if results:
            contexts = [r.get('context', '') for r in results if r.get('context')]
            table_names = [r.get('table_name', '') for r in results if r.get('table_name')]
            
            if table_names:
                response = f"📊 Analysis from {', '.join(table_names)} shows relevant configuration data"
                
                if contexts:
                    # Extract key terms from contexts
                    key_terms = self.extract_key_terms_from_contexts(contexts)
                    if key_terms:
                        response += f" including {', '.join(key_terms[:4])}"
                
                response += " for network optimization."
                return response
        
        return "📊 Network configuration analysis provides relevant system parameters."

    def generate_domain_inquiry_response(self, query_result):
        """Generate domain inquiry response"""
        results = query_result.get('results', {})
        related_tables = results.get('related_tables', [])
        
        if related_tables:
            table_names = [t.get('table_name', '') for t in related_tables[:4] if t.get('table_name')]
            domain = self.determine_domain_from_tables(table_names)
            
            response = f"🔍 {domain.title()} inquiry involves tables: {', '.join(table_names)}"
            response += f" with parameters for {domain} optimization and monitoring."
            return response
        
        return "🔍 Domain analysis involves relevant network configuration tables."

    def determine_domain_from_tables(self, table_names):
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

    def extract_key_terms_from_contexts(self, contexts):
        """Extract key terms from context strings"""
        all_terms = []
        for context in contexts:
            # Extract technical terms (capitalized words, identifiers)
            terms = re.findall(r'[A-Z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)*', context)
            all_terms.extend(terms)
        
        # Remove duplicates and return most relevant
        unique_terms = list(dict.fromkeys(all_terms))
        return unique_terms[:4]

    def get_table_functionality(self, table_name):
        """Get functionality description for table"""
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

    # ===== PHASE 3: ENHANCED ENTITY EXTRACTION =====
    
    def extract_enhanced_entities(self, response, query_context, query_result):
        """Enhanced entity extraction using domain knowledge and query context"""
        entities = []
        
        # 1. Extract entities from enhanced query processing results
        entities.extend(self.extract_entities_from_query_results(query_result))
        
        # 2. Extract entities from response text with domain awareness
        entities.extend(self.extract_ran_entities_from_text(response))
        
        # 3. Extract entities from query context if available
        if query_context:
            entities.extend(self.extract_entities_from_context(query_context))
        
        # 4. Normalize and deduplicate
        normalized_entities = self.normalize_and_deduplicate_entities(entities)
        
        return normalized_entities

    def extract_entities_from_query_results(self, query_result):
        """Extract entities from enhanced query processing results"""
        entities = []
        result_type = query_result.get('type', 'unknown')
        
        if result_type == 'explicit_table':
            # Direct table extraction
            results = query_result.get('results', {})
            table_name = results.get('table_name')
            if table_name:
                entities.append(table_name)
                
            # Extract column names
            columns = results.get('columns', [])
            for col in columns[:5]:
                col_name = col.get('name')
                if col_name:
                    entities.append(col_name)
                    # Add table.column format
                    entities.append(f"{table_name}.{col_name}")
        
        elif result_type == 'parallel_aggregated':
            # Extract from top_tables
            top_tables = query_result.get('top_tables', [])
            for table in top_tables[:8]:
                table_name = table.get('table_name')
                if table_name:
                    entities.append(table_name)
                    
            # Extract from domain insights
            key_results = query_result.get('key_results', {})
            domain_insights = key_results.get('domain_insights', {})
            
            # Related tables
            related_tables = domain_insights.get('related_tables', [])
            for table in related_tables[:5]:
                table_name = table.get('table_name')
                if table_name:
                    entities.append(table_name)
            
            # Key aspects as domain entities
            key_aspects = domain_insights.get('key_aspects', [])
            entities.extend(key_aspects[:4])
        
        elif result_type == 'semantic_search':
            # Extract from search results
            results = query_result.get('results', [])
            for result in results[:5]:
                table_name = result.get('table_name')
                if table_name:
                    entities.append(table_name)
        
        elif result_type == 'domain_inquiry':
            # Extract from domain results
            results = query_result.get('results', {})
            related_tables = results.get('related_tables', [])
            for table in related_tables[:5]:
                table_name = table.get('table_name')
                if table_name:
                    entities.append(table_name)
        
        return entities

    def extract_ran_entities_from_text(self, text):
        """Extract RAN-specific entities from text using domain patterns"""
        entities = []
        
        # 1. Table names (CapitalizedWords, often ending with Function/Profile/etc.)
        table_patterns = [
            r'([A-Z][a-zA-Z]*(?:Function|Profile|Management|Control|Config|Setting|Data|Info))',
            r'([A-Z][a-zA-Z]*(?:Cell|Node|Link|Port|Interface)(?:[A-Z][a-zA-Z]*)*)',
            r'([A-Z][a-zA-Z]*(?:Frequency|Spectrum|Carrier|Band)(?:[A-Z][a-zA-Z]*)*)',
            r'([A-Z][a-zA-Z]*(?:Energy|Power|Meter|Measurement)(?:[A-Z][a-zA-Z]*)*)'
        ]
        
        for pattern in table_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)
        
        # 2. Column references (Table.column pattern)
        column_refs = re.findall(r'([A-Z][a-zA-Z0-9]*\.[a-zA-Z][a-zA-Z0-9]*)', text)
        entities.extend(column_refs)
        
        # 3. Technical identifiers (camelCase, with underscores)
        tech_ids = re.findall(r'([a-z][a-zA-Z0-9]*(?:[A-Z][a-zA-Z0-9]*)+)', text)
        entities.extend(tech_ids)
        
        # 4. RAN-specific terminology
        ran_terms = self.extract_ran_terminology(text)
        entities.extend(ran_terms)
        
        # 5. Database/system terms
        system_terms = re.findall(r'(schema|database|table|column|index|key|primary|foreign)', text, re.IGNORECASE)
        entities.extend([term.lower() for term in system_terms])
        
        return entities

    def extract_ran_terminology(self, text):
        """Extract RAN-specific terminology from text"""
        # RAN domain terminology dictionary
        ran_terms_dict = {
            'power': ['energy', 'power', 'consumption', 'efficiency', 'saving', 'optimization'],
            'frequency': ['frequency', 'spectrum', 'bandwidth', 'carrier', 'allocation', 'band'],
            'cell': ['cell', 'sector', 'coverage', 'handover', 'neighbor', 'relation'],
            'performance': ['throughput', 'latency', 'quality', 'KPI', 'RSRP', 'RSRQ', 'measurement'],
            'network': ['network', 'topology', 'connectivity', 'configuration', 'management'],
            'synchronization': ['sync', 'timing', 'clock', 'synchronization', 'accuracy']
        }
        
        found_terms = []
        text_lower = text.lower()
        
        for category, terms in ran_terms_dict.items():
            for term in terms:
                if term in text_lower:
                    found_terms.append(term)
        
        # Additional RAN acronyms and technical terms
        ran_acronyms = ['RSRP', 'RSRQ', 'KPI', 'QCI', 'PCI', 'TAC', 'eNodeB', 'EUtran', 'LTE', 'NR']
        for acronym in ran_acronyms:
            if acronym.lower() in text_lower or acronym in text:
                found_terms.append(acronym)
        
        return found_terms

    def extract_entities_from_context(self, query_context):
        """Extract entities from query processing context"""
        entities = []
        
        # If query_context is a list of table names
        if isinstance(query_context, list):
            entities.extend(query_context)
        
        # If query_context is a dict with table information
        elif isinstance(query_context, dict):
            if 'tables' in query_context:
                entities.extend(query_context['tables'])
            if 'entities' in query_context:
                entities.extend(query_context['entities'])
        
        return entities

    def normalize_and_deduplicate_entities(self, entities):
        """Enhanced normalization and deduplication of entities"""
        def normalize_entity(entity):
            """Normalize entity for better matching"""
            if not entity or not isinstance(entity, str):
                return ""
            
            # Basic cleaning
            entity = str(entity).strip()
            
            # Handle camelCase and PascalCase
            # Convert to lowercase but preserve original if it starts with capital
            if entity and entity[0].isupper() and len(entity) > 1:
                # Keep original format for table names and proper nouns
                return entity
            else:
                # Lowercase for common terms
                return entity.lower()
        
        # Normalize all entities
        normalized = []
        seen = set()
        
        for entity in entities:
            if entity:
                norm_entity = normalize_entity(entity)
                if norm_entity and norm_entity not in seen and len(norm_entity) > 1:
                    normalized.append(norm_entity)
                    seen.add(norm_entity)
                    
                    # Also add lowercase version for matching
                    if norm_entity != norm_entity.lower():
                        seen.add(norm_entity.lower())
        
        return normalized

    def compute_enhanced_entity_metrics(self, predicted_entities, ground_truth_entities):
        """Compute enhanced entity metrics with better normalization"""
        
        # Normalize both sets
        def normalize_for_comparison(entities):
            if isinstance(entities, str):
                entities = [e.strip() for e in entities.split(',') if e.strip()]
            
            normalized = set()
            for entity in entities:
                if entity:
                    # Add original
                    normalized.add(entity.strip())
                    # Add lowercase
                    normalized.add(entity.strip().lower())
                    # Add without special characters
                    clean_entity = re.sub(r'[^a-zA-Z0-9]', '', entity.strip().lower())
                    if clean_entity:
                        normalized.add(clean_entity)
            
            return normalized
        
        pred_norm = normalize_for_comparison(predicted_entities)
        gt_norm = normalize_for_comparison(ground_truth_entities)
        
        if not gt_norm:
            return 0.0, 0.0, 0.0
        
        # Calculate overlap
        overlap = pred_norm & gt_norm
        
        # Enhanced metrics
        precision = len(overlap) / len(pred_norm) if pred_norm else 0.0
        recall = len(overlap) / len(gt_norm) if gt_norm else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return precision, recall, f1

    # ===== PHASE 4: ENHANCED SEMANTIC SIMILARITY =====
    
    def compute_enhanced_semantic_similarity(self, predicted_response, expected_response):
        """Multi-dimensional semantic similarity with domain awareness"""
        
        if not predicted_response or not expected_response:
            return 0.0
        
        # 1. Embedding-based similarity (40% weight)
        embedding_sim = self.compute_embedding_similarity_safe(predicted_response, expected_response)
        
        # 2. Domain terminology overlap (25% weight)
        domain_sim = self.compute_domain_similarity(predicted_response, expected_response)
        
        # 3. Structure similarity (20% weight)
        structure_sim = self.compute_structure_similarity(predicted_response, expected_response)
        
        # 4. Entity mention similarity (15% weight)
        entity_sim = self.compute_entity_mention_similarity(predicted_response, expected_response)
        
        # Weighted combination
        total_similarity = (0.4 * embedding_sim + 
                           0.25 * domain_sim + 
                           0.2 * structure_sim + 
                           0.15 * entity_sim)
        
        return min(1.0, max(0.0, total_similarity))

    def compute_embedding_similarity_safe(self, text1, text2):
        """Safe embedding-based similarity computation"""
        try:
            # Import here to avoid import issues
            from sentence_transformers import SentenceTransformer
            
            # Use a simple model that doesn't require streamlit session state
            try:
                if not hasattr(self, '_embedding_model'):
                    self._embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                model = self._embedding_model
            except:
                # Fallback to token similarity if model loading fails
                return self.compute_token_similarity(text1, text2)
            
            # Generate embeddings
            embeddings = model.encode([str(text1), str(text2)])
            
            # Compute cosine similarity
            sim = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]) + 1e-9
            )
            
            return float(sim)
            
        except Exception as e:
            # Fallback to token-based similarity
            return self.compute_token_similarity(text1, text2)

    def compute_domain_similarity(self, text1, text2):
        """Compute similarity based on domain-specific terminology"""
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

    def compute_structure_similarity(self, text1, text2):
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

    def compute_entity_mention_similarity(self, text1, text2):
        """Compute similarity based on entity mentions"""
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

    def compute_token_similarity(self, text1, text2):
        """Fallback token-based similarity"""
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
