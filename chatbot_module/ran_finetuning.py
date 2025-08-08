"""
RAN Domain-Specific Fine-tuning Module
Fine-tunes language models specifically for RAN knowledge graph queries and entities
"""

import json
import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from datasets import Dataset
import logging
from datetime import datetime

# These will need to be installed: pip install transformers datasets torch
try:
    from transformers import (
        AutoTokenizer, 
        AutoModelForSequenceClassification, 
        Trainer, 
        TrainingArguments,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not installed. Run: pip install transformers datasets torch")

class RANDomainModelTrainer:
    """Train a RAN-specific model for intent classification and entity recognition"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.model_name = "distilbert-base-uncased"
        
        # RAN-specific intents for fine-tuning
        self.ran_intents = {
            'performance_analysis': 'Analyze network performance metrics and KPIs',
            'power_optimization': 'Optimize power consumption and efficiency',
            'spectrum_management': 'Manage frequency spectrum allocation and bandwidth',
            'cell_configuration': 'Configure cell parameters and settings',
            'quality_assessment': 'Assess signal quality and coverage metrics',
            'traffic_analysis': 'Analyze network traffic patterns and load',
            'fault_detection': 'Detect and diagnose network faults and issues',
            'capacity_planning': 'Plan network capacity and resource allocation',
            'interference_analysis': 'Analyze and mitigate interference sources',
            'handover_optimization': 'Optimize handover procedures and mobility'
        }
        
        # RAN domain vocabulary
        self.ran_vocabulary = {
            'performance_terms': ['kpi', 'throughput', 'latency', 'performance', 'metric', 'benchmark'],
            'power_terms': ['power', 'energy', 'consumption', 'efficiency', 'dbm', 'watts', 'battery'],
            'spectrum_terms': ['frequency', 'spectrum', 'bandwidth', 'channel', 'carrier', 'mhz', 'ghz'],
            'cell_terms': ['cell', 'site', 'antenna', 'base_station', 'node', 'enb', 'gnb'],
            'quality_terms': ['rsrp', 'rsrq', 'sinr', 'quality', 'signal', 'coverage', 'interference'],
            'traffic_terms': ['traffic', 'volume', 'data', 'load', 'usage', 'throughput', 'session'],
            'mobility_terms': ['handover', 'mobility', 'roaming', 'tracking', 'movement', 'transition'],
            'config_terms': ['config', 'parameter', 'setting', 'threshold', 'value', 'configuration']
        }
    
    def generate_training_data(self) -> List[Dict]:
        """Generate domain-specific training data from KG"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for training")
            
        training_data = []
        
        try:
            # Extract real column names and table structures
            with self.integrator.driver.session() as session:
                # Get representative samples from each conceptual group
                result = session.run("""
                    MATCH (c1:Column)-[r:CONCEPTUAL_GROUP]-(c2:Column)
                    WHERE r.semantic_category IS NOT NULL
                    WITH r.semantic_category as category, 
                         collect(DISTINCT c1.name)[0..20] as sample_columns,
                         collect(DISTINCT c1.table_name)[0..10] as sample_tables
                    RETURN category, sample_columns, sample_tables
                    LIMIT 50
                """)
                
                for record in result:
                    category = record['category']
                    columns = record['sample_columns'] or []
                    tables = record['sample_tables'] or []
                    
                    # Generate realistic queries for this category
                    templates = self._get_category_templates(category)
                    for template in templates:
                        for column in columns[:5]:  # Limit to avoid too much data
                            for table in tables[:3]:
                                try:
                                    query = template.format(column=column, table=table)
                                    intent = self._map_category_to_intent(category)
                                    
                                    training_data.append({
                                        'text': query,
                                        'intent': intent,
                                        'label': list(self.ran_intents.keys()).index(intent),
                                        'entities': {
                                            'column': column,
                                            'table': table,
                                            'category': category
                                        }
                                    })
                                except (ValueError, KeyError):
                                    continue
                
                # Add general RAN queries
                general_queries = self._generate_general_ran_queries()
                training_data.extend(general_queries)
                
        except Exception as e:
            logging.error(f"Error generating training data: {e}")
            # Fallback to synthetic data
            training_data = self._generate_synthetic_training_data()
        
        return training_data
    
    def _get_category_templates(self, category: str) -> List[str]:
        """Get query templates for specific RAN categories"""
        templates = {
            'power_related': [
                "Show me {column} values from {table}",
                "What is the average {column} in {table}?",
                "Find tables with power consumption data like {column}",
                "Compare {column} across different cells",
                "Analyze power efficiency using {column}",
                "Get power consumption metrics from {column}"
            ],
            'frequency_related': [
                "Get frequency data from {column} in {table}",
                "Show spectrum allocation for {column}",
                "Find all frequency-related columns similar to {column}",
                "Analyze spectrum usage in {table}",
                "What frequency bands are in {column}?",
                "Show bandwidth allocation from {table}"
            ],
            'performance_metrics': [
                "Analyze {column} performance in {table}",
                "Show KPI trends for {column}",
                "Compare {column} across time periods",
                "Get performance metrics from {table}",
                "What are the performance indicators in {column}?",
                "Show throughput data from {column}"
            ],
            'cell_identifiers': [
                "Find cell information for {column}",
                "Show all data for cell {column}",
                "Get cell configuration from {table}",
                "List cells in {table}",
                "What cells are configured in {column}?",
                "Show cell topology from {table}"
            ],
            'quality_metrics': [
                "Show signal quality from {column}",
                "Analyze coverage using {column}",
                "Get RSRP values from {table}",
                "What is the signal quality in {column}?",
                "Show interference data from {table}",
                "Analyze quality metrics in {column}"
            ]
        }
        return templates.get(category, templates['performance_metrics'])
    
    def _map_category_to_intent(self, category: str) -> str:
        """Map semantic categories to intents"""
        mapping = {
            'power_related': 'power_optimization',
            'frequency_related': 'spectrum_management',
            'performance_metrics': 'performance_analysis',
            'cell_identifiers': 'cell_configuration',
            'quality_metrics': 'quality_assessment',
            'traffic_counters': 'traffic_analysis',
            'fault_indicators': 'fault_detection',
            'capacity_metrics': 'capacity_planning',
            'interference_data': 'interference_analysis',
            'mobility_data': 'handover_optimization'
        }
        return mapping.get(category, 'performance_analysis')
    
    def _generate_general_ran_queries(self) -> List[Dict]:
        """Generate general RAN domain queries"""
        queries = [
            # Performance analysis
            ("What are the KPI values for cell performance?", "performance_analysis"),
            ("Show me throughput metrics", "performance_analysis"),
            ("Analyze network performance trends", "performance_analysis"),
            
            # Power optimization
            ("How can I reduce power consumption?", "power_optimization"),
            ("Show power efficiency metrics", "power_optimization"),
            ("What are the energy consumption patterns?", "power_optimization"),
            
            # Spectrum management
            ("Show frequency allocation", "spectrum_management"),
            ("What spectrum is being used?", "spectrum_management"),
            ("Analyze bandwidth utilization", "spectrum_management"),
            
            # Cell configuration
            ("Show cell configuration parameters", "cell_configuration"),
            ("What are the cell settings?", "cell_configuration"),
            ("Display antenna configurations", "cell_configuration"),
            
            # Quality assessment
            ("Show signal quality metrics", "quality_assessment"),
            ("What is the coverage quality?", "quality_assessment"),
            ("Analyze RSRP and RSRQ values", "quality_assessment"),
            
            # Traffic analysis
            ("Show traffic volume", "traffic_analysis"),
            ("Analyze data usage patterns", "traffic_analysis"),
            ("What is the network load?", "traffic_analysis"),
            
            # Fault detection
            ("Are there any network faults?", "fault_detection"),
            ("Show error rates", "fault_detection"),
            ("Detect network anomalies", "fault_detection"),
            
            # Capacity planning
            ("Plan network capacity", "capacity_planning"),
            ("Show resource utilization", "capacity_planning"),
            ("Analyze capacity requirements", "capacity_planning"),
            
            # Interference analysis
            ("Show interference sources", "interference_analysis"),
            ("Analyze signal interference", "interference_analysis"),
            ("What causes interference?", "interference_analysis"),
            
            # Handover optimization
            ("Optimize handover procedures", "handover_optimization"),
            ("Show mobility patterns", "handover_optimization"),
            ("Analyze handover success rates", "handover_optimization")
        ]
        
        training_data = []
        for query, intent in queries:
            training_data.append({
                'text': query,
                'intent': intent,
                'label': list(self.ran_intents.keys()).index(intent),
                'entities': {}
            })
        
        return training_data
    
    def _generate_synthetic_training_data(self) -> List[Dict]:
        """Generate synthetic training data as fallback"""
        synthetic_data = []
        
        for intent, description in self.ran_intents.items():
            # Generate variations of the intent description
            variations = [
                description,
                description.lower(),
                f"I want to {description.lower()}",
                f"Can you help me {description.lower()}?",
                f"Show me how to {description.lower()}"
            ]
            
            for variation in variations:
                synthetic_data.append({
                    'text': variation,
                    'intent': intent,
                    'label': list(self.ran_intents.keys()).index(intent),
                    'entities': {}
                })
        
        return synthetic_data
    
    def prepare_training_dataset(self, training_data: List[Dict]) -> Dataset:
        """Prepare dataset for training"""
        texts = [item['text'] for item in training_data]
        labels = [item['label'] for item in training_data]
        
        return Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
    
    def train_ran_model(self, output_dir: str = "./ran_domain_model"):
        """Train the RAN domain-specific model"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for training")
            
        print("Generating training data...")
        training_data = self.generate_training_data()
        print(f"Generated {len(training_data)} training samples")
        
        dataset = self.prepare_training_dataset(training_data)
        
        # Initialize model and tokenizer
        print("Loading base model...")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=len(self.ran_intents)
        )
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)
        
        print("Tokenizing dataset...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            save_steps=1000,
            evaluation_strategy="no",  # No evaluation set for now
            save_total_limit=2,
        )
        
        # Train model
        print("Starting training...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer,
        )
        
        trainer.train()
        
        # Save model
        print(f"Saving model to {output_dir}")
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)
        
        # Save intent labels
        with open(f"{output_dir}/intent_labels.json", 'w') as f:
            json.dump(list(self.ran_intents.keys()), f)
        
        print("Training completed successfully!")
        return model, tokenizer
    
    def evaluate_model(self, model_path: str = "./ran_domain_model"):
        """Evaluate the trained model with test queries"""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers library is required for evaluation")
            
        try:
            # Load the trained model
            classifier = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path
            )
            
            # Load intent labels
            with open(f"{model_path}/intent_labels.json", 'r') as f:
                intent_labels = json.load(f)
            
            # Test queries
            test_queries = [
                "Show me power consumption data",
                "What are the frequency allocations?",
                "Analyze cell performance metrics",
                "Find signal quality issues",
                "Show network traffic patterns",
                "Detect any faults in the network",
                "Plan capacity for next quarter",
                "What's causing interference?",
                "Optimize handover parameters",
                "Show KPI dashboard"
            ]
            
            print("\n=== Model Evaluation ===")
            for query in test_queries:
                result = classifier(query)
                predicted_intent = intent_labels[int(result[0]['label'].split('_')[-1])]
                confidence = result[0]['score']
                
                print(f"Query: '{query}'")
                print(f"Predicted: {predicted_intent} (confidence: {confidence:.3f})")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error during evaluation: {e}")
    
    def export_training_data(self, filepath: str = "./ran_training_data.json"):
        """Export training data for manual review or external training"""
        training_data = self.generate_training_data()
        
        with open(filepath, 'w') as f:
            json.dump(training_data, f, indent=2)
        
        print(f"Training data exported to {filepath}")
        print(f"Total samples: {len(training_data)}")
        
        # Print statistics
        intent_counts = {}
        for item in training_data:
            intent = item['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("\nIntent distribution:")
        for intent, count in sorted(intent_counts.items()):
            print(f"  {intent}: {count} samples")

class RANEntityRecognitionTrainer:
    """Train NER model for RAN-specific entities"""
    
    def __init__(self, neo4j_integrator):
        self.integrator = neo4j_integrator
        self.entity_types = {
            'TABLE_NAME': 'B-TAB',
            'COLUMN_NAME': 'B-COL',
            'CELL_ID': 'B-CELL',
            'FREQUENCY': 'B-FREQ',
            'POWER_VALUE': 'B-PWR',
            'METRIC_NAME': 'B-MET',
            'TIME_VALUE': 'B-TIME'
        }
    
    def generate_ner_training_data(self) -> List[Tuple[str, List[Tuple]]]:
        """Generate NER training data in spaCy format"""
        training_data = []
        
        try:
            with self.integrator.driver.session() as session:
                # Get entity examples from the graph
                result = session.run("""
                    MATCH (t:Table)-[:HAS_COLUMN]->(c:Column)
                    RETURN t.name as table_name, collect(c.name)[0..10] as column_names
                    LIMIT 50
                """)
                
                for record in result:
                    table_name = record['table_name']
                    column_names = record['column_names'] or []
                    
                    # Generate sentences with entities
                    sentences = [
                        f"Show data from {table_name} table",
                        f"Get {column_names[0]} from {table_name}" if column_names else f"Query {table_name}",
                        f"Find tables related to {table_name}",
                        f"What columns are in {table_name}?"
                    ]
                    
                    for sentence in sentences:
                        entities = []
                        
                        # Find table name position
                        table_start = sentence.find(table_name)
                        if table_start != -1:
                            entities.append((table_start, table_start + len(table_name), 'TABLE_NAME'))
                        
                        # Find column name position
                        for col in column_names[:2]:  # Limit to first 2 columns
                            col_start = sentence.find(col)
                            if col_start != -1:
                                entities.append((col_start, col_start + len(col), 'COLUMN_NAME'))
                        
                        if entities:
                            training_data.append((sentence, entities))
                            
        except Exception as e:
            logging.error(f"Error generating NER training data: {e}")
        
        return training_data
    
    def export_ner_training_data(self, filepath: str = "./ran_ner_training.json"):
        """Export NER training data"""
        training_data = self.generate_ner_training_data()
        
        # Convert to a more readable format
        export_data = []
        for text, entities in training_data:
            export_data.append({
                'text': text,
                'entities': [{'start': start, 'end': end, 'label': label} for start, end, label in entities]
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"NER training data exported to {filepath}")
        print(f"Total samples: {len(export_data)}")

# Example usage function
def train_ran_models(neo4j_integrator):
    """Train both intent classification and NER models"""
    print("Starting RAN domain model training...")
    
    # Train intent classification model
    intent_trainer = RANDomainModelTrainer(neo4j_integrator)
    
    try:
        model, tokenizer = intent_trainer.train_ran_model()
        print("Intent classification model trained successfully!")
        
        # Evaluate the model
        intent_trainer.evaluate_model()
        
    except Exception as e:
        print(f"Error training intent model: {e}")
        print("Exporting training data for manual review...")
        intent_trainer.export_training_data()
    
    # Train NER model
    ner_trainer = RANEntityRecognitionTrainer(neo4j_integrator)
    ner_trainer.export_ner_training_data()
    print("NER training data exported!")
    
    print("Model training process completed.")

if __name__ == "__main__":
    # Example usage - you would need to provide your neo4j_integrator
    print("RAN Fine-tuning Module")
    print("This module requires a Neo4j integrator instance to run.")
    print("Example usage:")
    print("  from ran_finetuning import train_ran_models")
    print("  train_ran_models(your_neo4j_integrator)")
