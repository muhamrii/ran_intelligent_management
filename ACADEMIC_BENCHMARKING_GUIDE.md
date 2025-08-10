# Academic Benchmarking Guide

## Overview
The RAN Intelligent Management Chatbot now includes comprehensive academic benchmarking using standard Information Retrieval (IR) and Natural Language Understanding (NLU) evaluation protocols. 

**New Features:**
- 🎯 **Separate Academic Benchmarking Tab** - Dedicated interface in Research Lab
- 📊 **Auto-Generated Sample Data** - KG-based realistic evaluation data  
- 📈 **12 Visualization Types** - Comprehensive analysis dashboard
- 🔄 **Integrated Data Generator** - Creates domain-specific ground truth

## Academic Benchmarking Interface

### Separate Tab Design
Academic benchmarking now has its own dedicated tab in the Research Lab:
1. **Research Lab** → **Academic Benchmarking**
2. Auto-detection of generated sample files
3. Option to use custom uploaded data
4. Comprehensive results with 12 chart types

### Auto-Generated Sample Data
The system automatically uses KG-generated sample data by default:
- ✅ **sample_ir_ground_truth.csv** - 75 IR evaluation queries
- ✅ **sample_nlu_ground_truth.csv** - 75 NLU evaluation examples
- Generated from actual knowledge graph structure
- Domain-specific RAN queries across 6 categories
- 84% query uniqueness, 3.3 avg entities per query

## Information Retrieval (IR) Benchmarks

### Metrics Implemented
- **Precision@K**: Fraction of relevant tables in top-K results
- **Recall@K**: Fraction of relevant tables found in top-K results  
- **Mean Average Precision (MAP)**: Quality of ranking across all queries
- **Mean Reciprocal Rank (MRR)**: Quality of first relevant result

### Auto-Generated IR Data
The system includes 75 auto-generated IR queries covering:
- **Power Analysis**: Energy consumption and optimization
- **Frequency Management**: Spectrum allocation and carriers  
- **Performance Metrics**: KPIs and quality indicators
- **Cell Configuration**: Parameters and antenna settings
- **Neighbor Relations**: Adjacency and handover data
- **Timing Synchronization**: Sync accuracy and parameters
- **Multi-Table Analysis**: Advanced relationship queries

### Ground Truth Format (IR)
CSV with columns: `query,relevant_tables`
```csv
query,relevant_tables
"Show power data","ConsumedEnergyMeasurement,PowerMeasurement,EnergyConsumption"
"Find frequency patterns","EUtranFrequency,FrequencyBand,SpectrumAllocation"
```

### IR Evaluation Process
1. Run E2E evaluation to get query results
2. Upload IR ground truth CSV
3. System extracts table names from contexts
4. Computes P@K, R@K, MAP, MRR for each query
5. Aggregates metrics and shows distributions

## Natural Language Understanding (NLU) Benchmarks

### Metrics Implemented
- **Semantic Similarity**: Embedding-based similarity between generated and expected answers
- **Entity Precision/Recall/F1**: Named entity extraction performance
- **Response Quality**: Length, structure, completeness analysis
- **Intent Distribution**: Query categorization analysis

### Ground Truth Format (NLU)
CSV with columns: `query,answer,entities`
```csv
query,answer,entities
"Show power data","Power consumption data in ConsumedEnergyMeasurement table","ConsumedEnergyMeasurement,consumedEnergyMeasurementId"
```

### NLU Evaluation Process
1. Computes semantic similarity using sentence transformers
2. Extracts entities from responses vs ground truth
3. Calculates precision, recall, F1 for entity extraction
4. Analyzes response quality and structure
5. Shows performance distribution by intent

## Visualizations Provided

### IR Visualizations
- Precision@K and Recall@K bar charts
- MAP vs MRR comparison
- MAP score distribution histogram
- Query-level performance analysis

### NLU Visualizations  
- Semantic similarity distribution
- Entity F1 score distribution
- Intent distribution pie chart
- Response length vs similarity scatter plot
- Entity metrics comparison
- Performance breakdown by intent

## Standard Model Comparison

### Models Supported
- **DistilBERT**: Lightweight BERT variant
- **BERT-base**: Original transformer model
- **RoBERTa-base**: Robustly optimized BERT
- **Sentence-Transformers**: Embedding-based models

### Comparison Methods
- **Zero-shot Classification**: For BERT-like models
- **Similarity-based**: For sentence transformer models
- **Confusion Matrices**: Visual comparison across models
- **Performance Metrics**: Accuracy, macro-F1

## Academic Standards Compliance

### IR Evaluation Standards
- Follows TREC evaluation protocols
- Standard metrics (P@K, R@K, MAP, MRR)
- Ranking-aware evaluation
- Multi-level relevance assessment

### NLU Evaluation Standards
- Semantic similarity using established embeddings
- Entity extraction F1 (CoNLL standard)
- Intent classification accuracy
- Response quality assessment

## Usage Instructions

### Step 1: Run E2E Evaluation
```
Research Lab → End-to-end evaluation → Run E2E evaluation
```

### Step 2: Prepare Ground Truth
- Create IR CSV: query, relevant_tables (comma-separated)
- Create NLU CSV: query, answer, entities (comma-separated)

### Step 3: Run Academic Benchmarks
- Upload IR ground truth CSV
- Upload NLU ground truth CSV  
- Click "Run Academic Benchmarks"

### Step 4: Analyze Results
- Review aggregated metrics
- Examine visualization plots
- Download comprehensive results JSON

## Interpreting Results

### Good IR Performance
- P@1 > 0.7: Most queries find relevant table in top result
- MAP > 0.6: Good overall ranking quality
- MRR > 0.8: Relevant results appear early in rankings

### Good NLU Performance  
- Semantic Similarity > 0.7: Generated answers match expectations
- Entity F1 > 0.8: Good entity extraction accuracy
- Structured Response Rate > 0.9: Consistent formatting

### Model Comparison Insights
- Fine-tuned model should outperform standard models on domain-specific intents
- BERT-like models may excel at general language understanding
- Sentence transformers often good for semantic tasks

## Sample Files
- `sample_ir_ground_truth.csv`: Example IR evaluation data
- `sample_nlu_ground_truth.csv`: Example NLU evaluation data

## Advanced Features

### Custom Evaluation
- Modify entity extraction patterns for domain-specific entities
- Adjust similarity computation methods
- Add custom relevance scoring

### Batch Evaluation
- Process multiple query sets
- Compare different model configurations
- Track performance over time

### Export Options
- Comprehensive JSON results
- Individual metric CSVs
- Visualization plots as images
