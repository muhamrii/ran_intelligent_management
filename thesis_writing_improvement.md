# Intelligent Network Management for Radio Access Networks Using Fine‑Tuned Large Language Models and Knowledge Graph Retrieval

Author: <Your Name>
Supervisor: <Supervisor Name>
Institution: <Institution, Program>
Date: August 12, 2025

Note on title: The previously drafted title referenced a “Mixture‑of‑Agent Architecture,” however the final implementation integrates a fine‑tuned intent model and a single LLM‑assisted chatbot grounded by a Neo4j knowledge graph, without a multi‑agent controller. The title is revised accordingly while preserving the core elements: Intelligent Network Management, RAN, fine‑tuned LLM, and Knowledge Graph Retrieval.

---

## Abstract
Radio Access Networks (RAN) comprise thousands of interdependent parameters and tables, making day‑to‑day operations, triage, and performance analysis complex for engineers. This thesis proposes an intelligent network management system that unifies: (i) a parser that converts 3GPP Configuration Management (CM) XML into structured tables, (ii) a typed Neo4j knowledge graph of tables, columns, and discovered relationships, and (iii) a fine‑tuned intent model with an LLM‑assisted, domain‑aware Natural Language Understanding (NLU) pipeline. The knowledge graph encodes semantic links discovered through name similarity, value overlap, RAN‑pattern matches, and reference (key) relationships. The chatbot combines explicit table/column detection, semantic/domain retrieval, intent‑guided ranking boosts, and structured response generation. The implementation provides a Streamlit research UI and an evaluation harness that separately measures Information Retrieval (IR) and NLU quality using curated ground truths. A phased NLU upgrade—enhanced responses, RAN‑aware entity extraction, and multi‑dimensional semantic similarity—improves entity F1 and semantic alignment while preserving IR performance. This study follows Design Science Research Methodology (DSRM) to guide problem identification, artifact design, development, and evaluation. Results indicate robust IR accuracy for explicit queries and consistent NLU gains, suggesting that a schema‑grounded, fine‑tuned LLM approach can increase explainability and operator trust in RAN management.

Keywords: Radio Access Networks (RAN), Knowledge Graph Retrieval, Fine‑Tuned Large Language Models, Intent Detection, Intelligent Network Management, Semantic Similarity, Entity Extraction, Neo4j

---

## 1. Introduction

### 1.1 Background
Modern RAN deployments involve heterogeneous vendors and evolving standards (4G/5G/6G). Engineers must correlate configuration, counters, and alarms that are spread across many tables with opaque naming conventions. 3GPP CM XML typically enumerates network elements (e.g., eNodeB/gNodeB) and vendor‑specific `vsData*` objects that expand into tabular parameters. Columns may be nested, sparsely populated, or vendor‑aliased, and entity relationships are implicit in names and values rather than explicitly declared foreign keys. Traditional tools (SQL/BI dashboards) require detailed schema knowledge and are brittle to naming idiosyncrasies and schema evolution.

LLMs can bridge user intent and data access, but they require strong grounding in structure and domain semantics to avoid hallucinations and off‑target retrieval. A typed knowledge graph encodes structure and relationships, while a fine‑tuned intent detector steers retrieval toward domain‑relevant regions of the schema. A domain‑aware NLU stack provides explicit entity extraction and multi‑signal semantic similarity to stabilize answer quality across query types.

### 1.2 Research Problem
How can accurate, explainable, natural‑language access to RAN configuration and performance data be enabled, ensuring that LLM outputs are schema‑aware and traceable, while providing reliable, reproducible evaluation of both retrieval and NLU quality? The solution must support explicit table/column queries, domain‑oriented requests (e.g., power, performance, timing), and entity‑rich prompts, with consistent behavior under schema growth and vendor variance.

### 1.3 Objectives and Questions
Primary objective: Build and evaluate a system that parses RAN CM data, constructs a typed knowledge graph, and leverages a fine‑tuned intent model with an LLM‑assisted chatbot to answer queries with grounded, structured responses.

Secondary objectives: (a) isolate and preserve IR quality while iterating on NLU; (b) implement reproducible evaluation with curated ground truths; (c) provide explainable outputs with traceable entities and relationships.

Research questions:  
RQ1: What parsing and normalization are needed to convert 3GPP CM XML into a retrieval‑friendly graph?  
RQ2: How should a fine‑tuned intent model be trained and integrated to guide graph retrieval?  
RQ3: Which domain‑aware NLU components most improve answer quality without degrading IR performance?  
RQ4: How can IR and NLU be evaluated separately to support safe iteration?  
Success criteria: measurable NLU improvement (entity F1, semantic similarity) with no statistically significant IR regression on the curated benchmark set.

### 1.4 State of the Art (Summary)
Prior work spans LLMs, knowledge graphs, retrieval‑augmented generation (RAG), and network automation/SON. Many LLM‑centric systems perform text retrieval over documentation but lack fidelity to tabular schemas. Traditional dashboards and SQL tools provide precision but require schema expertise. Graph‑RAG integrates a knowledge graph but often omits explicit intent modeling tied to domain signals that can steer retrieval and explanations. A gap remains in schema‑grounded, explainable LLM answers that leverage a typed graph plus an intent model for domain alignment.

### 1.5 Gap Analysis
Gaps include: (1) limited typed relational structure under LLM answers; (2) weak domain grounding for steering retrieval; (3) scarce evaluation protocols that decouple IR from NLU; and (4) absence of an explicit intent model that informs ranking and explanation. This thesis addresses these by constructing a typed Neo4j graph, training an intent classifier, integrating domain‑aware NLU components, and evaluating with decoupled IR/NLU metrics.

---

## 2. Literature Review

### 2.1 Radio Access Networks (RAN) – evolution and challenges
RAN architectures have evolved from LTE to 5G NR with new spectrum, beamforming, and virtualization trends. Operational data grows in breadth (counters, KPIs) and heterogeneity (vendor‑specific parameters under `vsData*`). Configuration management relies on CM XML aligned to 3GPP Telecommunication Management (28‑series) but leaves practical mapping to operator tooling. These characteristics motivate a schema‑aware, explainable interface for engineers.

References (selection):  
- J. G. Andrews et al., “What Will 5G Be?,” IEEE JSAC, 2014.  
- T. S. Rappaport et al., “Millimeter Wave Mobile Communications for 5G,” IEEE Access, 2013.  
- E. Dahlman, S. Parkvall, J. Sköld, 5G NR, Academic Press, 2018.  
- M. Shafi et al., “5G: A Tutorial Overview of Standards, Trials, Challenges,” IEEE JSAC, 2017.  
- O‑RAN Alliance, White Papers, 2018–.  
- 3GPP TS 28.xxx series, Telecommunication management.

### 2.2 Fine‑Tuned LLMs and Intent Classification
Transformer models support transfer learning for classification, including intent detection. Fine‑tuning a pre‑trained encoder (e.g., BERT family) with domain‑specific utterances improves routing compared to keyword rules. Practical considerations include label taxonomy design, class imbalance, thresholding for confidence, and calibration to support fallbacks when uncertain.

References (selection):  
- A. Vaswani et al., NeurIPS, 2017.  
- J. Devlin et al., NAACL, 2019.  
- T. Brown et al., NeurIPS, 2020.  
- Y. Sun et al., CCL, 2019.  
- J. Howard, S. Ruder, ACL, 2018.  
- Hugging Face Transformers documentation.

### 2.3 Knowledge Graph Retrieval and Graph‑RAG
Knowledge graphs provide typed nodes and relationships that can constrain retrieval and improve explainability. Graph‑RAG augments generation with graph neighborhoods rather than text snippets alone. Neo4j is a practical backbone for schema‑centric applications due to Cypher, constraints, and tooling. Relationship discovery can combine embeddings, value overlap, pattern heuristics, and reference inference to enrich graph connectivity.

References (selection): Hamilton et al., 2017; Nickel et al., 2016; Ji et al., 2021; Lewis et al., 2020; Zhao et al., 2023; Neo4j docs.

### 2.4 Intelligent Network Management and SON
Intelligent network management spans configuration optimization, anomaly detection, and performance analysis. SON research emphasizes automation and policy‑driven adaptation. Integrating LLMs with operator data introduces new opportunities for intent‑driven workflows and explainable analytics when tightly coupled with structured representations like KGs.

References (selection): Aliu et al., 2013; Zhang et al., 2017; Musumeci et al., 2019; O‑RAN Alliance.

### 2.5 Related Works (comparative)
| Approach | Data | Retrieval | LLM/Model | Strengths | Limits |
|---|---|---|---|---|---|
| LLM‑only QA | Unstructured | Vector/keyword | Gen only | Easy to start | Weak schema fidelity |
| BI/SQL | RDB/CSV | SQL | None | Precise | Needs schema knowledge |
| Text‑RAG | Text | Vector | RAG | Good for docs | Weak on tabular semantics |
| Graph‑RAG | KG | Graph + vector | RAG | Structure‑aware | Graph curation needed |
| This work | Typed Neo4j KG | Multi‑strategy + domain | Fine‑tuned intent + LLM NLU | Schema‑aware, traceable | Parser/KG ops required |

---

## 3. Methodology

### 3.1 Research Design (DSRM)
This study adopts DSRM: problem identification → objectives → design/development → demonstration → evaluation → communication. The artifact is the end‑to‑end pipeline plus evaluation harness. Success is defined by reproducible builds, isolated IR benchmarks, and measurable NLU improvements without IR regressions.

### 3.2 System Architecture (code‑aligned)
End‑to‑end modules and data flow:

1) Parser Module (`parser_module/parser.py`)  
• Parses 3GPP CM XML and normalizes `vsData*` tags, including vendor‑specific expansions.  
• Builds per‑table DataFrames; nested attributes are flattened with dot notation (`parent.child`).  
• Outputs `(dfs, metadata, metadata2)`; `metadata2` lists parameter names per table to assist KG construction and UI surfacing.  
• Error handling: tolerant to missing optional attributes; logs anomalies for operator review.

2) Knowledge Graph Module (`knowledge_graph_module/kg_builder.py`)  
• Creates constraints for `Table(name)`, `Column(id)`, `Concept(name)` to enforce uniqueness and speed lookups.  
• Materializes nodes for tables and columns; attaches column statistics (cardinality, sample values) for inspection.  
• Discovers relationships with multiple detectors:  
  – NAME_SIMILARITY via MiniLM‑L6‑v2 embeddings and cosine similarity with thresholds.  
  – VALUE_OVERLAP using Jaccard and overlap percentage on tokenized column values.  
  – PATTERN_MATCH through RAN‑specific regex categories (IDs, time, status, measurement, frequency, power, config).  
  – REFERENCES by subset checks hinting at FK→PK correspondences.  
• Serves schema overviews for UI and feeds retrieval with relationship‑aware context.

3) Intent Model Fine‑Tuning (new, code‑aligned)  
Files: `chatbot_module/ran_training_data.json`, `chatbot_module/ran_finetuning.py`, `chatbot_module/ran_finetuning.ipynb`, `chatbot_module/ran_model_evaluation.ipynb`, artifacts in `chatbot_module/ran_domain_model/`.  
• Labels: intents relevant to RAN operations (e.g., performance_analysis, power_optimization, spectrum_management, cell_configuration, quality_assessment, traffic_analysis, fault_detection, capacity_planning, interference_analysis, handover_optimization).  
• Dataset: curated prompts and short utterances aligned with schema/domain terms; augmented with synonyms.  
• Preprocessing: train/val split; tokenization via `AutoTokenizer`.  
• Model: `AutoModelForSequenceClassification` (Transformer family).  
• Training: cross‑entropy loss; early stopping; class weighting if needed; typical hyperparams (lr ≈ 2e‑5, batch 16–32, epochs 3–8). Confidence calibration and thresholding are used to gate domain boosts at runtime.  
• Evaluation: accuracy, macro‑F1, per‑class F1; confusion matrix analyzed for adjacent‑intent confusions (e.g., performance vs capacity).  
• Export: `config.json`, `tokenizer.json`, `model.safetensors`, `intent_labels.json` in `ran_domain_model/` for runtime use.  
• Data governance: dataset versioned alongside code; label taxonomy documented for reproducibility.

4) Chatbot Module (`chatbot_module/chatbot.py`)  
• Classes: `RANChatbot` (mixed with `EnhancedNLUMixin`) and `EnhancedRANChatbot`.  
• Optional model loading: detects `ran_domain_model/` and loads tokenizer/model when present; otherwise applies keyword rules fallback.  
• Optimized retrieval: explicit table extraction, semantic search, and domain boosts guided by predicted intent (`intent_domain_map`) and synonyms; caching (query/table/embedding/domain) with TTL and warm‑up.  
• Response generation: structured summaries of tables, columns, and relationships; formatting via `enhanced_nlu_functions.py`.  
• KG context: includes related tables and relationship counts in ranking and explanation to improve transparency.

5) NLU Enhancements (`chatbot_module/enhanced_nlu_functions.py`)  
• Phase 2: domain‑aware response templates with structure markers (📋/🔍/📊/⚡/📡/⏱️).  
• Phase 3: entity extraction from query result + text (table/column regex, camelCase/PascalCase, RAN terminology + acronyms), normalization/deduplication, enhanced entity metrics.  
• Phase 4: multi‑dimensional semantic similarity (embedding + domain term overlap + structure + entity‑mentions), safe fallbacks to token similarity for robustness.

6) Streamlit UI & Benchmarks (`chatbot_module/chatbot_ui.py`)  
• Connection panel; auto‑discovery of `ran_domain_model/`; cache metrics.  
• Enhanced examples covering explicit, column‑qualified, domain, and entity queries.  
• Research Lab: separate IR and NLU benchmarks; visual metrics.

### 3.3 Intent Model ↔ Knowledge Graph Relationship
The intent detector informs retrieval by mapping predicted intents to domains (e.g., performance → performance metrics, power → energy/power tables). The chatbot applies:  
• Domain boosts: intents bias ranking toward domain‑relevant tables (via cached domain buckets).  
• Relationship‑aware context: KG relationships (NAME_SIMILARITY, REFERENCES, etc.) expand candidates and provide explanation fields (related tables, relationship types).  
• Confidence gating: boosts are applied only when model confidence exceeds a threshold; otherwise the system defaults to neutral ranking.  
• Safety: when the model is absent or uncertain, keyword rules and semantic search provide robust fallbacks.

Ranking formulation (conceptual):  
score(table) = α · semantic(table, query) + β · domain_boost(intent, table) + γ · relationship_context(table), with α, β, γ tuned to avoid IR regressions.

### 3.4 Implementation Notes
• Regex for explicit tables supports uppercase, snake_case, and PascalCase; column patterns include camelCase/PascalCase.  
• Caching warm‑up loads all table names and builds domain buckets for fast boosts; cache keys include query text and normalized entities.  
• The UI displays cache hit rate, cached tables, and extraction quality; provides quick “Enhanced Features” tests.  
• Logging captures extraction steps, applied boosts, and selected relationships to aid explainability.  
• Configuration is minimal and file‑path driven; absence of a fine‑tuned model triggers automatic fallback.  
• All NLU upgrades are isolated from IR logic to prevent regressions by design.

---

## 4. Evaluation Design

### 4.1 Datasets and Ground Truth
• IR: `improved_ir_ground_truth.csv` with input queries and expected tables/columns, used to compute retrieval precision/recall/F1.  
• NLU: `enhanced_nlu_ground_truth.csv` (100 entries; query/answer/entities) for semantic similarity and entity extraction metrics.  
• Intent: `ran_training_data.json` for fine‑tuning; model artifacts in `ran_domain_model/` for runtime inference.

### 4.2 Metrics
• IR: precision/recall/F1 for tables and columns; mean reciprocal rank (MRR) as an optional rank metric.  
• NLU: multi‑dimensional semantic similarity; entity precision/recall/F1; structured‑response rate; answer latency.  
• Intent: accuracy, macro‑F1, per‑class F1, confusion matrix; confidence calibration quality (ECE) if available.

### 4.3 Protocol
• Baseline: IR pipeline with keyword intent rules enabled and no fine‑tuned model loaded.  
• Intent Ablation: enable the fine‑tuned model; measure retrieval and end‑to‑end answer impact versus baseline.  
• NLU Phases: P1 (ground truth fix), P2 (responses), P3 (entities), P4 (similarity); measure deltas with IR held constant via isolated retrieval logic.  
• Robustness: validate that fallbacks preserve answer quality when the model is missing or low‑confidence; confirm that confidence gating prevents adverse boosts.  
• Reproducibility: fix random seeds for data splits; version datasets and model artifacts together with the codebase.

---

## 5. Results

Summary aligned to the current implementation and validations:
• IR: Near‑perfect explicit table retrieval on the curated set; competitive domain/entity retrieval aided by relationship‑aware boosts; no regressions observed during NLU upgrades due to retrieval isolation.  
• Intent: The fine‑tuned model increases routing accuracy relative to keyword rules; improves ranking alignment and response focus. Confusions primarily occur among closely related intents (e.g., performance vs capacity), which is mitigated by confidence gating.  
• NLU: Structured responses increase answer consistency; RAN‑aware extraction improves entity F1; multi‑dimensional similarity stabilizes semantic alignment across query forms. Caching reduces median latency and supports interactive use.

Ablation notes:  
• P2 (responses) yields immediate gains in structured‑response rate and perceived clarity.  
• P3 (entities) raises entity precision/recall and enhances explainability via explicit entity lists.  
• P4 (similarity) reduces variance in semantic alignment, especially for domain‑heavy queries.  
• Enabling the intent model provides measurable ranking gains on domain queries without harming explicit retrieval.

---

## 6. Discussion

### 6.1 Contributions
• A practical, code‑complete pipeline from 3GPP parsing to a typed Neo4j graph to a fine‑tuned, intent‑aware chatbot.  
• A multi‑signal KG relationship discovery pipeline tailored to RAN data.  
• A phase‑based NLU enhancement stack and a decoupled IR/NLU benchmark harness.  
• Integration of a fine‑tuned intent model with domain boosts over a KG for explainable improvements.

### 6.2 Practical Impact
• Faster, explainable schema exploration; grounded, structured answers; operator‑friendly interaction.  
• Intent‑guided ranking improves relevance for performance, power, frequency, timing, and configuration tasks.  
• Caching supports responsive UX as graph size and table counts grow.

### 6.3 Risks and Mitigations
• Hallucinations → Grounding in KG; explicit table/column mentions; conservative templates with structured sections.  
• Model drift → Versioned ground truths and datasets; periodic re‑training; keyword fallbacks; confidence thresholds.  
• Graph staleness → Scheduled re‑ingestion and relationship refresh; change detection and alerting; provenance metadata.

### 6.4 Future Work
• Enrich ontology and counters; concept hierarchies; provenance across hops.  
• Graph and cross‑encoder rerankers for retrieval; uncertainty estimates and confidence bands in responses.  
• Larger public‑like datasets for broader generalization; online learning from operator feedback.

---

## 7. Conclusion
This thesis presents an intelligent, schema‑grounded RAN management system unifying a fine‑tuned intent model, a Neo4j knowledge graph, and an LLM‑assisted NLU stack. The approach keeps IR strong while improving NLU quality, producing explainable answers with traceable entities and relationships. The codebase includes reproducible parsing, KG construction, model training/evaluation, and a Streamlit research UI, enabling future extensions in both retrieval and generation. The separation of IR and NLU evaluation enables safe iteration and supports continued research on domain‑aware LLM systems for telecom operations.

---

## 8. References (IEEE style)
[1] J. G. Andrews et al., IEEE JSAC, 2014.  
[2] T. S. Rappaport et al., IEEE Access, 2013.  
[3] E. Dahlman et al., 5G NR, Academic Press, 2018.  
[4] M. Shafi et al., IEEE JSAC, 2017.  
[5] O‑RAN Alliance, White Papers.  
[6] 3GPP TS 28.xxx, Telecommunication management.  
[7] A. Vaswani et al., NeurIPS, 2017.  
[8] J. Devlin et al., NAACL, 2019.  
[9] T. B. Brown et al., NeurIPS, 2020.  
[10] Y. Sun et al., CCL, 2019.  
[11] J. Howard, S. Ruder, ACL, 2018.  
[12] W. Hamilton et al., 2017.  
[13] M. Nickel et al., Proc. IEEE, 2016.  
[14] S. Ji et al., IEEE TKDE, 2021.  
[15] P. Lewis et al., NeurIPS, 2020.  
[16] S. Zhao et al., 2023.  
[17] C. Sun et al., ACL, 2019.  
[18] M. Aliu et al., IEEE Comm. Surveys & Tutorials, 2013.  
[19] H. Zhang et al., IEEE Comm. Mag., 2017.  
[20] Neo4j product documentation.

---

## Appendices

### A. Code Artifacts and Paths (for replication)
• Parser: `parser_module/parser.py`  
• Knowledge Graph: `knowledge_graph_module/kg_builder.py`  
• Intent training: `chatbot_module/ran_finetuning.py`, `ran_finetuning.ipynb`, data `ran_training_data.json`  
• Model artifacts: `chatbot_module/ran_domain_model/` (config, tokenizer, weights, labels)  
• Chatbot: `chatbot_module/chatbot.py` (loads model if available; keyword fallback)  
• NLU: `chatbot_module/enhanced_nlu_functions.py`  
• UI/Benchmarks: `chatbot_module/chatbot_ui.py`  
• Ground truths: `improved_ir_ground_truth.csv`, `enhanced_nlu_ground_truth.csv`

### B. Suggested Figures
• End‑to‑end architecture (Parser → KG → Intent Model → Chatbot → UI)  
• KG relationship types and examples  
• Intent model training/evaluation pipeline  
• IR vs NLU benchmark separation and metrics

### C. Environment and Reproducibility Notes
• Python: see `requirements.txt`; ensure compatible versions for Transformers, SentenceTransformers, Neo4j driver, Streamlit.  
• Data paths: paths are file‑based; the chatbot auto‑detects `ran_domain_model/` if present.  
• Reproducibility: datasets and model artifacts are versioned with the repository; random seeds fixed in fine‑tuning scripts/notebooks.  
• Privacy: training data should avoid sensitive operator information; use synthetic or anonymized prompts when necessary.
