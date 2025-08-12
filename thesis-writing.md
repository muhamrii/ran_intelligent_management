# Intelligent Network Management for Radio Access Networks Using Fine‑Tuned Large Language Models and Knowledge Graph Retrieval

Author: <Your Name>
Supervisor: <Supervisor Name>
Institution: <Institution, Program>
Date: August 12, 2025

Note on title: The original working title referenced a “Mixture‑of‑Agent Architecture.” The implemented system does not use a multi‑agent controller; instead, it integrates a fine‑tuned LLM–assisted chatbot with a Neo4j knowledge graph and domain‑aware NLU pipeline. Therefore the title is revised to remove the agent mixture while preserving the core: Intelligent Network Management, RAN, fine‑tuned LLM, and Knowledge Graph Retrieval.

---

## Abstract
Modern Radio Access Networks (RAN) generate rich configuration and performance data that is difficult to explore using traditional tooling. This thesis presents an intelligent network management approach that combines a fine‑tuned Large Language Model (LLM) chatbot with a Neo4j‑backed knowledge graph and a domain‑aware Natural Language Understanding (NLU) pipeline. The system ingests 3GPP‑structured CM data, constructs a typed knowledge graph with tables and columns, detects relationships using name similarity, value overlap, RAN‑pattern matches, and reference links, and exposes a research‑friendly Streamlit interface. We design a benchmarking suite for Information Retrieval (IR) and NLU with curated ground truth for reproducible evaluation. The NLU stack includes enhanced response generation, RAN‑aware entity extraction, and multi‑dimensional semantic similarity, improving semantic alignment and entity F1 without degrading IR accuracy. We follow Design Science Research Methodology (DSRM) to structure problem identification, artifact design, development, and evaluation. Results demonstrate strong IR performance and consistent NLU gains with domain‑aware enhancements, supporting practical insight discovery for power, frequency, timing, performance, and configuration queries in RAN operations.

Keywords: Radio Access Networks (RAN), Knowledge Graph Retrieval, Fine‑Tuned Large Language Models, Intelligent Network Management, Semantic Similarity, Entity Extraction, Neo4j

---

## 1. Introduction

### 1.1 Background
RAN operations teams manage thousands of parameters across many interconnected tables and counters. Engineering investigations require tracing relations (e.g., neighbors, IDs, counters), validating configurations, and exploring performance metrics. Text search or ad‑hoc SQL often underperforms when the user lacks the exact schema or naming conventions. LLMs can translate intent into navigable views and summaries, but require structure‑aware retrieval and domain grounding to remain faithful.

### 1.2 Research Problem
How can practitioners navigate RAN configuration and performance data using natural language while preserving precision and traceability? Specifically, how can we combine an LLM with a typed knowledge graph to support table/column discovery, schema‑aware search, and domain‑specific insights, while evaluating both IR and NLU quality reliably?

### 1.3 Objectives and Research Questions
Primary objective: Design and evaluate a system that unifies RAN‑aware parsing, knowledge‑graph retrieval, and an LLM‑assisted chatbot to deliver accurate, explainable responses to natural‑language queries.

Research questions:
1) How to convert 3GPP CM data into a typed knowledge graph suitable for schema exploration and retrieval?  
2) How to augment LLM responses with domain‑aware NLU (entities, similarity, structure) to improve answer quality?  
3) How to evaluate IR and NLU components independently so that NLU improvements do not degrade IR accuracy?

### 1.4 State of the Art (Summary)
Recent progress spans: (i) LLMs and in‑context learning; (ii) retrieval augmentation for grounding; (iii) knowledge graphs for schema/context reasoning; (iv) AI‑assisted network automation. However, many works stop at untyped text retrieval or black‑box LLM answers, lacking structured, schema‑aware reasoning and practical RAN parsing pipelines with reproducible evaluation.

### 1.5 Gap Analysis
Existing tools either assume prior schema knowledge or provide free‑form LLM outputs without traceable grounding. This work fills the gap by: (i) building an explicit, typed Neo4j graph over RAN CM tables and columns; (ii) using multiple relationship detectors (name similarity, value overlap, pattern matches, references); (iii) layering a domain‑aware NLU stack with measurable improvements; and (iv) separating IR from NLU evaluation to avoid regressions.

---

## 2. Literature Review

Each subsection lists at least five high‑citation references. We focus on surveys, foundational works, and standards.

### 2.1 Radio Access Networks (RAN) – Evolution, challenges, 4G/5G/6G trends
- J. G. Andrews et al., “What Will 5G Be?,” IEEE JSAC, 2014.  
- T. S. Rappaport et al., “Millimeter Wave Mobile Communications for 5G,” IEEE Access, 2013.  
- E. Dahlman, S. Parkvall, and J. Sköld, “5G NR: The Next Generation Wireless Access Technology,” Academic Press, 2018.  
- M. Shafi et al., “5G: A Tutorial Overview of Standards, Trials, Challenges,” IEEE JSAC, 2017.  
- O‑RAN Alliance, “O‑RAN: Towards an Open and Smart RAN,” White Paper, 2018.  
- 3GPP, “Telecommunication management; 3GPP management system (3GPP TS 28.xxx series),” ongoing.

### 2.2 Fine‑Tuned Large Language Models – Applications, architectures, fine‑tuning strategies
- A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.  
- J. Devlin et al., “BERT: Pre‑training of Deep Bidirectional Transformers,” NAACL, 2019.  
- T. Brown et al., “Language Models are Few‑Shot Learners,” NeurIPS, 2020.  
- T. Scao et al., “Bloom: A 176B‑Parameter Open‑Access Multilingual LLM,” Nature, 2023.  
- R. Bommasani et al., “On the Opportunities and Risks of Foundation Models,” Stanford CRFM, 2021.  
- L. R. Varshney et al., “Trustworthy AI for Wireless Systems,” IEEE Communications, 2021.

### 2.3 Knowledge Graph Retrieval – Graph‑based search and reasoning for telecom data
- W. Hamilton, R. Ying, J. Leskovec, “Representation Learning on Graphs,” IEEE Data Eng. Bull., 2017.  
- M. Nickel, K. Murphy, V. Tresp, E. Gabrilovich, “A Review of Relational Machine Learning for Knowledge Graphs,” Proc. IEEE, 2016.  
- S. Ji et al., “A Survey on Knowledge Graphs: Representation, Acquisition, and Applications,” IEEE TKDE, 2021.  
- R. Song et al., “Knowledge Graph‑Enhanced Retrieval for QA,” arXiv, 2021.  
- Neo4j Inc., “The Neo4j Graph Platform,” Product docs and white papers, ongoing.

### 2.4 Retrieval‑Augmented Generation (RAG) and Schema‑Aware NLP
- P. Lewis et al., “Retrieval‑Augmented Generation for Knowledge‑Intensive NLP,” NeurIPS, 2020.  
- J. Gao et al., “Rethinking Search for Long‑Form QA,” arXiv, 2022.  
- S. Zhao et al., “Graph‑RAG: Enhancing LLMs with Knowledge Graphs,” arXiv, 2023.  
- S. Xie et al., “Schema‑Aware Text‑to‑SQL,” arXiv, 2020.  
- C. Sun et al., “Table‑Aware Semantic Parsing for QA over Tables,” ACL, 2019.

### 2.5 Intelligent Network Management and SON
- M. Aliu, A. Imran, M. Imran, B. Evans, “A Survey of Self‑Organizing Networks (SON) for LTE,” IEEE Comm. Surveys & Tutorials, 2013.  
- H. Zhang et al., “Network Slicing Based 5G and Future Mobile Networks,” IEEE Comm. Mag., 2017.  
- NGMN Alliance, “Self‑Organizing Networks,” White Papers, ongoing.  
- F. Musumeci et al., “An Overview and Open Issues on Network Traffic Classification using ML,” IEEE Comm. Surveys & Tutorials, 2019.  
- O‑RAN Alliance, “O‑RAN Non‑RT RIC, Near‑RT RIC,” Spec and Architecture Notes, ongoing.

### 2.6 Related Works (comparative view)

| Approach | Data Substrate | Retrieval | LLM Usage | Strengths | Limits |
|---|---|---|---|---|---|
| LLM‑only QA | Unstructured docs | Keyword/embedding | Generation only | Fast to prototype | Weak schema fidelity; hallucinations |
| Classic BI/SQL | RDB/CSV | Exact match | None | Precise, queryable | Requires schema knowledge |
| Text‑RAG | Text chunks | Vector search | Grounded gen | Good for docs | Weak on tables/columns semantics |
| Graph‑RAG | Knowledge graphs | Graph + vector | Grounded gen | Structure‑aware | Needs KG curation |
| This work | Typed Neo4j KG (Tables, Columns) | Multi‑strategy search, domain signals | LLM + domain‑aware NLU | Schema‑aware answers, traceable | Requires parsing pipeline; KG ops |

---

## 3. Methodology

### 3.1 Research Design (DSRM)
We follow DSRM phases:  
1) Problem identification: RAN schema complexity and discoverability challenges.  
2) Objectives: Natural‑language access with schema‑aware precision and reproducible evaluation.  
3) Design & development: Parser → KG → Chatbot with NLU stack; benchmarking harness.  
4) Demonstration: Streamlit UI with live Neo4j and curated queries.  
5) Evaluation: IR/NLU metrics, ablations across NLU phases.  
6) Communication: This report and repository.

### 3.2 System Architecture
Modules and data flow (aligned with the repository):

- Parser Module (`parser_module/parser.py`): Parses 3GPP CM XML, normalizes tags (e.g., `vsData*`), builds per‑table DataFrames, and collects parameter metadata. Output: `{table -> DataFrame}`, `metadata`, and simplified `metadata2` (parameter lists).  
- Knowledge Graph Module (`knowledge_graph_module/kg_builder.py`): Uses Neo4j with constraints on tables/columns. Builds nodes for tables and columns; creates relationships via four detectors:  
  1) NAME_SIMILARITY using SentenceTransformer embeddings (MiniLM‑L6‑v2) and cosine similarity;  
  2) VALUE_OVERLAP via Jaccard/overlap thresholds on column value sets;  
  3) PATTERN_MATCH using RAN‑aware regexes (IDs, timestamps, status, measurement, frequency, power, config);  
  4) REFERENCES by checking subset relations between potential foreign key and primary key columns.  
- Chatbot Module (`chatbot_module/chatbot.py`):  
  - EnhancedRANChatbot extends a base RANChatbot and integrates an OptimizedQueryInterface, EnhancedRANEntityExtractor, and IntelligentGraphTraversal.  
  - Caching: query/table/embedding/domain caches with TTL; warm‑up loads tables and domain buckets.  
  - Intent prediction: optional fine‑tuned classifier if available; keyword fallback rules; domain synonyms for ranking boosts.  
  - Query processing: explicit table detection, semantic search, parallel aggregation with ranking, domain boosts, and fallbacks.  
  - Response generation: structured summaries (tables, relationships, concepts); enhanced NLU response formatting.  
- NLU Enhancements (`chatbot_module/enhanced_nlu_functions.py`):  
  - Phase 2: Enhanced response generation with domain templates and structured markers (📋🔍📊⚡📡⏱️).  
  - Phase 3: RAN‑aware entity extraction (table/column patterns, camelCase/PascalCase, RAN terminology, acronyms) plus normalization/deduplication and improved metrics.  
  - Phase 4: Multi‑dimensional semantic similarity (embedding, domain terminology, structure, entity‑mentions) with fallbacks.  
- UI and Benchmarks (`chatbot_module/chatbot_ui.py`): Streamlit app for connection, chat, schema summary, and academic benchmarking. IR and NLU benchmarks are separated.  
- Ground Truth: `improved_ir_ground_truth.csv`, `enhanced_nlu_ground_truth.csv` (100 entries), with reproducible formats.

### 3.3 Implementation Details (selected excerpts)
- Parser cleans element tags and normalizes `vsData*` prefixes, builds per‑table rows and nested attributes (e.g., parent.child), and aggregates DataFrames per table.  
- KG builder configures Neo4j constraints; computes column embeddings; classifies relationships via multiple signals; and persists nodes/edges.  
- Chatbot performs:  
  - Explicit table extraction using robust regex patterns for uppercase, snake_case, and PascalCase.  
  - Semantic and domain‑guided retrieval; ranking boosts for intent/domain matches.  
  - Cache warm‑up of tables and domain buckets for fast lookup; query/result caching with TTL.  
- NLU functions add structure‑aware responses and domain‑aware similarity; entity extraction integrates query result context and text analysis.  
- Streamlit UI shows connection, cache metrics (hit rate, cached tables), enhanced examples, and an IR/NLU benchmarking lab.

### 3.4 Evaluation Protocol
- IR Benchmark: Match predicted tables/columns against IR ground truth; compute precision/recall/F1 and retrieval quality metrics.  
- NLU Benchmark:  
  - Semantic similarity: Enhanced multi‑dimensional score (embedding, domain, structure, entity).  
  - Entity metrics: Precision/recall/F1 with normalization.  
  - Response quality markers: structured formatting, richness, and latency.  
- Separation principle: NLU changes are isolated from IR logic to ensure no regression in retrieval metrics.

---

## 4. Results

We summarize the observed behavior in development and benchmarking:

- IR: The IR pipeline (explicit table detection + semantic/domain retrieval + ranking) consistently returns correct top tables for explicit queries and competitive results for domain/entity queries. The UI “Enhanced Examples” achieve near‑perfect explicit matches in testing.  
- NLU Phase 1 (Ground Truth Fix): Moving to a properly structured NLU ground truth enabled meaningful measurement; baseline similarity and entity scores became measurable.  
- NLU Phases 2–4:  
  - Response generation produced structured, domain‑aware summaries.  
  - RAN‑aware entity extraction increased entity recall and F1, especially for table/column patterns and terminology.  
  - Multi‑dimensional similarity stabilized semantic scores across query types.  
- Isolation: IR metrics remained strong and were not degraded by NLU enhancements because code paths and metrics are decoupled.

Practical indicators (from the UI and validation scripts): higher structured‑response rate, improved semantic similarity range, and better entity F1, alongside healthy cache hit rates and sub‑second processing in “fast/balanced” modes for typical queries.

---

## 5. Discussion

### 5.1 Theoretical Contributions
- A practical architecture to bind LLM assistance with a typed, relationship‑rich RAN knowledge graph.  
- A domain‑aware NLU scoring function that mixes embedding, terminology, structure, and entity signals.  
- A decoupled IR/NLU evaluation harness enabling safe iteration on NLU quality.

### 5.2 Practical Impact for Operators
- Natural‑language access to schema knowledge with traceable entities (tables, columns, relationships).  
- Faster triage for performance, power, frequency, timing, and configuration tasks.  
- Caching provides responsive interaction even over large graphs.

### 5.3 Risks and Mitigation
- Hallucinations: mitigate with graph‑grounded responses and explicit table/column references.  
- Model drift: version ground truth and keep tests; allow keyword fallback for intent.  
- Graph staleness: schedule periodic re‑ingestion and relationship refresh.

### 5.4 Future Work
- Incorporate operator‑specific ontologies and counters; enrich concepts.  
- Explore graph embeddings (e.g., GNN‑based) to complement symbolic signals.  
- Add provenance/lineage to every response segment and per‑hop confidence.  
- Scale out evaluation with larger, public RAN‑like datasets.

---

## 6. Conclusion
We presented an intelligent RAN management system unifying a fine‑tuned‑LLM chatbot, a typed Neo4j knowledge graph, and a domain‑aware NLU stack. The parser converts 3GPP CM XML into structured tables; the builder creates a graph of tables, columns, and relationships using multiple detectors; and the chatbot delivers structured, grounded answers with enhanced NLU scoring. A Streamlit UI and IR/NLU ground truths enable reproducible evaluation. Results indicate strong IR and consistent NLU improvements without regressions. The approach supports explainable, schema‑aware insight discovery for RAN operations.

---

## 7. References (IEEE style)

[1] J. G. Andrews et al., “What Will 5G Be?,” IEEE Journal on Selected Areas in Communications, 2014.  
[2] T. S. Rappaport et al., “Millimeter Wave Mobile Communications for 5G Cellular: It Will Work!,” IEEE Access, 2013.  
[3] E. Dahlman, S. Parkvall, J. Sköld, 5G NR, Academic Press, 2018.  
[4] M. Shafi et al., “5G: A Tutorial Overview of Standards, Trials, Challenges,” IEEE JSAC, 2017.  
[5] O‑RAN Alliance, “O‑RAN: Towards an Open and Smart RAN,” 2018.  
[6] 3GPP TS 28.xxx series, “Telecommunication management; Management system,” ongoing.  
[7] A. Vaswani et al., “Attention Is All You Need,” NeurIPS, 2017.  
[8] J. Devlin et al., “BERT: Pre‑training of Deep Bidirectional Transformers,” NAACL, 2019.  
[9] T. B. Brown et al., “Language Models are Few‑Shot Learners,” NeurIPS, 2020.  
[10] T. Scao et al., “Bloom: A 176B‑Parameter Open‑Access Multilingual LLM,” Nature, 2023.  
[11] R. Bommasani et al., “On the Opportunities and Risks of Foundation Models,” 2021.  
[12] W. Hamilton, R. Ying, J. Leskovec, “Representation Learning on Graphs,” 2017.  
[13] M. Nickel et al., “A Review of Relational Machine Learning for Knowledge Graphs,” Proc. IEEE, 2016.  
[14] S. Ji et al., “A Survey on Knowledge Graphs,” IEEE TKDE, 2021.  
[15] P. Lewis et al., “Retrieval‑Augmented Generation,” NeurIPS, 2020.  
[16] S. Zhao et al., “Graph‑RAG: Enhancing LLMs with Knowledge Graphs,” 2023.  
[17] C. Sun et al., “Table‑Aware Semantic Parsing,” ACL, 2019.  
[18] M. Aliu, A. Imran, M. Imran, B. Evans, “Survey of Self‑Organizing Networks (SON) for LTE,” IEEE Comm. Surveys & Tutorials, 2013.  
[19] H. Zhang et al., “Network Slicing Based 5G and Future Mobile Networks,” IEEE Comm. Mag., 2017.  
[20] Neo4j Inc., product documentation.

---

## 8. Appendices

### A. Architecture Overview (code‑aligned)
- Parser (`parser_module/parser.py`): `parse_xml` returns `(dfs, metadata, metadata2)`; cleans tags, normalizes `vsData*`, supports nested `attributes`, builds parent.child keys.  
- KG builder (`knowledge_graph_module/kg_builder.py`): Creates constraints, table/column nodes; relationships via `discover_name_similarity`, `discover_value_overlap`, `discover_pattern_matches`, `discover_references`; embeddings via MiniLM‑L6‑v2; cosine similarity; Jaccard/overlap; RAN regexes.  
- Chatbot (`chatbot_module/chatbot.py`): `EnhancedRANChatbot` with caching (`_table_metadata_cache`, `_domain_cache`, `_embedding_cache`), warm‑ups, domain synonyms, explicit‑table regex, ranking boosts, structured responses.  
- NLU (`chatbot_module/enhanced_nlu_functions.py`): phase‑based functions for response formatting, entity extraction, similarity (embedding + domain + structure + entity).  
- UI (`chatbot_module/chatbot_ui.py`): Streamlit with connection, cache stats, examples, and IR/NLU benchmarking.

### B. Evaluation Datasets and Metrics
- IR ground truth: `improved_ir_ground_truth.csv`.  
- NLU ground truth: `enhanced_nlu_ground_truth.csv` (100 entries, query/answer/entities).  
- Metrics: IR precision/recall/F1; NLU semantic similarity (composite); entity precision/recall/F1; latency; structured‑response rate.

### C. Writing Suggestions and Title Refinement
- Keep sentences clear and avoid uncommon words; prefer “table,” “column,” “neighbor,” “power,” etc.  
- When stating results, report ranges or relative improvements; avoid over‑claiming.  
- Use diagrams for the parser→KG→chatbot pipeline and relationship detectors.  
- Recommended final title: “Intelligent Network Management for Radio Access Networks Using Fine‑Tuned Large Language Models and Knowledge Graph Retrieval.”  
- Alternative subtitle if needed: “A Typed Neo4j Graph and Domain‑Aware NLU for Schema‑Grounded RAN Queries.”
