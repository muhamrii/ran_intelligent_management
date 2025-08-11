"""
RAN Intelligent Management Chatbot UI (Clean, research-friendly)
"""

import os
import sys
import time
import json
import random
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score

# Ensure project root on path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from knowledge_graph_module.kg_builder import RANNeo4jIntegrator
from chatbot_module.chatbot import EnhancedRANChatbot


# --- Page config ---
st.set_page_config(page_title="RAN Chatbot & Research Lab", page_icon="📡", layout="wide")


# --- Helpers ---
DEFAULT_INTENTS = [
    'performance_analysis','power_optimization','spectrum_management','cell_configuration',
    'quality_assessment','traffic_analysis','fault_detection','capacity_planning',
    'interference_analysis','handover_optimization'
]

def init_state():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = None
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'chat' not in st.session_state:
        st.session_state.chat = []  # list of dict: {q, a, meta}
    if 'model_enabled' not in st.session_state:
        st.session_state.model_enabled = True
    if 'model_dir' not in st.session_state:
        st.session_state.model_dir = ''
    if 'use_caching' not in st.session_state:
        st.session_state.use_caching = True
    if 'performance_mode' not in st.session_state:
        st.session_state.performance_mode = 'balanced'  # 'fast', 'balanced', 'comprehensive'

def discover_model_dir() -> str | None:
    candidates = [
        os.path.join(os.path.dirname(__file__), 'ran_domain_model'),
        os.path.abspath(os.path.join(os.getcwd(), 'chatbot_module', 'ran_domain_model')),
        os.path.abspath(os.path.join(os.getcwd(), 'ran_domain_model')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'ran_domain_model', 'checkpoint-296', '..')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), 'ran_domain_model', 'checkpoint-296')),
    ]
    for d in candidates:
        if d and os.path.isdir(d) and os.path.isfile(os.path.join(d, 'config.json')):
            return d
    return None

def load_intent_labels(model_dir: str | None) -> list[str]:
    try:
        if model_dir:
            p = os.path.join(model_dir, 'intent_labels.json')
            if os.path.isfile(p):
                with open(p, 'r') as f:
                    labels = json.load(f)
                if isinstance(labels, list) and labels:
                    return labels
    except Exception:
        pass
    return DEFAULT_INTENTS

def keyword_intent_predict(text: str) -> str:
    q = text.lower()
    rules = [
        ('performance_analysis', ['throughput','latency','kpi','performance','utilization','efficiency']),
        ('power_optimization', ['power','energy','dbm','consumption','efficiency','sleep']),
        ('spectrum_management', ['frequency','spectrum','bandwidth','carrier','channel','band']),
        ('cell_configuration', ['config','parameter','setting','threshold','cell','antenna','tilt']),
        ('quality_assessment', ['rsrp','rsrq','sinr','quality','noise','interference','cqi','bler']),
        ('traffic_analysis', ['traffic','volume','bytes','packets','usage','load','congestion']),
        ('fault_detection', ['fault','error','alarm','failure','issue','alert']),
        ('capacity_planning', ['capacity','headroom','forecast','plan','scaling','scale']),
        ('interference_analysis', ['interference','overlap','neighbor','pci','collision','noise']),
        ('handover_optimization', ['handover','handoff','mobility','roaming','x2','ng'])
    ]
    for label, kws in rules:
        if any(k in q for k in kws):
            return label
    return 'performance_analysis'


# --- Connection and initialization ---
def connect(uri: str, user: str, password: str, enable_model: bool, model_dir: str | None, use_caching: bool = True):
    try:
        integrator = RANNeo4jIntegrator(uri, user, password)
        bot = EnhancedRANChatbot(
            integrator,
            use_domain_model=enable_model,
            model_dir=model_dir,
            use_caching=use_caching
        )
        st.session_state.chatbot = bot
        st.session_state.connected = True
        return True, "Connected"
    except Exception as e:
        return False, str(e)


# --- UI sections ---
def sidebar():
    st.header("⚙️ Setup")
    uri = st.text_input("Neo4j URI", value="bolt://localhost:7687")
    user = st.text_input("Username", value="neo4j")
    pwd = st.text_input("Password", type="password", value="neo4j")

    st.markdown("---")
    st.subheader("🧠 Domain Model")
    model_enabled = st.checkbox("Use fine-tuned intent model", value=st.session_state.model_enabled)
    auto_dir = discover_model_dir() or ''
    model_dir = st.text_input("Model directory (auto-detected if empty)", value=st.session_state.model_dir or auto_dir)
    
    st.markdown("---")
    st.subheader("🚀 Performance Settings")
    use_caching = st.checkbox("Enable caching", value=st.session_state.use_caching, 
                             help="Cache query results and entity extractions for better performance")
    
    performance_mode = st.selectbox(
        "Performance mode",
        ['fast', 'balanced', 'comprehensive'],
        index=['fast', 'balanced', 'comprehensive'].index(st.session_state.performance_mode),
        help="Fast: Skip expensive processes, Balanced: Selective processing, Comprehensive: Run all processes"
    )
    
    # Update session state
    st.session_state.use_caching = use_caching
    st.session_state.performance_mode = performance_mode

    cols = st.columns(2)
    with cols[0]:
        if st.button("Connect"):
            ok, msg = connect(uri, user, pwd, model_enabled, model_dir or None, use_caching)
            if ok:
                st.session_state.model_enabled = model_enabled
                st.session_state.model_dir = model_dir
                st.success("✅ Connected and chatbot initialized")
                st.rerun()
            else:
                st.error(f"Connection failed: {msg}")
    with cols[1]:
        if st.button("Clear chat"):
            st.session_state.chat = []

    if st.session_state.connected:
        st.info("Connected to Neo4j")
        
        # Show cache statistics if available
        if hasattr(st.session_state.chatbot, 'get_cache_stats'):
            cache_stats = st.session_state.chatbot.get_cache_stats()
            st.metric("Cache Hit Rate", f"{cache_stats.get('hit_rate', 0)}%")
            st.metric("Avg Query Time", f"{cache_stats.get('avg_query_time', 0)}s")
    else:
        st.warning("Not connected")

    st.markdown("---")
    st.subheader("💡 Examples")
    examples = [
        "Show me power domain insights",
        "Find frequency related patterns",
        "What timing synchronization data exists?",
        "Search for performance concepts",
        "Find tables related to neighbor_relations",
        "Get details about cell_config",
    ]
    ex = st.selectbox("Pick an example", [""] + examples)
    return ex


def show_schema_summary():
    try:
        schema = st.session_state.chatbot.query_interface.get_schema_overview()
    except Exception:
        schema = None
    if not schema:
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Tables", schema.get('tables', {}).get('total_tables', 0))
    with c2:
        st.metric("Columns", schema.get('columns', {}).get('total_columns', 0))
    with c3:
        total_rels = sum(r['count'] for r in schema.get('relationships', []))
        st.metric("Relationships", f"{total_rels:,}")
    with c4:
        st.metric("Concepts", len(schema.get('concepts', [])))
    with c5:
        avg_rows = schema.get('tables', {}).get('avg_row_count', 0)
        st.metric("Avg Rows", f"{avg_rows:.0f}" if avg_rows else "N/A")


def handle_chat(query: str):
    bot = st.session_state.chatbot
    t0 = time.time()
    try:
        result = bot.enhanced_process_query(query)
    except AttributeError:
        # fallback to base flow
        result = bot.process_query(query)
    latency = (time.time() - t0) * 1000
    response = result.get('response') or bot.generate_response(result)
    intent, conf = bot.predict_intent(query)
    
    # Enhanced metadata for parallel aggregated results
    meta = {
        'type': result.get('type'),
        'intent': intent,
        'confidence': conf,
        'domain': result.get('domain'),
        'pattern': result.get('pattern_type'),
        'entities': result.get('entities'),
        'debug': result.get('debug'),
        'latency_ms': round(latency, 1),
        'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'cache_hit': result.get('cache_hit', False),
        'query_time': result.get('query_time', latency / 1000)
    }
    
    # Add parallel processing metadata if available
    if result.get('type') == 'parallel_aggregated':
        meta.update({
            'top_tables_count': len(result.get('top_tables', [])),
            'top_columns_count': len(result.get('top_columns', [])),
            'processing_time_ms': result.get('processing_time_ms', latency),
            'processes_executed': len(result.get('parallel_results', {})),
            'parallel_stats': result.get('debug', {}).get('processing_stats', {}),
            'performance_stats': result.get('performance', {})
        })
    
    # Enhanced entity information
    entities = result.get('entities', {})
    if entities:
        meta['extracted_entities'] = {
            'technical_terms': entities.get('technical_terms', []),
            'measurements': entities.get('measurements', []),
            'network_elements': entities.get('network', []),
            'confidence_scores': entities.get('confidence_scores', {})
        }
    
    # Add gentle hint if empty/"No results"
    if not response or response.strip().lower() in {"no results found.", "no matching tables or columns found."}:
        response += "\n\nTip: Try adding a table or column hint (e.g., 'BoundaryOrdinaryClock' or 'AnrFunction'), or a domain keyword like 'power' or 'frequency'."
    
    st.session_state.chat.append({'q': query, 'a': response, 'meta': meta})


def chat_tab(example_prefill: str):
    st.subheader("💬 Chat")
    if not st.session_state.connected:
        st.info("Connect to Neo4j in the sidebar to start.")
        return
    show_schema_summary()
    col1, col2 = st.columns([4,1])
    with col1:
        q = st.text_input("Ask about your RAN data", value=example_prefill or "", placeholder="e.g., Show me power domain insights")
    with col2:
        send = st.button("Send")
    if send and q.strip():
        with st.spinner("Thinking..."):
            handle_chat(q.strip())
        st.rerun()

    for i, item in enumerate(reversed(st.session_state.chat)):
        with st.expander(("You: " + item['q'])[:80], expanded=(i==0)):
            st.markdown(item['a'])
            m = item['meta']
            
            # Enhanced display for parallel aggregated results
            if m.get('type') == 'parallel_aggregated':
                # Main metrics row
                cols = st.columns(6)
                with cols[0]:
                    st.caption(f"Intent: {m.get('intent')}")
                with cols[1]:
                    c = m.get('confidence')
                    st.caption(f"Confidence: {c:.2f}" if isinstance(c,(int,float)) else "Confidence: -")
                with cols[2]:
                    st.caption(f"Type: Parallel Aggregated")
                with cols[3]:
                    st.caption(f"Latency: {m.get('processing_time_ms', m.get('latency_ms'))} ms")
                with cols[4]:
                    st.caption(f"Tables: {m.get('top_tables_count', 0)}")
                with cols[5]:
                    st.caption(f"Columns: {m.get('top_columns_count', 0)}")
                
                # Processing statistics
                if m.get('parallel_stats'):
                    st.caption("**Processing Results:**")
                    stats = m['parallel_stats']
                    success_processes = [k for k, v in stats.items() if v == 'success']
                    if success_processes:
                        st.caption(f"✅ Successful: {', '.join(success_processes)}")
                    
                    error_processes = [k for k, v in stats.items() if 'error' in str(v)]
                    if error_processes:
                        st.caption(f"⚠️ Errors: {', '.join(error_processes)}")
                
                # Enhanced entity extraction display
                if m.get('extracted_entities'):
                    entities = m['extracted_entities']
                    st.caption("**Extracted Entities:**")
                    
                    # Technical terms
                    if entities.get('technical_terms'):
                        st.caption(f"🔧 Technical: {', '.join(entities['technical_terms'][:5])}")
                    
                    # Measurements
                    if entities.get('measurements'):
                        st.caption(f"📊 Measurements: {', '.join(entities['measurements'][:3])}")
                    
                    # Network elements
                    if entities.get('network_elements'):
                        st.caption(f"🌐 Network: {', '.join(entities['network_elements'][:3])}")
                
                # Performance indicators
                if m.get('cache_hit'):
                    st.caption("⚡ Cache Hit - Fast Response")
                elif m.get('query_time', 0) < 0.1:
                    st.caption("🚀 Ultra-fast Response")
                elif m.get('query_time', 0) > 5.0:
                    st.caption("🐌 Slow Response - Consider optimization")
                
            else:
                # Standard display for other result types
                cols = st.columns(5)
                with cols[0]:
                    st.caption(f"Intent: {m.get('intent')}")
                with cols[1]:
                    c = m.get('confidence')
                    st.caption(f"Confidence: {c:.2f}" if isinstance(c,(int,float)) else "Confidence: -")
                with cols[2]:
                    st.caption(f"Type: {m.get('type')}")
                with cols[3]:
                    st.caption(f"Latency: {m.get('latency_ms')} ms")
                with cols[4]:
                    st.caption(m.get('time'))
            
            # Debug path if available
            dbg = item.get('meta', {}).get('debug') or {}
            if dbg and dbg.get('path'):
                st.caption(f"Path: {dbg.get('path')}")
            
            # Show additional debug info for parallel processing
            if m.get('type') == 'parallel_aggregated' and dbg.get('processes_run'):
                with st.expander("🔍 Detailed Processing Info", expanded=False):
                    debug_info = {
                        'processes_executed': dbg.get('processes_run', []),
                        'processing_stats': m.get('parallel_stats', {}),
                        'entities_extracted': bool(m.get('entities')),
                        'domain_identified': bool(m.get('domain')),
                        'cache_performance': {
                            'cache_hit': m.get('cache_hit', False),
                            'query_time': f"{m.get('query_time', 0):.3f}s"
                        }
                    }
                    
                    # Add performance timing if available
                    if m.get('performance_stats'):
                        debug_info['timing_breakdown'] = m['performance_stats']
                    
                    # Add entity confidence scores if available
                    entities = m.get('extracted_entities', {})
                    if entities.get('confidence_scores'):
                        debug_info['entity_confidence'] = entities['confidence_scores']
                    
                    st.json(debug_info)


def research_tab():
    st.subheader("🧪 Research Lab")
    tabs = st.tabs(["Intent evaluation", "End-to-end evaluation", "Academic benchmarking", "Schema & exports"])

    # Intent evaluation
    with tabs[0]:
        st.caption("Compare fine-tuned model vs keyword baseline and standard models on labeled text.")
        colA, colB, colC = st.columns([2,1,1])
        with colA:
            source = st.selectbox("Dataset source", ["Sample from training JSON", "Upload CSV (text,intent)"])
        with colB:
            n = st.number_input("Sample size", 20, 2000, 100, step=20)
        with colC:
            seed = st.number_input("Seed", 0, 10_000, 42, step=1)

        # Standard model comparison options
        st.markdown("#### Standard Model Comparison")
        compare_standard = st.checkbox("Compare with standard pre-trained models")
        standard_models = []
        if compare_standard:
            available_models = [
                "distilbert-base-uncased",
                "bert-base-uncased", 
                "roberta-base",
                "sentence-transformers/all-MiniLM-L6-v2"
            ]
            selected_models = st.multiselect(
                "Select standard models to compare:", 
                available_models, 
                default=["distilbert-base-uncased"]
            )
            standard_models = selected_models

        data_df = None
        if source == "Upload CSV (text,intent)":
            f = st.file_uploader("CSV with columns: text,intent", type=["csv"]) 
            if f is not None:
                try:
                    data_df = pd.read_csv(f)
                    st.success(f"Loaded {len(data_df)} rows")
                except Exception as e:
                    st.error(f"CSV error: {e}")

        if st.button("Run intent evaluation"):
            try:
                labels = load_intent_labels(getattr(st.session_state.chatbot, 'model_dir', None))
                if source == "Sample from training JSON":
                    td_path = os.path.join(os.path.dirname(__file__), 'ran_training_data.json')
                    with open(td_path, 'r') as f:
                        data = json.load(f)
                    pool = [(d.get('text',''), d.get('intent','')) for d in data if d.get('text') and d.get('intent') in labels]
                    random.seed(int(seed))
                    examples = random.sample(pool, k=min(len(pool), int(n)))
                else:
                    if data_df is None or data_df.empty:
                        st.warning("Upload data first.")
                        st.stop()
                    dfc = data_df.dropna(subset=['text','intent'])
                    pool = list(dfc[['text','intent']].itertuples(index=False, name=None))
                    random.seed(int(seed))
                    examples = random.sample(pool, k=min(len(pool), int(n)))

                bot = st.session_state.chatbot
                y_true, y_pred_m, y_pred_kw, confs = [], [], [], []
                standard_predictions = {model: [] for model in standard_models}
                
                # Standard model setup if requested
                standard_classifiers = {}
                if standard_models:
                    try:
                        from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                        for model_name in standard_models:
                            try:
                                if "sentence-transformers" in model_name:
                                    # For sentence transformers, we'll use similarity-based classification
                                    from sentence_transformers import SentenceTransformer
                                    standard_classifiers[model_name] = SentenceTransformer(model_name)
                                else:
                                    # For BERT-like models, use zero-shot classification
                                    classifier = pipeline("zero-shot-classification", model=model_name)
                                    standard_classifiers[model_name] = classifier
                                st.info(f"Loaded {model_name}")
                            except Exception as e:
                                st.warning(f"Failed to load {model_name}: {e}")
                    except ImportError:
                        st.error("transformers library required for standard model comparison")
                        standard_models = []

                for text, y in examples:
                    y_true.append(y)
                    m_label, m_conf = bot.predict_intent(text)
                    y_pred_m.append(m_label)
                    y_pred_kw.append(keyword_intent_predict(text))
                    confs.append(m_conf if m_conf is not None else np.nan)
                    
                    # Standard model predictions
                    for model_name in standard_models:
                        if model_name in standard_classifiers:
                            try:
                                classifier = standard_classifiers[model_name]
                                if "sentence-transformers" in model_name:
                                    # Similarity-based classification
                                    text_emb = classifier.encode([text])
                                    intent_embs = classifier.encode(labels)
                                    similarities = (text_emb @ intent_embs.T)[0]
                                    predicted_intent = labels[np.argmax(similarities)]
                                else:
                                    # Zero-shot classification
                                    result = classifier(text, labels)
                                    predicted_intent = result['labels'][0]
                                standard_predictions[model_name].append(predicted_intent)
                            except Exception as e:
                                standard_predictions[model_name].append('performance_analysis')  # fallback
                        else:
                            standard_predictions[model_name].append('performance_analysis')  # fallback

                def metrics(y_t, y_p):
                    return {
                        'accuracy': accuracy_score(y_t, y_p),
                        'macro_f1': f1_score(y_t, y_p, average='macro', labels=labels, zero_division=0)
                    }

                # Compute metrics for all models
                results_data = []
                m1 = metrics(y_true, y_pred_m)
                results_data.append({'method':'Fine-tuned model', **m1})
                
                m2 = metrics(y_true, y_pred_kw)
                results_data.append({'method':'Keyword baseline', **m2})
                
                for model_name in standard_models:
                    if model_name in standard_predictions:
                        m_std = metrics(y_true, standard_predictions[model_name])
                        results_data.append({'method':f'Standard: {model_name.split("/")[-1]}', **m_std})

                st.write(pd.DataFrame(results_data).set_index('method'))

                # Confusion matrices
                num_models = 2 + len(standard_models)
                cols_per_row = 3
                rows = (num_models + cols_per_row - 1) // cols_per_row
                
                fig, axes = plt.subplots(rows, min(cols_per_row, num_models), figsize=(5*min(cols_per_row, num_models), 4*rows))
                if rows == 1:
                    axes = axes if num_models > 1 else [axes]
                else:
                    axes = axes.flatten()

                # Fine-tuned model confusion matrix
                cm1 = confusion_matrix(y_true, y_pred_m, labels=labels)
                sns.heatmap(cm1, ax=axes[0], cmap='Blues', xticklabels=labels, yticklabels=labels)
                axes[0].set_title('Fine-tuned Model')
                axes[0].tick_params(axis='x', rotation=45)
                
                # Keyword baseline confusion matrix  
                cm2 = confusion_matrix(y_true, y_pred_kw, labels=labels)
                sns.heatmap(cm2, ax=axes[1], cmap='Oranges', xticklabels=labels, yticklabels=labels)
                axes[1].set_title('Keyword Baseline')
                axes[1].tick_params(axis='x', rotation=45)
                
                # Standard models confusion matrices
                for i, model_name in enumerate(standard_models):
                    if model_name in standard_predictions and i+2 < len(axes):
                        cm_std = confusion_matrix(y_true, standard_predictions[model_name], labels=labels)
                        sns.heatmap(cm_std, ax=axes[i+2], cmap='Greens', xticklabels=labels, yticklabels=labels)
                        axes[i+2].set_title(f'Standard: {model_name.split("/")[-1]}')
                        axes[i+2].tick_params(axis='x', rotation=45)
                
                # Hide unused subplots
                for i in range(num_models, len(axes)):
                    axes[i].set_visible(False)
                    
                plt.tight_layout()
                st.pyplot(fig)

                # Detailed results dataframe
                result_columns = {
                    'text': [t for t,_ in examples],
                    'true_intent': y_true,
                    'model_intent': y_pred_m,
                    'model_conf': confs,
                    'keyword_intent': y_pred_kw,
                }
                
                for model_name in standard_models:
                    if model_name in standard_predictions:
                        col_name = f'std_{model_name.split("/")[-1]}_intent'
                        result_columns[col_name] = standard_predictions[model_name]
                
                out = pd.DataFrame(result_columns)
                st.dataframe(out.head(50), use_container_width=True)
                st.download_button("Download CSV", out.to_csv(index=False), file_name="intent_eval_comparison.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    # E2E evaluation
    with tabs[1]:
        st.caption("Measure success, entity extraction and latency across queries. Optionally compare with RAG-style metrics.")
        default_queries = [
            "Show power data from ConsumedEnergyMeasurement.consumedEnergyMeasurementId",
            "Get network topology from SubscriberGroupProfile.cellTriggerList",
            "Find signal quality from RadioBearerTable.radioBearerTableId",
            "Show frequency data from EUtranFrequency.freqBand",
            "Display timing data from NRSynchronization.nRSynchronizationId",
            "Get handover data from CellSleepFunction.wakeUpLastHoTime",
        ]
        preset = st.radio("Query set", ["Training-style defaults", "Custom list"], horizontal=True)
        custom = st.text_area("Custom queries (one per line)", value="\n".join(default_queries) if preset=="Custom list" else "")
        if st.button("Run E2E evaluation"):
            try:
                qs = default_queries if preset == "Training-style defaults" else [q.strip() for q in custom.splitlines() if q.strip()]
                if not qs:
                    st.warning("Provide some queries.")
                    st.stop()
                bot = st.session_state.chatbot
                rows = []
                # small helper to generate pseudo-contexts from KG results
                def contexts_from_res(res: dict) -> list[str]:
                    ctx: list[str] = []
                    rtype = res.get('type')
                    
                    # Handle new parallel aggregated results
                    if rtype == 'parallel_aggregated':
                        # Extract table names from top_tables
                        for table_info in (res.get('top_tables') or [])[:10]:
                            table_name = table_info.get('table_name')
                            if table_name:
                                sources = ', '.join(table_info.get('sources', [])[:3])
                                ctx.append(f"Table {table_name} (sources: {sources})")
                        
                        # Extract columns from top_columns
                        for col_info in (res.get('top_columns') or [])[:5]:
                            col_name = col_info.get('column_name')
                            tables = ', '.join(list(col_info.get('tables', []))[:2])
                            if col_name and tables:
                                ctx.append(f"Column {col_name} (in: {tables})")
                    
                    elif rtype == 'semantic_search':
                        for r in (res.get('results') or [])[:3]:
                            cols = r.get('columns') or r.get('top_columns') or []
                            ctx.append(f"Table {r.get('table_name')} cols: {', '.join(cols[:10])}")
                    elif rtype == 'domain_inquiry':
                        tables = (res.get('results') or {}).get('related_tables') or []
                        for t in tables[:3]:
                            cols = t.get('matching_columns') or []
                            ctx.append(f"Table {t.get('table_name')} cols: {', '.join(cols[:10])}")
                    elif rtype == 'table_details':
                        d = res.get('results') or {}
                        cols = [f"{c.get('name')}({c.get('data_type','')})" for c in (d.get('columns') or [])[:10]]
                        if d.get('table_name') or cols:
                            ctx.append(f"Table {d.get('table_name')} cols: {', '.join(cols)}")
                    elif rtype == 'concept_search':
                        for r in (res.get('results') or [])[:3]:
                            tnames = r.get('tables') or []
                            ctx.append(f"Concept {r.get('concept') or r.get('concept_name')}: tables {', '.join(tnames[:5])}")
                    elif rtype == 'multi_hop_relationships':
                        for r in (res.get('results') or [])[:3]:
                            ctx.append(f"Related table {r.get('related_table')} path {r.get('shortest_path')} strength {r.get('avg_relationship_strength')}")
                    elif rtype == 'semantic_clustering':
                        for r in (res.get('results') or [])[:3]:
                            ctx.append(f"Cluster {r.get('cluster_name')} tables {', '.join((r.get('sample_tables') or [])[:5])}")
                    
                    # Fallback: use generated response itself
                    if not ctx:
                        resp = res.get('response')
                        if resp:
                            ctx = [resp[:500]]
                    return ctx
                for q in qs:
                    t0 = time.time()
                    try:
                        res = bot.enhanced_process_query(q)
                    except AttributeError:
                        res = bot.process_query(q)
                    dt = (time.time() - t0) * 1000
                    resp = res.get('response') or bot.generate_response(res)
                    intent, _ = bot.predict_intent(q)
                    ents = res.get('entities', {})
                    rows.append({
                        'query': q,
                        'intent': intent,
                        'has_entities': bool(ents) and any(bool(v) for v in ents.values()),
                        'has_response': bool(resp),
                        'resp_len': len(resp or ''),
                        'latency_ms': dt,
                        'type': res.get('type'),
                        'response': resp,
                        'contexts': contexts_from_res(res)
                    })
                df = pd.DataFrame(rows)
                kpis = {
                    'Success rate (%)': 100*df['has_response'].mean(),
                    'Entity rate (%)': 100*df['has_entities'].mean(),
                    'Avg response length': df['resp_len'].mean(),
                    'P95 latency (ms)': float(np.percentile(df['latency_ms'], 95))
                }
                st.json({k:(round(v,1) if isinstance(v,(int,float,np.floating)) else v) for k,v in kpis.items()})
                fig, ax = plt.subplots(1,2, figsize=(12,4))
                sns.histplot(df['latency_ms'], bins=10, ax=ax[0], color='skyblue')
                ax[0].set_title('Latency (ms)')
                sns.countplot(x='type', data=df, ax=ax[1])
                ax[1].set_title('Query types')
                plt.tight_layout(); st.pyplot(fig)
                st.dataframe(df.drop(columns=['contexts']) if 'contexts' in df else df, use_container_width=True)
                st.download_button("Download CSV", df.to_csv(index=False), file_name="e2e_eval.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Evaluation failed: {e}")

    # Academic benchmarking (separate tab)
    with tabs[2]:
        st.caption("Comprehensive IR and NLU evaluation using standard academic metrics for richer insights.")
        
        # Load sample data paths
        sample_ir_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_ir_ground_truth.csv')
        sample_nlu_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sample_nlu_ground_truth.csv')
        
        # Check if sample files exist
        has_sample_ir = os.path.exists(sample_ir_path)
        has_sample_nlu = os.path.exists(sample_nlu_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Information Retrieval (IR) Evaluation")
            use_sample_ir = st.checkbox("Use generated sample IR data", value=has_sample_ir, disabled=not has_sample_ir)
            if not use_sample_ir:
                ir_csv = st.file_uploader("IR Ground Truth CSV (columns: query,relevant_tables)", type=["csv"], key="ir_csv")
            else:
                ir_csv = sample_ir_path
                if has_sample_ir:
                    st.success("✅ Using generated sample IR data")
                else:
                    st.warning("Sample IR data not found - run data generation notebook first")
        
        with col2:
            st.markdown("#### Natural Language Understanding (NLU) Evaluation")
            use_sample_nlu = st.checkbox("Use generated sample NLU data", value=has_sample_nlu, disabled=not has_sample_nlu)
            if not use_sample_nlu:
                nlu_csv = st.file_uploader("NLU Ground Truth CSV (columns: query,answer,entities)", type=["csv"], key="nlu_csv")
            else:
                nlu_csv = sample_nlu_path
                if has_sample_nlu:
                    st.success("✅ Using generated sample NLU data")
                else:
                    st.warning("Sample NLU data not found - run data generation notebook first")
        
        if st.button("Run Academic Benchmarks", type="primary"):
            benchmark_results = {}
            
            # === IR BENCHMARKS ===
            if ir_csv is not None:
                try:
                    if isinstance(ir_csv, str):
                        ir_df = pd.read_csv(ir_csv).dropna(subset=['query','relevant_tables'])
                    else:
                        ir_df = pd.read_csv(ir_csv).dropna(subset=['query','relevant_tables'])
                    
                    st.markdown("##### IR Benchmark Results")
                    
                    def parse_relevant_tables(tables_str):
                        """Parse comma-separated table names"""
                        if pd.isna(tables_str):
                            return []
                        return [t.strip() for t in str(tables_str).split(',') if t.strip()]
                    
                    def compute_ir_metrics(retrieved_tables, relevant_tables, k_values=[1,3,5,10]):
                        """Compute IR metrics including MAP@K using ranked list order."""
                        metrics = {}
                        if not relevant_tables:
                            base = {f'P@{k}':0.0 for k in k_values} | {f'R@{k}':0.0 for k in k_values}
                            base.update({'MAP':0.0,'MRR':0.0})
                            for k in k_values:
                                base[f'MAP@{k}']=0.0
                            return base
                        relevant_set = set(relevant_tables)
                        # Precision/Recall@K
                        for k in k_values:
                            seg = retrieved_tables[:k]
                            if seg:
                                rel_k = sum(1 for t in seg if t in relevant_set)
                                metrics[f'P@{k}']= rel_k/len(seg)
                                metrics[f'R@{k}']= rel_k/len(relevant_set)
                            else:
                                metrics[f'P@{k}']=0.0; metrics[f'R@{k}']=0.0
                        # MAP full
                        ap_sum=0.0; rel_found=0
                        for i,t in enumerate(retrieved_tables,1):
                            if t in relevant_set:
                                rel_found+=1
                                ap_sum += rel_found / i
                        metrics['MAP']= ap_sum/len(relevant_set) if relevant_set else 0.0
                        # MAP@K variants
                        for k in k_values:
                            ap_k=0.0; rel_k_found=0
                            for i,t in enumerate(retrieved_tables[:k],1):
                                if t in relevant_set:
                                    rel_k_found+=1
                                    ap_k += rel_k_found / i
                            denom = min(len(relevant_set), k)
                            metrics[f'MAP@{k}']= ap_k/denom if denom>0 else 0.0
                        # MRR
                        mrr=0.0
                        for i,t in enumerate(retrieved_tables,1):
                            if t in relevant_set:
                                mrr = 1.0/i
                                break
                        metrics['MRR']=mrr
                        return metrics
                    
                    # Process queries first
                    if not st.session_state.connected:
                        st.error("Please connect to Neo4j first")
                        st.stop()
                    
                    bot = st.session_state.chatbot
                    progress_bar = st.progress(0)
                    
                    # Process all queries to get contexts
                    query_contexts = {}
                    for i, query in enumerate(ir_df['query'].unique()):
                        try:
                            res = bot.enhanced_process_query(query)
                        except AttributeError:
                            res = bot.process_query(query)
                        
                        # Extract table names depending on result type
                        contexts = []
                        rtype = res.get('type')
                        if rtype == 'parallel_aggregated':
                            # Take ranked top tables directly
                            for t in res.get('top_tables', [])[:15]:
                                tname = t.get('table_name') if isinstance(t, dict) else None
                                if tname:
                                    contexts.append(tname)
                            # Fallback to key_results domain tables if any
                            dom_insights = res.get('key_results', {}).get('domain_insights', {})
                            for t in dom_insights.get('related_tables', [])[:5]:
                                tn = t.get('table_name')
                                if tn:
                                    contexts.append(tn)
                        elif rtype == 'semantic_search':
                            for r in (res.get('results') or [])[:10]:
                                table_name = r.get('table_name')
                                if table_name:
                                    contexts.append(table_name)
                        elif rtype == 'domain_inquiry':
                            tables = (res.get('results') or {}).get('related_tables') or []
                            for t in tables[:10]:
                                table_name = t.get('table_name')
                                if table_name:
                                    contexts.append(table_name)
                        elif rtype == 'table_details':
                            d = res.get('results') or {}
                            table_name = d.get('table_name')
                            if table_name:
                                contexts.append(table_name)
                        elif rtype == 'concept_search':
                            for r in (res.get('results') or [])[:10]:
                                tnames = r.get('tables') or []
                                contexts.extend(tnames[:5])
                        query_contexts[query] = list(dict.fromkeys(contexts))  # preserve order, dedupe
                        progress_bar.progress((i + 1) / len(ir_df['query'].unique()))
                    
                    # Compute IR metrics for each query
                    ir_results = []
                    for _, row in ir_df.iterrows():
                        query = row['query']
                        relevant_tables = parse_relevant_tables(row['relevant_tables'])
                        retrieved_tables = query_contexts.get(query, [])
                        
                        metrics = compute_ir_metrics(retrieved_tables, relevant_tables)
                        metrics['query'] = query
                        metrics['relevant_count'] = len(relevant_tables)
                        metrics['retrieved_count'] = len(retrieved_tables)
                        ir_results.append(metrics)
                    
                    if ir_results:
                        ir_results_df = pd.DataFrame(ir_results)
                        
                        # Aggregate metrics (include MAP@K)
                        agg_metrics = {}
                        metric_cols = [col for col in ir_results_df.columns if col not in ['query', 'relevant_count', 'retrieved_count']]
                        for col in metric_cols:
                            agg_metrics[f'Avg_{col}'] = ir_results_df[col].mean()
                        
                        st.json(agg_metrics)
                        
                        # Visualizations
                        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
                        
                        # Precision@K
                        k_values = [1, 3, 5, 10]
                        p_at_k = [agg_metrics.get(f'Avg_P@{k}', 0) for k in k_values]
                        axes[0,0].bar(k_values, p_at_k, color='skyblue')
                        axes[0,0].set_title('Precision@K')
                        axes[0,0].set_xlabel('K')
                        axes[0,0].set_ylabel('Precision')
                        
                        # Recall@K
                        r_at_k = [agg_metrics.get(f'Avg_R@{k}', 0) for k in k_values]
                        axes[0,1].bar(k_values, r_at_k, color='lightcoral')
                        axes[0,1].set_title('Recall@K')
                        axes[0,1].set_xlabel('K')
                        axes[0,1].set_ylabel('Recall')
                        
                        # MAP and MRR comparison
                        map_mrr = [agg_metrics.get('Avg_MAP', 0), agg_metrics.get('Avg_MRR', 0)]
                        axes[1,0].bar(['MAP', 'MRR'], map_mrr, color=['lightgreen', 'orange'])
                        axes[1,0].set_title('MAP vs MRR')
                        axes[1,0].set_ylabel('Score')
                        
                        # Query-level performance distribution
                        axes[1,1].hist(ir_results_df['MAP'], bins=10, alpha=0.7, color='purple')
                        axes[1,1].set_title('MAP Score Distribution')
                        axes[1,1].set_xlabel('MAP Score')
                        axes[1,1].set_ylabel('Frequency')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(ir_results_df, use_container_width=True)
                        benchmark_results['IR'] = agg_metrics
                
                except Exception as e:
                    st.error(f"IR evaluation failed: {e}")
            
            # === NLU BENCHMARKS ===
            if nlu_csv is not None:
                try:
                    if isinstance(nlu_csv, str):
                        nlu_df = pd.read_csv(nlu_csv).dropna(subset=['query'])
                    else:
                        nlu_df = pd.read_csv(nlu_csv).dropna(subset=['query'])
                    
                    st.markdown("##### NLU Benchmark Results")
                    
                    def get_embedding_model():
                        if 'embedding_model' not in st.session_state:
                            from sentence_transformers import SentenceTransformer
                            st.session_state.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
                        return st.session_state.embedding_model
                    def compute_semantic_similarity(text1, text2):
                        try:
                            model = get_embedding_model()
                            emb = model.encode([str(text1), str(text2)])
                            similarity = (emb[0] @ emb[1].T) / (np.linalg.norm(emb[0])*np.linalg.norm(emb[1])+1e-9)
                            return float(similarity)
                        except Exception:
                            tokens1 = set(re.findall(r"[a-z0-9_]+", str(text1).lower()))
                            tokens2 = set(re.findall(r"[a-z0-9_]+", str(text2).lower()))
                            if not tokens1 or not tokens2:
                                return 0.0
                            return len(tokens1 & tokens2)/len(tokens1|tokens2)
                    
                    def extract_entities_from_text(text):
                        """Simple entity extraction"""
                        entities = []
                        # Table.column patterns
                        entities.extend(re.findall(r'([A-Za-z][A-Za-z0-9_]*)\.[A-Za-z][A-Za-z0-9_]*', text))
                        # Identifiers with underscores
                        entities.extend(re.findall(r'[A-Za-z][A-Za-z0-9_]{3,}', text))
                        return list(set(entities))
                    
                    if not st.session_state.connected:
                        st.error("Please connect to Neo4j first")
                        st.stop()
                    
                    bot = st.session_state.chatbot
                    progress_bar = st.progress(0)
                    
                    nlu_results = []
                    for i, (_, row) in enumerate(nlu_df.iterrows()):
                        query = row['query']
                        gt_answer = row.get('answer', '')
                        gt_entities = row.get('entities', '')
                        
                        # Process query
                        try:
                            res = bot.enhanced_process_query(query)
                        except AttributeError:
                            res = bot.process_query(query)
                        
                        response = res.get('response') or bot.generate_response(res)
                        intent, _ = bot.predict_intent(query)
                        
                        # Semantic similarity
                        sem_sim = compute_semantic_similarity(response, gt_answer) if gt_answer else 0.0
                        
                        # Entity F1
                        pred_entities = extract_entities_from_text(response)
                        gt_entities_list = [e.strip() for e in str(gt_entities).split(',') if e.strip()] if gt_entities else []
                        # Normalize entities for fair comparison (case, symbols)
                        def norm(x: str) -> str:
                            x = x.strip().lower()
                            x = re.sub(r'[^a-z0-9_]+', '_', x)
                            x = re.sub(r'_+', '_', x)
                            return x.strip('_')
                        pred_norm = {norm(e) for e in pred_entities if e}
                        gt_norm = {norm(e) for e in gt_entities_list if e}
                        
                        if gt_norm:
                            tp = len(pred_norm & gt_norm)
                            fp = len(pred_norm - gt_norm)
                            fn = len(gt_norm - pred_norm)
                            
                            entity_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                            entity_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                            entity_f1 = 2 * entity_precision * entity_recall / (entity_precision + entity_recall) if (entity_precision + entity_recall) > 0 else 0.0
                        else:
                            entity_precision = entity_recall = entity_f1 = 0.0
                        
                        # Response quality metrics
                        response_length = len(response.split())
                        has_structured_response = any(marker in response for marker in ['📋', '🔍', '⚡', '📊'])
                        
                        nlu_results.append({
                            'query': query,
                            'intent': intent,
                            'semantic_similarity': sem_sim,
                            'entity_precision': entity_precision,
                            'entity_recall': entity_recall,
                            'entity_f1': entity_f1,
                            'response_length': response_length,
                            'structured_response': has_structured_response,
                            'predicted_entities': len(pred_entities),
                            'ground_truth_entities': len(gt_entities_list)
                        })
                        
                        progress_bar.progress((i + 1) / len(nlu_df))
                    
                    if nlu_results:
                        nlu_results_df = pd.DataFrame(nlu_results)
                        
                        # Aggregate NLU metrics
                        nlu_agg = {
                            'Avg_Semantic_Similarity': nlu_results_df['semantic_similarity'].mean(),
                            'Avg_Entity_Precision': nlu_results_df['entity_precision'].mean(),
                            'Avg_Entity_Recall': nlu_results_df['entity_recall'].mean(),
                            'Avg_Entity_F1': nlu_results_df['entity_f1'].mean(),
                            'Avg_Response_Length': nlu_results_df['response_length'].mean(),
                            'Structured_Response_Rate': nlu_results_df['structured_response'].mean()
                        }
                        
                        st.json(nlu_agg)
                        
                        # NLU Visualizations
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                        
                        # Semantic similarity distribution
                        axes[0,0].hist(nlu_results_df['semantic_similarity'], bins=15, alpha=0.7, color='blue')
                        axes[0,0].set_title('Semantic Similarity Distribution')
                        axes[0,0].set_xlabel('Similarity Score')
                        
                        # Entity F1 scores
                        axes[0,1].hist(nlu_results_df['entity_f1'], bins=10, alpha=0.7, color='green')
                        axes[0,1].set_title('Entity F1 Score Distribution')
                        axes[0,1].set_xlabel('F1 Score')
                        
                        # Intent distribution
                        intent_counts = nlu_results_df['intent'].value_counts()
                        axes[0,2].pie(intent_counts.values, labels=intent_counts.index, autopct='%1.1f%%')
                        axes[0,2].set_title('Intent Distribution')
                        
                        # Response length vs semantic similarity
                        axes[1,0].scatter(nlu_results_df['response_length'], nlu_results_df['semantic_similarity'], alpha=0.6)
                        axes[1,0].set_xlabel('Response Length')
                        axes[1,0].set_ylabel('Semantic Similarity')
                        axes[1,0].set_title('Response Length vs Similarity')
                        
                        # Entity metrics comparison
                        entity_metrics = ['entity_precision', 'entity_recall', 'entity_f1']
                        entity_scores = [nlu_results_df[metric].mean() for metric in entity_metrics]
                        axes[1,1].bar(['Precision', 'Recall', 'F1'], entity_scores, color=['red', 'blue', 'green'])
                        axes[1,1].set_title('Entity Extraction Performance')
                        axes[1,1].set_ylabel('Score')
                        
                        # Performance by intent
                        intent_perf = nlu_results_df.groupby('intent')['semantic_similarity'].mean().sort_values(ascending=False)
                        axes[1,2].barh(range(len(intent_perf)), intent_perf.values)
                        axes[1,2].set_yticks(range(len(intent_perf)))
                        axes[1,2].set_yticklabels([intent.replace('_', '\n') for intent in intent_perf.index])
                        axes[1,2].set_title('Semantic Similarity by Intent')
                        axes[1,2].set_xlabel('Avg Similarity')
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        st.dataframe(nlu_results_df, use_container_width=True)
                        benchmark_results['NLU'] = nlu_agg
                
                except Exception as e:
                    st.error(f"NLU evaluation failed: {e}")
            
            # === COMPREHENSIVE REPORT ===
            if benchmark_results:
                st.markdown("##### Comprehensive Benchmark Report")
                
                report_data = []
                if 'IR' in benchmark_results:
                    for metric, value in benchmark_results['IR'].items():
                        report_data.append({'Category': 'Information Retrieval', 'Metric': metric, 'Score': round(value, 4)})
                
                if 'NLU' in benchmark_results:
                    for metric, value in benchmark_results['NLU'].items():
                        report_data.append({'Category': 'Natural Lang Understanding', 'Metric': metric, 'Score': round(value, 4)})
                
                if report_data:
                    report_df = pd.DataFrame(report_data)
                    st.dataframe(report_df, use_container_width=True)
                    
                    # Download comprehensive results
                    all_results = {
                        'summary': benchmark_results,
                        'timestamp': pd.Timestamp.now().isoformat()
                    }
                    st.download_button(
                        "Download Comprehensive Benchmark Results", 
                        json.dumps(all_results, indent=2), 
                        file_name="academic_benchmark_results.json", 
                        mime="application/json"
                    )

    # Schema & exports
    with tabs[3]:
        if not st.session_state.connected:
            st.info("Connect to view schema and export options.")
        else:
            show_schema_summary()
            c1, c2, c3 = st.columns(3)
            with c1:
                if st.button("Export chat history"):
                    if st.session_state.chat:
                        out = pd.DataFrame([{**{'query':x['q'],'response':x['a']}, **x['meta']} for x in st.session_state.chat])
                        st.download_button("Download CSV", out.to_csv(index=False), file_name="chat_history.csv", mime="text/csv")
            with c2:
                if st.button("Export schema"):
                    schema = st.session_state.chatbot.query_interface.get_schema_overview()
                    st.download_button("Download JSON", json.dumps(schema), file_name="schema.json", mime="application/json")
            with c3:
                if st.button("Generate NER samples"):
                    try:
                        samples = st.session_state.chatbot.ner_generator.generate_ner_training_data()
                        st.success(f"Generated {len(samples)} samples")
                        if samples:
                            st.code(f"Sample: {samples[0]}")
                    except Exception as e:
                        st.error(f"NER generation error: {e}")


def main():
    init_state()
    st.title("📡 RAN Intelligent Management Chatbot")
    st.caption("Simple to use. Research-ready. No external APIs.")
    example = sidebar()
    tabs = st.tabs(["Chat", "Research Lab"])
    with tabs[0]:
        chat_tab(example)
    with tabs[1]:
        research_tab()


if __name__ == "__main__":
    main()
