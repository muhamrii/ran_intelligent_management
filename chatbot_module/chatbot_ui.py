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
def connect(uri: str, user: str, password: str, enable_model: bool, model_dir: str | None):
    try:
        integrator = RANNeo4jIntegrator(uri, user, password)
        bot = EnhancedRANChatbot(
            integrator,
            use_domain_model=enable_model,
            model_dir=model_dir
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

    cols = st.columns(2)
    with cols[0]:
        if st.button("Connect"):
            ok, msg = connect(uri, user, pwd, model_enabled, model_dir or None)
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
    }
    # Add gentle hint if empty/"No results"
    if not response or response.strip().lower() in {"no results found.", "no matching tables or columns found."}:
        response += "\n\nTip: Try adding a table or column hint (e.g., 'EUtranFrequency' or 'cell_config'), or a domain keyword like 'power' or 'frequency'."
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
            if dbg:
                st.caption(f"Path: {dbg}")


def research_tab():
    st.subheader("🧪 Research Lab")
    tabs = st.tabs(["Intent evaluation", "End-to-end evaluation", "Schema & exports"])

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
                    if rtype == 'semantic_search':
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

                st.markdown("---")
                st.subheader("RAG-style evaluation (optional)")
                st.caption("If you have ground-truth answers, you can compute embedding-only similarity metrics locally. RAGAS support is available without external APIs.")
                gt_csv = st.file_uploader("Ground truth CSV (columns: query,answer)", type=["csv"], key="gt_csv")
                if gt_csv is not None:
                    try:
                        gt_df = pd.read_csv(gt_csv).dropna(subset=['query','answer'])
                        merged = df.merge(gt_df, how='left', on='query', suffixes=('', '_gt'))
                        # Heuristic semantic similarity via simple token overlap as fallback
                        def token_f1(a,b):
                            ta = set(re.findall(r"[a-z0-9_]+", str(a).lower()))
                            tb = set(re.findall(r"[a-z0-9_]+", str(b).lower()))
                            if not ta or not tb:
                                return 0.0
                            tp = len(ta & tb)
                            prec = tp/len(ta)
                            rec = tp/len(tb)
                            return 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
                        merged['sim_token_f1'] = merged.apply(lambda r: token_f1(r.get('answer'), r.get('response', '')), axis=1)
                        st.write(merged[['query','answer','response','sim_token_f1']].head(30))
                        st.json({'Avg token-F1': float(merged['sim_token_f1'].mean())})
                        st.download_button("Download RAG-style eval", merged.to_csv(index=False), file_name="rag_style_eval.csv", mime="text/csv")

                        # RAGAS embedding-only evaluation (best-effort)
                        st.markdown("#### RAGAS (embedding-only)")
                        run_ragas = st.checkbox("Compute RAGAS embedding metrics (no external API)")
                        if run_ragas:
                            try:
                                from datasets import Dataset
                                # Build RAGAS dataset fields
                                ragas_df = merged[['query','response','contexts','answer']].rename(columns={
                                    'query':'question', 'response':'answer', 'answer':'ground_truth'
                                })
                                # Ensure lists for contexts
                                ragas_df['contexts'] = ragas_df['contexts'].apply(lambda x: x if isinstance(x, list) else ([x] if pd.notna(x) else []))
                                ds = Dataset.from_pandas(ragas_df)

                                # Try to import RAGAS metrics that work with embeddings only
                                metrics = []
                                try:
                                    from ragas.metrics import semantic_similarity
                                    metrics.append(semantic_similarity())
                                except Exception:
                                    pass
                                # Prepare local embeddings
                                embedder = None
                                try:
                                    from ragas.embeddings import SentenceTransformerEmbeddings
                                    model_name = st.text_input("Embedding model (HF)", value="sentence-transformers/all-MiniLM-L6-v2")
                                    embedder = SentenceTransformerEmbeddings(model_name)
                                except Exception as e:
                                    st.warning(f"Embeddings backend unavailable: {e}")

                                if metrics and embedder is not None:
                                    from ragas import evaluate
                                    ragas_res = evaluate(ds, metrics=metrics, embeddings=embedder)
                                    st.write(ragas_res)
                                else:
                                    st.info("RAGAS semantic_similarity metric not available or embeddings backend failed. Falling back to local cosine similarity.")
                                    # Local cosine similarity as fallback (uses sentence-transformers if available)
                                    try:
                                        from sentence_transformers import SentenceTransformer
                                        smodel_name = st.text_input("Local ST model (fallback)", value="sentence-transformers/all-MiniLM-L6-v2", key="st_model_fallback")
                                        st_model = SentenceTransformer(smodel_name)
                                        ans_emb = st_model.encode(merged['response'].fillna('').tolist(), normalize_embeddings=True, show_progress_bar=False)
                                        gt_emb = st_model.encode(merged['answer'].fillna('').tolist(), normalize_embeddings=True, show_progress_bar=False)
                                        sims = (ans_emb * gt_emb).sum(axis=1)
                                        merged['sim_cosine'] = sims
                                        st.json({'Avg cosine': float(np.mean(sims))})
                                        st.dataframe(merged[['query','answer','response','sim_token_f1','sim_cosine']].head(30))
                                    except Exception as e:
                                        st.error(f"Local embedding similarity failed: {e}")
                            except Exception as e:
                                st.error(f"RAGAS evaluation failed: {e}")
                    except Exception as e:
                        st.error(f"RAG-style eval failed: {e}")
            except Exception as e:
                st.error(f"E2E failed: {e}")

    # Schema & exports
    with tabs[2]:
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
