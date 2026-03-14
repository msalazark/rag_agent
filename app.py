"""
app.py — RAG Agent E-commerce
Interfaz conversacional sobre documentos de negocio.
LangGraph + OpenAI + ChromaDB + Streamlit
"""
import streamlit as st
import os, sys, time
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

st.set_page_config(
    page_title="RAG Agent · E-commerce",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0f1117; }
[data-testid="stSidebar"] { background: #1a1d27; border-right: 1px solid #2d3748; }
p, label, div, span, h1, h2, h3, li { color: #e2e8f0 !important; }

.msg-user {
    background: #2d3748; border-radius: 12px 12px 4px 12px;
    padding: 12px 16px; margin: 8px 0; max-width: 80%; margin-left: auto;
    border-left: 3px solid #4299e1;
}
.msg-agent {
    background: #1a1d27; border-radius: 12px 12px 12px 4px;
    padding: 12px 16px; margin: 8px 0; max-width: 85%;
    border-left: 3px solid #48bb78; border: 1px solid #2d3748;
}
.source-badge {
    display: inline-block; font-size: 10px; padding: 2px 8px;
    border-radius: 99px; background: rgba(66,153,225,0.15);
    border: 1px solid rgba(66,153,225,0.3); color: #63b3ed !important;
    margin: 2px;
}
.confidence-bar {
    height: 4px; border-radius: 2px; margin-top: 6px;
}
.stat-box {
    background: #1a1d27; border: 1px solid #2d3748;
    border-radius: 8px; padding: 12px; text-align: center;
}
.stat-n { font-size: 22px; font-weight: 700; color: #f7fafc !important; }
.stat-l { font-size: 11px; color: #718096 !important; margin-top: 2px; }
.retry-badge {
    font-size: 10px; color: #f6ad55 !important;
    padding: 1px 6px; border-radius: 99px;
    border: 1px solid rgba(246,173,85,.3);
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    st.session_state.agent = None
if "api_key" not in st.session_state:
    st.session_state.api_key = os.getenv("OPENAI_API_KEY", "")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🤖 RAG Agent Config")
    st.markdown("---")

    # API Key
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=st.session_state.api_key,
        type="password",
        placeholder="sk-...",
    )
    if api_key_input:
        st.session_state.api_key = api_key_input

    model = st.selectbox(
        "Modelo", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0, help="gpt-4o-mini: más rápido y económico"
    )

    # LangSmith (opcional)
    st.markdown("---")
    st.markdown("### 🔍 LangSmith (opcional)")
    ls_key = st.text_input("LangSmith API Key", type="password", placeholder="lsv2_...")
    if ls_key:
        os.environ["LANGCHAIN_API_KEY"]     = ls_key
        os.environ["LANGCHAIN_TRACING_V2"]  = "true"
        os.environ["LANGCHAIN_PROJECT"]     = "rag-agent-ecommerce"
        st.success("Tracing activo", icon="✅")

    # Inicializar agente
    st.markdown("---")
    if st.button("🚀 Inicializar agente", use_container_width=True):
        if not st.session_state.api_key:
            st.error("Ingresa tu OpenAI API Key")
        else:
            with st.spinner("Cargando modelo y vectorstore..."):
                try:
                    from src.agent import RAGAgent
                    st.session_state.agent = RAGAgent(
                        api_key=st.session_state.api_key,
                        model=model,
                    )
                    st.success("Agente listo ✅")
                except Exception as e:
                    st.error(f"Error: {e}")

    # Stats del vectorstore
    if st.session_state.agent:
        stats = st.session_state.agent.get_stats()
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f'<div class="stat-box"><div class="stat-n">{stats["total_chunks"]}</div>'
                f'<div class="stat-l">chunks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(
                f'<div class="stat-box"><div class="stat-n">{len(stats["fuentes"])}</div>'
                f'<div class="stat-l">documentos</div></div>', unsafe_allow_html=True)

        if stats["fuentes"]:
            st.markdown("**Documentos indexados:**")
            for f in stats["fuentes"]:
                st.markdown(f'<span class="source-badge">📄 {f}</span>', unsafe_allow_html=True)

    # Limpiar historial
    if st.button("🗑 Limpiar conversación", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<h1 style="font-size:22px;margin:0;color:#f7fafc">'
    '🤖 RAG Agent — E-commerce Analytics'
    '<span style="font-size:12px;color:#718096;font-weight:400;margin-left:12px">'
    'LangGraph · OpenAI · ChromaDB'
    '</span></h1>',
    unsafe_allow_html=True
)
st.markdown("---")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "📄 Subir documentos", "⚙️ Cómo funciona"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    # Render historial
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.markdown(
                '<div style="text-align:center;padding:40px;color:#4a5568">'
                '📚 Sube documentos y pregunta sobre tu negocio<br>'
                '<small>Reportes de ventas, briefs de campaña, catálogos, análisis RFM...</small>'
                '</div>', unsafe_allow_html=True
            )
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="msg-user">👤 {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                sources_html = "".join([
                    f'<span class="source-badge">📄 {s}</span>'
                    for s in msg.get("sources", [])
                ])
                conf = msg.get("confidence", 0)
                conf_color = "#48bb78" if conf >= 0.6 else ("#f6ad55" if conf >= 0.3 else "#fc8181")
                retry_html = ""
                if msg.get("retries", 0) > 0:
                    retry_html = f'<span class="retry-badge">↺ reformulado {msg["retries"]}x</span>'

                st.markdown(
                    f'<div class="msg-agent">'
                    f'<div>🤖 {msg["content"]}</div>'
                    f'<div style="margin-top:8px">{sources_html} {retry_html}</div>'
                    f'<div class="confidence-bar" style="background:linear-gradient(90deg,{conf_color} {int(conf*100)}%,#2d3748 {int(conf*100)}%)"></div>'
                    f'<div style="font-size:10px;color:#4a5568;margin-top:2px">Confianza: {int(conf*100)}%</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

    # Preguntas sugeridas
    if not st.session_state.messages and st.session_state.agent:
        st.markdown("**Preguntas sugeridas:**")
        suggestions = [
            "¿Cuáles fueron las ventas totales del Q4 2024?",
            "¿Cuántos clientes Champions tenemos y cuánto generan?",
            "¿Qué canal tiene el mejor ROAS?",
            "¿Cuál es la tasa de churn proyectada para el segmento En Riesgo?",
            "¿Qué productos fueron los más vendidos?",
            "¿Cómo fue el rendimiento del Black Friday?",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(sug, key=f"sug_{i}", use_container_width=True):
                st.session_state._quick_question = sug
                st.rerun()

    # Input de pregunta
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_input = st.text_input(
            "Pregunta", label_visibility="collapsed",
            placeholder="¿Qué quieres saber sobre tu negocio?",
            key="chat_input"
        )
    with col_btn:
        send = st.button("Enviar", use_container_width=True)

    # Procesar quick question
    quick = getattr(st.session_state, "_quick_question", None)
    if quick:
        user_input = quick
        del st.session_state._quick_question
        send = True

    if send and user_input:
        if not st.session_state.agent:
            st.warning("Primero inicializa el agente en el sidebar →")
        else:
            # Agregar mensaje del usuario
            st.session_state.messages.append({
                "role": "user", "content": user_input
            })

            # Construir historial para el agente
            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
            ]

            # Llamar al agente con spinner
            with st.spinner("Pensando..."):
                start = time.time()
                result = st.session_state.agent.ask(
                    query=user_input,
                    chat_history=history,
                )
                elapsed = time.time() - start

            # Guardar respuesta
            st.session_state.messages.append({
                "role":       "assistant",
                "content":    result["answer"],
                "sources":    result["sources"],
                "confidence": result["confidence"],
                "retries":    result["retries"],
                "elapsed":    round(elapsed, 1),
            })
            st.rerun()


# ── Tab 2: Subir documentos ───────────────────────────────────────────────────
with tab2:
    st.subheader("📄 Agregar documentos a la base de conocimiento")
    st.caption("Soporta: PDF, TXT, MD, CSV, PNG, JPG (con OCR)")

    uploaded = st.file_uploader(
        "Arrastra o selecciona archivos",
        type=["pdf", "txt", "md", "csv", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    # Documentos de demo precargados
    st.markdown("---")
    st.markdown("**O carga los documentos de demo:**")
    demo_col1, demo_col2 = st.columns(2)

    with demo_col1:
        if st.button("📊 Reporte Ventas Q4 2024", use_container_width=True):
            if not st.session_state.api_key:
                st.warning("Ingresa tu API Key primero")
            else:
                with st.spinner("Indexando..."):
                    try:
                        from src.ingestion import ingest_file
                        demo_path = BASE_DIR / "docs_sample" / "reporte_ventas_q4_2024.txt"
                        result = ingest_file(
                            str(demo_path), st.session_state.api_key,
                            {"tipo_doc": "reporte_ventas", "periodo": "Q4_2024"}
                        )
                        if result["status"] == "ok":
                            st.success(f"✅ {result['chunks']} chunks indexados")
                        else:
                            st.error(result.get("msg"))
                    except Exception as e:
                        st.error(str(e))

    with demo_col2:
        if st.button("🎯 Brief Black Friday 2024", use_container_width=True):
            if not st.session_state.api_key:
                st.warning("Ingresa tu API Key primero")
            else:
                with st.spinner("Indexando..."):
                    try:
                        from src.ingestion import ingest_file
                        demo_path = BASE_DIR / "docs_sample" / "brief_blackfriday_2024.txt"
                        result = ingest_file(
                            str(demo_path), st.session_state.api_key,
                            {"tipo_doc": "brief_campana", "campana": "black_friday_2024"}
                        )
                        if result["status"] == "ok":
                            st.success(f"✅ {result['chunks']} chunks indexados")
                        else:
                            st.error(result.get("msg"))
                    except Exception as e:
                        st.error(str(e))

    # Procesar archivos subidos
    if uploaded and st.session_state.api_key:
        if st.button("📥 Indexar archivos subidos", use_container_width=True):
            from src.ingestion import ingest_file
            progress = st.progress(0)
            for i, file in enumerate(uploaded):
                # Guardar temporalmente
                tmp = BASE_DIR / "data" / "raw" / file.name
                tmp.parent.mkdir(parents=True, exist_ok=True)
                with open(tmp, "wb") as f:
                    f.write(file.read())
                with st.spinner(f"Indexando {file.name}..."):
                    result = ingest_file(str(tmp), st.session_state.api_key)
                    if result["status"] == "ok":
                        st.success(f"✅ {file.name} — {result['chunks']} chunks")
                    else:
                        st.error(f"❌ {file.name}: {result.get('msg')}")
                progress.progress((i + 1) / len(uploaded))
            if st.session_state.agent:
                st.info("Reinicia el agente para que detecte los nuevos documentos.")


# ── Tab 3: Arquitectura ───────────────────────────────────────────────────────
with tab3:
    st.markdown("""
**Arquitectura del agente RAG con LangGraph:**

```
Usuario (query)
      │
      ▼
  [retrieve]          ← ChromaDB MMR search (k=6)
      │
      ▼
  [grade_documents]   ← Evalúa relevancia de cada chunk con GPT-4o-mini
      │
      ├──(relevancia < 30%)──► [rewrite_query] ──► [retrieve] (máx. 2 reintentos)
      │
      └──(relevancia ≥ 30%)──► [generate]
                                    │
                                    ▼
                               Respuesta final
                               + fuentes citadas
                               + % confianza
```

**Stack técnico:**
- **LangGraph**: máquina de estados tipada con routing condicional
- **OpenAI gpt-4o-mini**: LLM para grading, rewriting y generación
- **OpenAI text-embedding-3-small**: embeddings de 1536 dimensiones
- **ChromaDB**: vector store local con persistencia en disco
- **Retrieval MMR**: Maximal Marginal Relevance — evita chunks redundantes
- **LangSmith**: trazabilidad de cada nodo del grafo (opcional)

**Cómo conecta con P1 y P3:**
- Los documentos pueden incluir datos de segmentación RFM del P3
- Las recomendaciones del P1 se documentan y el agente puede explicarlas
- El agente responde preguntas como "¿qué acción tomar con clientes En Riesgo?"
  usando el Action Engine del P3 como base documental
""")

    st.markdown("---")
    st.caption(
        "🤖 RAG Agent E-commerce · Miguel Salazar · "
        "github.com/msalazark/rag-agent-ecommerce"
    )
