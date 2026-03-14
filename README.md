# 🤖 RAG Agent — E-commerce Analytics

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green)](https://langchain.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.1+-orange)](https://langchain-ai.github.io/langgraph/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple)](https://openai.com)

Agente conversacional con LangGraph que responde preguntas de negocio sobre documentos de e-commerce: reportes de ventas, briefs de campaña, análisis RFM, datos de clientes.

## 🎯 Demo en vivo
> [URL Streamlit Cloud]

## 🧩 Problema de negocio

Los gerentes de retail en Perú pasan horas buscando datos en PDFs y reportes dispersos. Este agente permite preguntar en lenguaje natural: "¿cuántos clientes Champions tenemos?", "¿qué canal tuvo mejor ROAS en Black Friday?" y obtener respuesta inmediata con fuente citada.

**Conecta con P1 y P3:**
- Los segmentos RFM del P3 (Customer Segmentation) se documentan y el agente los explica
- Las recomendaciones del P1 (Recommender System) se justifican con datos del negocio
- El agente es el "cerebro conversacional" que une los 3 proyectos

## 🏗 Arquitectura LangGraph

```
query → [retrieve] → [grade_documents] → [generate] → respuesta
                           │
                    (baja confianza)
                           ↓
                    [rewrite_query] → [retrieve] (máx. 2 reintentos)
```

Nodos del grafo:
- **retrieve**: ChromaDB MMR search, k=6 chunks más relevantes
- **grade_documents**: evalúa relevancia de cada chunk con GPT-4o-mini
- **rewrite_query**: reformula la query si la confianza es baja (< 30%)
- **generate**: genera respuesta con contexto + historial de conversación

## 🚀 Cómo ejecutar localmente

```bash
git clone https://github.com/msalazark/rag-agent-ecommerce
cd rag-agent-ecommerce

pip install -r requirements.txt

# Configurar API key
cp .env.example .env
# Editar .env con tu OPENAI_API_KEY

streamlit run app.py
```

## 🛠 Stack

`Python 3.10` · `LangChain 0.2` · `LangGraph` · `LangSmith` · `OpenAI GPT-4o-mini` · `ChromaDB` · `Streamlit`

## 📁 Estructura

```
rag_agent/
├── app.py                    ← Streamlit UI
├── src/
│   ├── agent.py              ← LangGraph agent (retrieve→grade→generate)
│   └── ingestion.py          ← PDF/imagen/texto → ChromaDB
├── docs_sample/              ← Documentos de demo (retail peruano)
│   ├── reporte_ventas_q4_2024.txt
│   └── brief_blackfriday_2024.txt
├── data/vectorstore/         ← ChromaDB persistente (local)
├── .env.example
└── requirements.txt
```

## 🔮 Próximos pasos

- [ ] OCR de imágenes con EasyOCR (facturas, catálogos escaneados)
- [ ] Tool calling: consultar API del P1 (recomendaciones) y P3 (churn score)
- [ ] Deploy en Cloud Run con API FastAPI
- [ ] Multi-tenant: un vectorstore por cliente

---
**Miguel Salazar** · [LinkedIn](https://linkedin.com/in/msalazark) · [GitHub](https://github.com/msalazark)
