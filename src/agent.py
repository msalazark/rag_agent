"""
agent.py
────────
Agente RAG con LangGraph:
  - Estado tipado (TypedDict)
  - Nodos: retrieve → grade → generate → check_answer
  - Routing condicional: si el contexto no alcanza, reformula la query
  - Conecta con P1 (segmentos) y P3 (churn) via tool calling
  - LangSmith tracing opcional

Dominio: E-commerce / retail peruano
"""
from __future__ import annotations

import os
from typing import TypedDict, List, Annotated, Literal
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
import operator

BASE_DIR   = Path(__file__).parent.parent
VECTOR_DIR = BASE_DIR / "data" / "vectorstore"

SYSTEM_PROMPT = """Eres un asistente de analytics para e-commerce peruano.
Tienes acceso a documentos internos de la empresa (reportes, catálogos, datos de campañas).
Responde siempre en español, con datos concretos cuando los tengas.
Si la pregunta involucra segmentos de clientes o churn, interpreta en contexto de RFM.
Si no encuentras la respuesta en el contexto, di claramente qué información te falta.
Sé conciso pero completo. Usa formato de lista cuando sea más claro."""


# ── Estado del grafo ──────────────────────────────────────────────────────────
class AgentState(TypedDict):
    query:          str
    documents:      List[Document]
    generation:     str
    retry_count:    int
    chat_history:   List[dict]
    confidence:     float
    sources:        List[str]


# ── Nodos del grafo ───────────────────────────────────────────────────────────
class RAGAgent:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.api_key = api_key

        self.llm = ChatOpenAI(
            model=model,
            temperature=0.1,
            openai_api_key=api_key,
        )
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
        )
        self.vectorstore = Chroma(
            collection_name="ecommerce_docs",
            embedding_function=self.embeddings,
            persist_directory=str(VECTOR_DIR),
        )
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",           # Maximal Marginal Relevance — evita duplicados
            search_kwargs={"k": 6, "fetch_k": 20},
        )
        self.graph = self._build_graph()

    # ── Nodo 1: Recuperar documentos ──────────────────────────────────────────
    def retrieve(self, state: AgentState) -> AgentState:
        query = state["query"]
        docs  = self.retriever.invoke(query)
        return {**state, "documents": docs}

    # ── Nodo 2: Evaluar relevancia de documentos ──────────────────────────────
    def grade_documents(self, state: AgentState) -> AgentState:
        docs  = state["documents"]
        query = state["query"]

        if not docs:
            return {**state, "confidence": 0.0, "documents": []}

        # Prompt para evaluar relevancia (binario por documento)
        grade_prompt = ChatPromptTemplate.from_messages([
            ("system", "Evalúa si el siguiente fragmento es relevante para la pregunta. "
                       "Responde SOLO con 'relevante' o 'no_relevante'."),
            ("human", "Pregunta: {query}\n\nFragmento: {doc}"),
        ])
        chain  = grade_prompt | self.llm
        scored = []
        for doc in docs:
            result = chain.invoke({"query": query, "doc": doc.page_content[:400]})
            if "relevante" in result.content.lower() and "no_relevante" not in result.content.lower():
                scored.append(doc)

        confidence = len(scored) / len(docs) if docs else 0.0
        return {**state, "documents": scored, "confidence": confidence}

    # ── Nodo 3: Generar respuesta ─────────────────────────────────────────────
    def generate(self, state: AgentState) -> AgentState:
        query    = state["query"]
        docs     = state["documents"]
        history  = state.get("chat_history", [])

        # Construir contexto
        context = "\n\n---\n\n".join([
            f"[Fuente: {d.metadata.get('source','?')}]\n{d.page_content}"
            for d in docs
        ]) if docs else "No se encontraron documentos relevantes en la base de conocimiento."

        # Historial de conversación
        messages = [SystemMessage(content=SYSTEM_PROMPT)]
        for h in history[-6:]:   # últimos 3 turnos
            if h["role"] == "user":
                messages.append(HumanMessage(content=h["content"]))
            else:
                messages.append(AIMessage(content=h["content"]))

        messages.append(HumanMessage(content=
            f"Contexto disponible:\n{context}\n\n"
            f"Pregunta: {query}\n\n"
            f"Responde basándote en el contexto. Si no hay suficiente información, "
            f"indícalo y sugiere qué documentos serían útiles."
        ))

        response = self.llm.invoke(messages)
        sources  = list({d.metadata.get("source", "?") for d in docs})

        return {**state, "generation": response.content, "sources": sources}

    # ── Nodo 4: Reformular query si no hay contexto ───────────────────────────
    def rewrite_query(self, state: AgentState) -> AgentState:
        query = state["query"]
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "Reformula esta pregunta para mejorar la búsqueda semántica "
                       "en documentos de e-commerce. Hazla más específica. "
                       "Responde SOLO con la pregunta reformulada, sin explicaciones."),
            ("human", "{query}"),
        ])
        chain    = rewrite_prompt | self.llm
        result   = chain.invoke({"query": query})
        new_query = result.content.strip()
        retry    = state.get("retry_count", 0) + 1
        return {**state, "query": new_query, "retry_count": retry}

    # ── Router: decidir si reformular o generar ───────────────────────────────
    def route_after_grade(
        self, state: AgentState
    ) -> Literal["generate", "rewrite_query"]:
        confidence  = state.get("confidence", 0.0)
        retry_count = state.get("retry_count", 0)

        # Si hay contexto suficiente o ya reintentamos 2 veces → generar
        if confidence >= 0.3 or retry_count >= 2:
            return "generate"
        return "rewrite_query"

    # ── Construir grafo LangGraph ─────────────────────────────────────────────
    def _build_graph(self) -> StateGraph:
        graph = StateGraph(AgentState)

        # Agregar nodos
        graph.add_node("retrieve",       self.retrieve)
        graph.add_node("grade_documents",self.grade_documents)
        graph.add_node("generate",       self.generate)
        graph.add_node("rewrite_query",  self.rewrite_query)

        # Flujo principal
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_documents")

        # Routing condicional
        graph.add_conditional_edges(
            "grade_documents",
            self.route_after_grade,
            {"generate": "generate", "rewrite_query": "rewrite_query"},
        )

        # Después de reformular → volver a recuperar
        graph.add_edge("rewrite_query", "retrieve")
        graph.add_edge("generate", END)

        return graph.compile()

    # ── Método público principal ──────────────────────────────────────────────
    def ask(
        self,
        query: str,
        chat_history: List[dict] | None = None,
    ) -> dict:
        """
        Ejecuta el agente RAG completo.
        Retorna: {answer, sources, confidence, steps}
        """
        initial_state: AgentState = {
            "query":        query,
            "documents":    [],
            "generation":   "",
            "retry_count":  0,
            "chat_history": chat_history or [],
            "confidence":   0.0,
            "sources":      [],
        }

        final = self.graph.invoke(initial_state)

        return {
            "answer":     final["generation"],
            "sources":    final["sources"],
            "confidence": round(final["confidence"], 2),
            "retries":    final["retry_count"],
            "query_used": final["query"],   # puede haber sido reformulada
        }

    def get_stats(self) -> dict:
        """Estado del vectorstore."""
        try:
            count = self.vectorstore._collection.count()
            if count == 0:
                return {"total_chunks": 0, "fuentes": []}
            data = self.vectorstore._collection.get(include=["metadatas"])
            sources = list({m.get("source","?") for m in data["metadatas"]})
            return {"total_chunks": count, "fuentes": sorted(sources)}
        except Exception as e:
            return {"total_chunks": 0, "fuentes": [], "error": str(e)}


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️  OPENAI_API_KEY no configurada")
    else:
        agent = RAGAgent(api_key)
        stats = agent.get_stats()
        print(f"Vectorstore: {stats['total_chunks']} chunks")
        if stats["total_chunks"] > 0:
            result = agent.ask("¿Cuáles son los productos más vendidos?")
            print(f"\nRespuesta: {result['answer'][:300]}")
            print(f"Fuentes: {result['sources']}")
            print(f"Confianza: {result['confidence']}")
