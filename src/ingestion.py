"""
ingestion.py
────────────
Pipeline de ingesta de documentos:
  1. PDF → texto (pypdf)
  2. Imagen → texto (pytesseract OCR)
  3. Chunking semántico (RecursiveCharacterTextSplitter)
  4. Embeddings (OpenAI text-embedding-3-small)
  5. Persistencia en ChromaDB local

Dominio: E-commerce / retail peruano
"""
import os
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

# PDF
try:
    from pypdf import PdfReader
    HAS_PDF = True
except ImportError:
    HAS_PDF = False

# OCR
try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

BASE_DIR    = Path(__file__).parent.parent
VECTOR_DIR  = BASE_DIR / "data" / "vectorstore"
VECTOR_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION  = "ecommerce_docs"
CHUNK_SIZE  = 800
CHUNK_OVERLAP = 120


def _hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:12]


def extract_pdf(path: str) -> str:
    """Extrae texto de un PDF página por página."""
    if not HAS_PDF:
        raise ImportError("pypdf no instalado: pip install pypdf")
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[Página {i+1}]\n{text.strip()}")
    return "\n\n".join(pages)


def extract_image_ocr(path: str, lang: str = "spa+eng") -> str:
    """Extrae texto de imagen con pytesseract."""
    if not HAS_OCR:
        raise ImportError("pytesseract o Pillow no instalados")
    img = Image.open(path)
    return pytesseract.image_to_string(img, lang=lang)


def extract_text(path: str) -> str:
    """Detecta tipo de archivo y extrae texto."""
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return extract_pdf(path)
    elif ext in [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]:
        return extract_image_ocr(path)
    elif ext in [".txt", ".md", ".csv"]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError(f"Formato no soportado: {ext}")


def chunk_text(text: str, metadata: Dict[str, Any]) -> List[Document]:
    """Divide texto en chunks con overlap semántico."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "?", "!", " "],
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        meta = {**metadata, "chunk_index": i, "chunk_id": _hash(chunk)}
        docs.append(Document(page_content=chunk, metadata=meta))
    return docs


def get_vectorstore(api_key: str) -> Chroma:
    """Carga o crea el vectorstore ChromaDB."""
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=api_key,
    )
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(VECTOR_DIR),
    )


def ingest_file(
    path: str,
    api_key: str,
    extra_metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Ingesta un archivo completo al vectorstore.
    Retorna resumen de la operación.
    """
    path = str(path)
    filename = Path(path).name

    # Extraer texto
    text = extract_text(path)
    if not text.strip():
        return {"status": "error", "msg": "No se pudo extraer texto", "file": filename}

    # Metadata base
    metadata = {
        "source": filename,
        "tipo": Path(path).suffix.lower().replace(".", ""),
        "dominio": "ecommerce",
        **(extra_metadata or {}),
    }

    # Chunking
    docs = chunk_text(text, metadata)

    # Guardar en ChromaDB
    vs = get_vectorstore(api_key)
    vs.add_documents(docs)

    return {
        "status":   "ok",
        "file":     filename,
        "chars":    len(text),
        "chunks":   len(docs),
        "palabras": len(text.split()),
    }


def ingest_directory(folder: str, api_key: str) -> List[Dict]:
    """Ingesta todos los archivos compatibles de una carpeta."""
    supported = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".csv"}
    results = []
    for f in Path(folder).iterdir():
        if f.suffix.lower() in supported:
            result = ingest_file(str(f), api_key)
            results.append(result)
            status = "✅" if result["status"] == "ok" else "❌"
            print(f"{status} {result['file']} — {result.get('chunks','?')} chunks")
    return results


def get_collection_stats(api_key: str) -> Dict[str, Any]:
    """Retorna estadísticas del vectorstore actual."""
    vs = get_vectorstore(api_key)
    collection = vs._collection
    count = collection.count()
    if count == 0:
        return {"total_chunks": 0, "fuentes": [], "dominio": "ecommerce"}
    results = collection.get(include=["metadatas"])
    sources = list({m.get("source", "?") for m in results["metadatas"]})
    return {
        "total_chunks": count,
        "fuentes":      sorted(sources),
        "n_fuentes":    len(sources),
        "dominio":      "ecommerce",
    }


if __name__ == "__main__":
    import os
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("⚠️  Agrega OPENAI_API_KEY en .env")
    else:
        stats = get_collection_stats(api_key)
        print(f"Vectorstore: {stats['total_chunks']} chunks · {stats['n_fuentes']} fuentes")
