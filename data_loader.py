from openai import OpenAI
from llama_index.readers.file import PDFReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

client = OpenAI()
EMBED_MODEL = OPENAI_EMBEDDING_MODEL
EMBED_DIM = EMBEDDING_DIMENSION

splitter = RecursiveCharacterTextSplitter(
    chunk_size=RAG_CHUNK_SIZE,
    chunk_overlap=RAG_CHUNK_OVERLAP,
)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=Path(path))
    texts = [d.text for d in docs if getattr(d, "text", None)]
    if not texts:
        return []
    full_doc_text = "\n\n".join(texts)
    return splitter.split_text(full_doc_text)


def summarize_chunks(chunks: list[str]) -> dict:
    if not chunks:
        return {"chunk_count": 0, "avg_chunk_length": 0.0, "min_chunk_length": 0, "max_chunk_length": 0}
    lengths = [len(c) for c in chunks]
    return {
        "chunk_count": len(chunks),
        "avg_chunk_length": round(sum(lengths) / len(lengths), 2),
        "min_chunk_length": min(lengths),
        "max_chunk_length": max(lengths),
    }

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    return [item.embedding for item in response.data]
