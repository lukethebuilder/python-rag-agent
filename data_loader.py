from openai import OpenAI
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path
from config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION, RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP

client = OpenAI()
EMBED_MODEL = OPENAI_EMBEDDING_MODEL
EMBED_DIM = EMBEDDING_DIMENSION

splitter = SentenceSplitter(chunk_size=RAG_CHUNK_SIZE, chunk_overlap=RAG_CHUNK_OVERLAP)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=Path(path))
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBED_MODEL
    )
    return [item.embedding for item in response.data]
