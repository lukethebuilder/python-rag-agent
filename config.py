import os
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "docs")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_CHAT_MAX_TOKENS = int(os.getenv("OPENAI_CHAT_MAX_TOKENS", "1024"))
OPENAI_CHAT_TEMPERATURE = float(os.getenv("OPENAI_CHAT_TEMPERATURE", "0.2"))
RAG_SYSTEM_PROMPT = os.getenv(
    "RAG_SYSTEM_PROMPT",
    "You answer questions using only the provided context."
)
RAG_DEFAULT_TOP_K = int(os.getenv("RAG_DEFAULT_TOP_K", "5"))
