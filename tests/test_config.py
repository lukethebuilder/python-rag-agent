"""
Tests for config.py — verifies that all env vars load with correct defaults,
that os.environ overrides work, and that type casting is enforced.

Each test uses importlib.reload(config) with a patched load_dotenv to ensure
complete isolation from any .env file on disk.
"""
import importlib
import pytest
from unittest.mock import patch

CONFIG_KEYS = [
    "QDRANT_URL",
    "QDRANT_COLLECTION",
    "EMBEDDING_DIMENSION",
    "OPENAI_EMBEDDING_MODEL",
    "RAG_CHUNK_SIZE",
    "RAG_CHUNK_OVERLAP",
    "RAG_CHUNKING_STRATEGY",
    "OPENAI_CHAT_MODEL",
    "OPENAI_CHAT_MAX_TOKENS",
    "OPENAI_CHAT_TEMPERATURE",
    "RAG_SYSTEM_PROMPT",
    "RAG_DEFAULT_TOP_K",
]


def reload_config(monkeypatch, overrides: dict | None = None):
    """
    Delete all config-related env vars, apply any overrides, patch load_dotenv
    to a no-op so the on-disk .env file cannot interfere, then reload config.
    """
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, val in (overrides or {}).items():
        monkeypatch.setenv(key, val)
    import config
    with patch("config.load_dotenv"):
        return importlib.reload(config)


# ── String defaults ────────────────────────────────────────────────────────────

def test_default_qdrant_url(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.QDRANT_URL == "http://localhost:6333"


def test_default_qdrant_collection(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.QDRANT_COLLECTION == "docs"


def test_default_embedding_model(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.OPENAI_EMBEDDING_MODEL == "text-embedding-3-small"


def test_default_chat_model(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.OPENAI_CHAT_MODEL == "gpt-4o-mini"


def test_default_system_prompt_contains_context(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert "provided context" in cfg.RAG_SYSTEM_PROMPT


def test_default_chunking_strategy(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.RAG_CHUNKING_STRATEGY == "recursive_character_text_splitter"


# ── Numeric defaults ───────────────────────────────────────────────────────────

def test_default_embedding_dimension(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.EMBEDDING_DIMENSION == 1536


def test_default_chunk_size(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.RAG_CHUNK_SIZE == 1000


def test_default_chunk_overlap(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.RAG_CHUNK_OVERLAP == 200


def test_default_max_tokens(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.OPENAI_CHAT_MAX_TOKENS == 1024


def test_default_top_k(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.RAG_DEFAULT_TOP_K == 5


def test_default_temperature(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert cfg.OPENAI_CHAT_TEMPERATURE == pytest.approx(0.2)


# ── Type enforcement ───────────────────────────────────────────────────────────

def test_int_types(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert isinstance(cfg.EMBEDDING_DIMENSION, int)
    assert isinstance(cfg.RAG_CHUNK_SIZE, int)
    assert isinstance(cfg.RAG_CHUNK_OVERLAP, int)
    assert isinstance(cfg.OPENAI_CHAT_MAX_TOKENS, int)
    assert isinstance(cfg.RAG_DEFAULT_TOP_K, int)


def test_float_type(monkeypatch):
    cfg = reload_config(monkeypatch)
    assert isinstance(cfg.OPENAI_CHAT_TEMPERATURE, float)


# ── Env var overrides ──────────────────────────────────────────────────────────

def test_override_qdrant_collection(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"QDRANT_COLLECTION": "my_docs"})
    assert cfg.QDRANT_COLLECTION == "my_docs"


def test_override_qdrant_url(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"QDRANT_URL": "http://remote:6333"})
    assert cfg.QDRANT_URL == "http://remote:6333"


def test_override_chunk_size(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"RAG_CHUNK_SIZE": "512"})
    assert cfg.RAG_CHUNK_SIZE == 512


def test_override_chunking_strategy(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"RAG_CHUNKING_STRATEGY": "custom"})
    assert cfg.RAG_CHUNKING_STRATEGY == "custom"


def test_override_temperature(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"OPENAI_CHAT_TEMPERATURE": "0.9"})
    assert cfg.OPENAI_CHAT_TEMPERATURE == pytest.approx(0.9)


def test_override_top_k(monkeypatch):
    cfg = reload_config(monkeypatch, overrides={"RAG_DEFAULT_TOP_K": "10"})
    assert cfg.RAG_DEFAULT_TOP_K == 10


# ── Invalid values raise at import ────────────────────────────────────────────

def test_invalid_embedding_dimension_raises(monkeypatch):
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("EMBEDDING_DIMENSION", "notanumber")
    import config
    with patch("config.load_dotenv"):
        with pytest.raises(ValueError):
            importlib.reload(config)


def test_invalid_chunk_size_raises(monkeypatch):
    for key in CONFIG_KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("RAG_CHUNK_SIZE", "big")
    import config
    with patch("config.load_dotenv"):
        with pytest.raises(ValueError):
            importlib.reload(config)
