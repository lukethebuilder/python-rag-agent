"""
Tests for vector_db.py — verifies upsert, search (with and without source
filter), and list_sources against an in-memory Qdrant instance.

QdrantClient(":memory:") is used so no running Qdrant server is required.
The patch replaces vector_db.QdrantClient only during __init__ so the
storage object's .client attribute holds the real in-memory client for all
subsequent method calls.
"""
import uuid
import pytest
from unittest.mock import patch
from qdrant_client import QdrantClient as RealQdrantClient

import vector_db
from vector_db import QdrantStorage

DIM = 4  # small dimensionality to keep tests fast


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def store():
    """QdrantStorage backed by an isolated in-memory Qdrant instance."""
    mem_client = RealQdrantClient(":memory:")
    with patch("vector_db.QdrantClient", return_value=mem_client):
        storage = QdrantStorage(collection="test_col", dim=DIM)
    return storage


def _vec(a: float, b: float, c: float, d: float) -> list[float]:
    return [a, b, c, d]


def _ingest(store: QdrantStorage, source: str, n: int = 2):
    """Helper: ingest n chunks attributed to `source` using UUID5 IDs matching the app."""
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}_{i}")) for i in range(n)]
    vectors = [_vec(0.9, 0.1, 0.0, float(i) * 0.05) for i in range(n)]
    payloads = [{"source": source, "text": f"chunk {i} from {source}"} for i in range(n)]
    store.upsert(ids, vectors, payloads)


# ── upsert + search round-trip ─────────────────────────────────────────────────

def test_upsert_then_search_returns_results(store):
    _ingest(store, "doc_a.pdf")
    query = _vec(0.9, 0.1, 0.0, 0.0)
    result = store.search(query, top_k=5)
    assert len(result["contexts"]) > 0


def test_search_contexts_are_strings(store):
    _ingest(store, "doc_a.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=5)
    assert all(isinstance(c, str) for c in result["contexts"])


def test_search_result_has_expected_keys(store):
    _ingest(store, "doc_a.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0))
    assert "contexts" in result
    assert "sources" in result


def test_search_top_k_limits_results(store):
    _ingest(store, "doc_a.pdf", n=4)
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=2)
    assert len(result["contexts"]) <= 2


# ── source filter ──────────────────────────────────────────────────────────────

def test_search_without_filter_returns_all_sources(store):
    _ingest(store, "doc_a.pdf")
    _ingest(store, "doc_b.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=10)
    assert "doc_a.pdf" in result["sources"]
    assert "doc_b.pdf" in result["sources"]


def test_search_with_source_filter_returns_only_matching(store):
    _ingest(store, "doc_a.pdf")
    _ingest(store, "doc_b.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=10, source_filter="doc_a.pdf")
    assert result["sources"] == ["doc_a.pdf"]
    assert all("doc_a.pdf" in c for c in result["contexts"])


def test_search_source_filter_excludes_other_sources(store):
    _ingest(store, "doc_a.pdf")
    _ingest(store, "doc_b.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=10, source_filter="doc_b.pdf")
    assert "doc_a.pdf" not in result["sources"]


def test_search_nonexistent_source_filter_returns_empty(store):
    _ingest(store, "doc_a.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=10, source_filter="nonexistent.pdf")
    assert result["contexts"] == []
    assert result["sources"] == []


def test_search_empty_source_filter_searches_all(store):
    """Falsy source_filter (empty string) should behave as no filter."""
    _ingest(store, "doc_a.pdf")
    _ingest(store, "doc_b.pdf")
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=10, source_filter="")
    assert "doc_a.pdf" in result["sources"]
    assert "doc_b.pdf" in result["sources"]


# ── list_sources ───────────────────────────────────────────────────────────────

def test_list_sources_empty_collection(store):
    assert store.list_sources() == []


def test_list_sources_single_document(store):
    _ingest(store, "report.pdf")
    assert store.list_sources() == ["report.pdf"]


def test_list_sources_multiple_documents(store):
    _ingest(store, "doc_a.pdf")
    _ingest(store, "doc_b.pdf")
    sources = store.list_sources()
    assert "doc_a.pdf" in sources
    assert "doc_b.pdf" in sources


def test_list_sources_returns_sorted(store):
    _ingest(store, "zebra.pdf")
    _ingest(store, "apple.pdf")
    _ingest(store, "mango.pdf")
    sources = store.list_sources()
    assert sources == sorted(sources)


def test_list_sources_no_duplicates(store):
    """Re-ingesting the same source (same IDs) must not duplicate it."""
    _ingest(store, "doc_a.pdf", n=3)
    _ingest(store, "doc_a.pdf", n=3)
    sources = store.list_sources()
    assert sources.count("doc_a.pdf") == 1


# ── upsert determinism ─────────────────────────────────────────────────────────

def test_upsert_same_id_overwrites(store):
    """Upserting a point with the same ID should overwrite, not duplicate."""
    point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, "doc.pdf_0"))
    store.upsert(
        ids=[point_id],
        vectors=[_vec(0.9, 0.1, 0.0, 0.0)],
        payloads=[{"source": "doc.pdf", "text": "original"}],
    )
    store.upsert(
        ids=[point_id],
        vectors=[_vec(0.9, 0.1, 0.0, 0.0)],
        payloads=[{"source": "doc.pdf", "text": "overwritten"}],
    )
    result = store.search(_vec(0.9, 0.1, 0.0, 0.0), top_k=5)
    assert len(result["contexts"]) == 1
    assert result["contexts"][0] == "overwritten"
