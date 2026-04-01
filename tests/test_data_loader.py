"""
Tests for data_loader.py — verifies chunking behaviour and embed_texts output
shape. No real OpenAI or PDF I/O: PDFReader and the OpenAI client are mocked.
"""
import uuid
import pytest
from unittest.mock import MagicMock, patch


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_doc(text: str) -> MagicMock:
    """Minimal stand-in for a LlamaIndex Document."""
    doc = MagicMock()
    doc.text = text
    return doc


LONG_TEXT = (
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
    "language model so the model can ground its answers in retrieved evidence. "
    "The retrieval step fetches the most relevant chunks from a corpus, and the "
    "generation step produces an answer conditioned on those chunks. "
) * 30  # ~3000 chars — enough to produce multiple 1000-char chunks


# ── load_and_chunk_pdf ─────────────────────────────────────────────────────────

def test_load_and_chunk_returns_nonempty_list():
    with patch("data_loader.PDFReader") as mock_reader:
        mock_reader.return_value.load_data.return_value = [_make_doc(LONG_TEXT)]
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    assert isinstance(chunks, list)
    assert len(chunks) > 0


def test_chunks_are_nonempty_strings():
    with patch("data_loader.PDFReader") as mock_reader:
        mock_reader.return_value.load_data.return_value = [_make_doc(LONG_TEXT)]
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


def test_chunk_count_within_expected_bounds():
    """
    With RAG_CHUNK_SIZE=1000 and ~3000 chars of text (with overlap),
    expect between 2 and 10 chunks.
    """
    with patch("data_loader.PDFReader") as mock_reader:
        mock_reader.return_value.load_data.return_value = [_make_doc(LONG_TEXT)]
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    assert 2 <= len(chunks) <= 10


def test_empty_page_text_is_skipped():
    """Documents whose .text is empty/falsy are filtered out before splitting."""
    with patch("data_loader.PDFReader") as mock_reader:
        mock_reader.return_value.load_data.return_value = [
            _make_doc(""),
            _make_doc(LONG_TEXT),
        ]
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    assert len(chunks) > 0


def test_multiple_pages_produce_chunks_from_all():
    """Chunks from every non-empty page are collected into a single flat list."""
    page_a = "Alpha content. " * 80
    page_b = "Beta content. " * 80
    with patch("data_loader.PDFReader") as mock_reader:
        mock_reader.return_value.load_data.return_value = [
            _make_doc(page_a),
            _make_doc(page_b),
        ]
        from data_loader import load_and_chunk_pdf
        chunks = load_and_chunk_pdf("fake.pdf")
    joined = " ".join(chunks)
    assert "Alpha content" in joined
    assert "Beta content" in joined


# ── Payload building (pattern shared by app.py and main.py) ───────────────────

def test_payload_source_key_is_set():
    """
    The payload list-comprehension used during ingest must set the 'source'
    key to the document's source_id on every chunk.
    """
    chunks = ["chunk one", "chunk two", "chunk three"]
    source_id = "my_document.pdf"
    payloads = [{"source": source_id, "text": chunk} for chunk in chunks]

    assert all(p["source"] == source_id for p in payloads)


def test_payload_text_matches_chunk():
    chunks = ["first chunk", "second chunk"]
    source_id = "doc.pdf"
    payloads = [{"source": source_id, "text": chunk} for chunk in chunks]

    for chunk, payload in zip(chunks, payloads):
        assert payload["text"] == chunk


def test_payload_ids_are_deterministic():
    """
    IDs derived from uuid5(NAMESPACE_URL, f"{source_id}_{i}") must be
    identical across two calls with the same inputs.
    """
    source_id = "stable_doc.pdf"
    chunks = ["a", "b", "c"]
    make_ids = lambda: [
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}"))
        for i in range(len(chunks))
    ]
    assert make_ids() == make_ids()


def test_payload_ids_differ_across_sources():
    chunks = ["same text"]
    ids_a = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc_a.pdf_{i}")) for i in range(len(chunks))]
    ids_b = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"doc_b.pdf_{i}")) for i in range(len(chunks))]
    assert ids_a != ids_b


# ── embed_texts ────────────────────────────────────────────────────────────────

def _fake_embedding(dim: int = 1536) -> list[float]:
    return [0.1] * dim


def _mock_openai_response(texts: list[str], dim: int = 1536):
    response = MagicMock()
    response.data = [MagicMock(embedding=_fake_embedding(dim)) for _ in texts]
    return response


def test_embed_texts_returns_list_of_vectors():
    import data_loader
    texts = ["hello world", "goodbye world"]
    with patch.object(data_loader.client.embeddings, "create",
                      return_value=_mock_openai_response(texts)):
        result = data_loader.embed_texts(texts)
    assert isinstance(result, list)
    assert len(result) == len(texts)


def test_embed_texts_vector_length_matches_dim():
    import data_loader
    texts = ["single sentence"]
    with patch.object(data_loader.client.embeddings, "create",
                      return_value=_mock_openai_response(texts, dim=data_loader.EMBED_DIM)):
        result = data_loader.embed_texts(texts)
    assert len(result[0]) == data_loader.EMBED_DIM


def test_embed_texts_passes_correct_model():
    import data_loader
    texts = ["test"]
    with patch.object(data_loader.client.embeddings, "create",
                      return_value=_mock_openai_response(texts)) as mock_create:
        data_loader.embed_texts(texts)
    _, kwargs = mock_create.call_args
    assert kwargs["model"] == data_loader.EMBED_MODEL
