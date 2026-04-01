import tempfile
import uuid
import streamlit as st

from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from config import (
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_TEMPERATURE,
    RAG_SYSTEM_PROMPT,
    RAG_DEFAULT_TOP_K,
)

st.set_page_config(page_title="RAG Agent", layout="centered")
st.title("RAG Agent")

# ── Ingest ──────────────────────────────────────────────────────────────────
st.header("Ingest PDF")
uploaded = st.file_uploader("Upload a PDF", type="pdf")

if uploaded and st.button("Ingest"):
    with st.spinner("Loading and chunking…"):
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        chunks = load_and_chunk_pdf(tmp_path)

    with st.spinner(f"Embedding {len(chunks)} chunks…"):
        vecs = embed_texts(chunks)
        source_id = uploaded.name
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
        QdrantStorage().upsert(ids, vecs, payloads)

    st.success(f"Ingested {len(chunks)} chunks from **{uploaded.name}**")

st.divider()

# ── Query ────────────────────────────────────────────────────────────────────
st.header("Ask a Question")
question = st.text_input("Question", placeholder="What is this document about?")
if st.button("Ask") and question:
    with st.spinner("Searching…"):
        query_vec = embed_texts([question])[0]
        result = QdrantStorage().search(query_vec, RAG_DEFAULT_TOP_K)
        contexts = result["contexts"]
        sources = result["sources"]

    if not contexts:
        st.warning("No relevant context found. Try ingesting a document first.")
    else:
        with st.spinner("Generating answer…"):
            from openai import OpenAI
            context_block = "\n\n".join(f"- {c}" for c in contexts)
            user_content = (
                "Use the following context to answer the question:\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {question}\n\n"
                "Answer concisely using the context above."
            )
            res = OpenAI().chat.completions.create(
                model=OPENAI_CHAT_MODEL,
                max_tokens=OPENAI_CHAT_MAX_TOKENS,
                temperature=OPENAI_CHAT_TEMPERATURE,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            answer = res.choices[0].message.content.strip()

        st.subheader("Answer")
        st.write(answer)

        if sources:
            st.subheader("Sources")
            for src in sources:
                st.markdown(f"- `{src}`")

        with st.expander("Retrieved contexts"):
            for i, ctx in enumerate(contexts, 1):
                st.markdown(f"**Chunk {i}**")
                st.text(ctx)
