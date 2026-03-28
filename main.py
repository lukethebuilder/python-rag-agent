import logging
from fastapi import FastAPI
from openai import OpenAI
import inngest
import inngest.fast_api
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from custom_types import RAGQuery, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from dotenv import load_dotenv

load_dotenv()

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
)

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def ingest_pdf(ctx: inngest.Context):

    def _load() -> RAGChunkAndSrc:
        pdf_path = ctx.event.data.get("pdf_path")
        source_id = ctx.event.data.get("source_id", pdf_path)
        chunks = load_and_chunk_pdf(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        vecs = embed_texts(chunks)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", _load)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src))
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
)
async def query_pdf(ctx: inngest.Context):

    def _search(question: str, top_k: int = 5) -> dict:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        return store.search(query_vec, top_k)

    def _answer(found: dict, question: str) -> dict:
        context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
        user_content = (
            "Use the following context to answer the question:\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer concisely using the context above."
        )
        client = OpenAI()
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=1024,
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You answer questions using only the provided context."},
                {"role": "user", "content": user_content},
            ]
        )
        answer = res.choices[0].message.content.strip()
        return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}

    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", 5))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))
    return await ctx.step.run("llm-answer", lambda: _answer(found, question))


app = FastAPI()

inngest.fast_api.serve(app, inngest_client, [ingest_pdf, query_pdf])  # correct function names