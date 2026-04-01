import logging
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel, Field
import inngest
import inngest.fast_api
import uuid
import os
import datetime
from data_loader import load_and_chunk_pdf, embed_texts
from vector_db import QdrantStorage
from eval.chunk_stats import append_chunk_stats
from custom_types import RAGQuery, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
from config import (
    OPENAI_CHAT_MODEL,
    OPENAI_CHAT_MAX_TOKENS,
    OPENAI_CHAT_TEMPERATURE,
    RAG_SYSTEM_PROMPT,
    RAG_DEFAULT_TOP_K,
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=RAG_DEFAULT_TOP_K, ge=1, le=20)
    source_filter: str | None = None

class IngestResponse(BaseModel):
    ingested: int
    source_id: str

class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    contexts: list[str]


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
        try:
            append_chunk_stats(source_id, chunks)
        except Exception:
            pass
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

    def _search(question: str, top_k: int = RAG_DEFAULT_TOP_K) -> dict:
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
            model=OPENAI_CHAT_MODEL,
            max_tokens=OPENAI_CHAT_MAX_TOKENS,
            temperature=OPENAI_CHAT_TEMPERATURE,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        )
        answer = res.choices[0].message.content.strip()
        if os.getenv("RAG_EVAL_ENABLED", "false").lower() == "true":
            try:
                from eval.evaluate import evaluate_response
                evaluate_response(question, answer, found["contexts"])
            except Exception:
                pass
        return {"answer": answer, "sources": found["sources"], "num_contexts": len(found["contexts"])}

    question = ctx.event.data.get("question")
    top_k = int(ctx.event.data.get("top_k", RAG_DEFAULT_TOP_K))

    found = await ctx.step.run("embed-and-search", lambda: _search(question, top_k))
    return await ctx.step.run("llm-answer", lambda: _answer(found, question))


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

inngest.fast_api.serve(app, inngest_client, [ingest_pdf, query_pdf])  # correct function names


@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(file: UploadFile):
    import tempfile
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    source_id = file.filename
    chunks = load_and_chunk_pdf(tmp_path)
    try:
        append_chunk_stats(source_id, chunks)
    except Exception:
        pass
    vecs = embed_texts(chunks)
    ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}_{i}")) for i in range(len(chunks))]
    payloads = [{"source": source_id, "text": chunk} for chunk in chunks]
    QdrantStorage().upsert(ids, vecs, payloads)
    return IngestResponse(ingested=len(chunks), source_id=source_id)


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    query_vec = embed_texts([req.question])[0]
    found = QdrantStorage().search(query_vec, req.top_k, source_filter=req.source_filter)
    if not found["contexts"]:
        return QueryResponse(answer="No relevant context found.", sources=[], contexts=[])
    context_block = "\n\n".join(f"- {c}" for c in found["contexts"])
    user_content = (
        "Use the following context to answer the question:\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {req.question}\n\n"
        "Answer concisely using the context above."
    )
    client = OpenAI()
    res = client.chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        max_tokens=OPENAI_CHAT_MAX_TOKENS,
        temperature=OPENAI_CHAT_TEMPERATURE,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    answer = res.choices[0].message.content.strip()
    if os.getenv("RAG_EVAL_ENABLED", "false").lower() == "true":
        try:
            from eval.evaluate import evaluate_response
            evaluate_response(req.question, answer, found["contexts"], req.source_filter)
        except Exception:
            pass
    return QueryResponse(answer=answer, sources=found["sources"], contexts=found["contexts"])