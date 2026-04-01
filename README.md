# Python RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built in Python.
Ingest PDFs, embed chunks into Qdrant, and answer questions using retrieved
context with both Streamlit and Next.js frontends.

> Built by following along with [**How to Build a Production-Ready RAG AI Agent in Python (Step-by-Step)**](https://www.youtube.com/watch?v=AUQJ9eeP-Ls) with Claude assistance.

## Stack

- **OpenAI** — embeddings (`text-embedding-3-small`) and answers (`gpt-4o-mini`)
- **Qdrant** — local vector database
- **Inngest** — durable, event-driven function execution
- **FastAPI** — REST API (`/ingest`, `/query`) plus Inngest handler
- **Streamlit** — Python UI
- **Next.js** — React frontend in `frontend/`
- **LlamaIndex + LangChain Text Splitters** — PDF loading and cross-page chunking

## Features

- Ingest PDFs via Streamlit, Next.js, or Inngest event
- Cross-page chunking via `RecursiveCharacterTextSplitter`
- Source-aware retrieval via Qdrant payload filtering (`source_filter`)
- Query through `POST /query` and receive `answer`, `sources`, and `contexts`
- Log RAGAS scores to `eval/scores.jsonl`
- Log chunking benchmark stats to `eval/chunk_stats.jsonl`
- Test and observe function runs via the Inngest Dev Server

## Setup

### Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv)
- Docker (for Qdrant)
- [Inngest CLI](https://github.com/inngest/inngest) (`brew install inngest/tap/inngest`)

### Install

```bash
git clone <your-repo-url>
cd python-rag-agent
uv sync
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### Environment Variables

Core variables are documented in `.env.example`. Key values:

- `OPENAI_API_KEY` — required for embeddings/chat
- `OPENAI_EMBEDDING_MODEL` + `EMBEDDING_DIMENSION` — must stay in sync
- `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` / `RAG_CHUNKING_STRATEGY`
- `QDRANT_URL` / `QDRANT_COLLECTION`
- `RAG_EVAL_ENABLED` — set `true` to score each query with RAGAS

### Run Qdrant

```bash
docker run -p 6333:6333 qdrant/qdrant
```

If you change `OPENAI_EMBEDDING_MODEL` or `EMBEDDING_DIMENSION`, recreate your
Qdrant collection and re-ingest documents so vector size stays in sync.

## Usage

### Streamlit UI (normal use)

```bash
uv run streamlit run app.py
```

Open `localhost:8501`, upload a PDF, click **Ingest**, then ask a question.

### FastAPI routes

```bash
uv run uvicorn main:app --reload
```

With the API running at `http://localhost:8000`:

- `POST /ingest` (multipart file upload, PDF only)
- `POST /query` (JSON body with `question`, optional `top_k`, `source_filter`)

Example query:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"What is this document about?","top_k":5,"source_filter":null}'
```

### Next.js frontend

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open `localhost:3000` to use the frontend components (`FileUploader`,
`QueryBox`, `AnswerCard`, `ContextDrawer`) against the FastAPI routes.

### Inngest testing (event-driven path)

```bash
# Terminal 1
uv run uvicorn main:app --reload

# Terminal 2
inngest dev
```

Open `localhost:8288`, go to **Functions**, and invoke `RAG: Query PDF`:

```json
{
  "data": {
    "question": "Your question here"
  }
}
```

To ingest via Inngest:

```json
{
  "data": {
    "pdf_path": "/absolute/path/to/your.pdf"
  }
}
```

## Project Structure

```
├── main.py             # FastAPI app + Inngest functions
├── app.py              # Streamlit UI
├── data_loader.py      # PDF loading, chunking, embeddings
├── vector_db.py        # Qdrant wrapper + source filter
├── eval/evaluate.py    # RAGAS scoring and JSONL logging
├── eval/chunk_stats.py # Chunk statistics logger
├── pages/1_eval.py     # Streamlit eval history page
├── frontend/           # Next.js frontend
└── tests/              # Pytest suite
```
