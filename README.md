# Python RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline built in Python. Upload PDFs, embed them into a vector database, and query them with natural language — powered by OpenAI and Qdrant.

> Built by following along with [**How to Build a Production-Ready RAG AI Agent in Python (Step-by-Step)**](https://www.youtube.com/watch?v=AUQJ9eeP-Ls) with Claude assistance.

## Stack

- **OpenAI** — embeddings (`text-embedding-3-small`) and answers (`gpt-4o-mini`)
- **Qdrant** — local vector database
- **Inngest** — durable, event-driven function execution
- **FastAPI** — API server for Inngest functions
- **Streamlit** — frontend UI
- **LlamaIndex** — PDF loading and text chunking

## Features

- Ingest PDFs via drag-and-drop UI or Inngest event
- Chunk, embed, and store documents in Qdrant
- Query documents with natural language
- View retrieved source chunks
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
├── main.py          # FastAPI app + Inngest functions (ingest & query)
├── app.py           # Streamlit frontend
├── data_loader.py   # PDF loading and chunking
├── vector_db.py     # Qdrant wrapper
├── custom_types.py  # Pydantic models
└── pyproject.toml   # Dependencies
```
