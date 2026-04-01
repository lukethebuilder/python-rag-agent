@AGENTS.md

## Frontend Context

This `frontend/` app is the Next.js UI for the Python RAG backend.

- API base URL comes from `NEXT_PUBLIC_API_URL`
- Upload calls `POST /ingest`
- Query calls `POST /query`
- Main page composes:
  - `components/FileUploader.tsx`
  - `components/QueryBox.tsx`
  - `components/AnswerCard.tsx`
  - `components/ContextDrawer.tsx`

## Local Development

```bash
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Expected backend: FastAPI server running from repo root via:

```bash
uv run uvicorn main:app --reload
```

## Conventions

- Keep components typed and client-safe (`"use client"` where needed).
- Preserve current API contracts:
  - ingest response: `{ ingested, source_id }`
  - query response: `{ answer, sources, contexts }`
- Keep PDF validation in uploader (`.pdf` only) aligned with backend validation.
