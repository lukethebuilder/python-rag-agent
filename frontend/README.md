# Frontend (Next.js)

This app is the web UI for the Python RAG backend in the parent project.
It uploads PDFs to FastAPI, sends questions to the query route, and renders
answers, sources, and retrieved contexts.

## Requirements

- Node.js 20+
- Backend API running at `http://localhost:8000`

From the project root, run:

```bash
uv run uvicorn main:app --reload
```

## Run Locally

```bash
cd frontend
npm install
NEXT_PUBLIC_API_URL=http://localhost:8000 npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Environment

- `NEXT_PUBLIC_API_URL` (required): FastAPI base URL, for example
  `http://localhost:8000`

## API Contract Used

- `POST /ingest` (multipart form-data with `file`)
  - Response: `{ "ingested": number, "source_id": string }`
- `POST /query` (JSON body)
  - Request: `{ "question": string, "top_k"?: number, "source_filter"?: string | null }`
  - Response: `{ "answer": string, "sources": string[], "contexts": string[] }`

## Component Map

- `components/FileUploader.tsx` - drag/drop or click upload for PDFs
- `components/QueryBox.tsx` - question input and submit
- `components/AnswerCard.tsx` - answer text and sources list
- `components/ContextDrawer.tsx` - expandable retrieved chunks
- `app/page.tsx` - page composition and local UI state

## Troubleshooting

- If upload/query fails, verify `NEXT_PUBLIC_API_URL` and that FastAPI is running.
- If browser requests are blocked, check CORS settings in `main.py`.
- Only `.pdf` files are accepted by both frontend and backend validators.
