"use client";
import { useState } from "react";
import FileUploader from "@/components/FileUploader";
import QueryBox from "@/components/QueryBox";
import AnswerCard from "@/components/AnswerCard";
import ContextDrawer from "@/components/ContextDrawer";

interface QueryResult {
  answer: string;
  sources: string[];
  contexts: string[];
}

export default function Home() {
  const [result, setResult] = useState<QueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [ingestedDocs, setIngestedDocs] = useState<string[]>([]);

  function handleIngested(sourceId: string) {
    setIngestedDocs((prev) => [...new Set([...prev, sourceId])]);
  }

  return (
    <main className="mx-auto max-w-2xl px-4 py-10">
      <h1 className="mb-8 text-2xl font-bold text-gray-900">RAG Agent</h1>

      <section className="mb-6">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Ingest a Document
        </h2>
        <FileUploader onIngested={handleIngested} />
        {ingestedDocs.length > 0 && (
          <p className="mt-2 text-xs text-gray-500">
            Ingested: {ingestedDocs.join(", ")}
          </p>
        )}
      </section>

      <section className="mb-6">
        <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">
          Ask a Question
        </h2>
        <QueryBox onResult={setResult} onLoading={setLoading} />
      </section>

      {loading && (
        <p className="text-sm text-gray-500">Thinking…</p>
      )}

      {result && !loading && (
        <div className="space-y-4">
          <AnswerCard answer={result.answer} sources={result.sources} />
          <ContextDrawer contexts={result.contexts} />
        </div>
      )}
    </main>
  );
}
