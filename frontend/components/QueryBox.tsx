"use client";
import { useState } from "react";

interface Props {
  onResult: (result: { answer: string; sources: string[]; contexts: string[] }) => void;
  onLoading: (loading: boolean) => void;
}

export default function QueryBox({ onResult, onLoading }: Props) {
  const [question, setQuestion] = useState("");
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  async function ask() {
    const q = question.trim();
    if (!q) return;
    onLoading(true);
    try {
      const res = await fetch(`${apiUrl}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q, top_k: 5, source_filter: null }),
      });
      if (!res.ok) throw new Error(await res.text());
      onResult(await res.json());
    } catch (e: unknown) {
      onResult({ answer: `Error: ${e instanceof Error ? e.message : String(e)}`, sources: [], contexts: [] });
    } finally {
      onLoading(false);
    }
  }

  return (
    <div className="flex gap-2">
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        onKeyDown={(e) => e.key === "Enter" && ask()}
        placeholder="Ask a question about your documents…"
        maxLength={2000}
        className="flex-1 rounded-lg border border-gray-300 px-4 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
      />
      <button
        onClick={ask}
        disabled={!question.trim()}
        className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-40"
      >
        Ask
      </button>
    </div>
  );
}
