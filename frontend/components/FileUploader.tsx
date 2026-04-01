"use client";
import { useState, useRef } from "react";

interface Props {
  onIngested: (sourceId: string, chunkCount: number) => void;
}

export default function FileUploader({ onIngested }: Props) {
  const [status, setStatus] = useState<"idle" | "uploading" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const apiUrl = process.env.NEXT_PUBLIC_API_URL;

  async function upload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setStatus("error");
      setMessage("Only PDF files are supported.");
      return;
    }
    setStatus("uploading");
    setMessage(`Uploading ${file.name}…`);
    const form = new FormData();
    form.append("file", file);
    try {
      const res = await fetch(`${apiUrl}/ingest`, { method: "POST", body: form });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setStatus("done");
      setMessage(`Ingested ${data.ingested} chunks from ${data.source_id}`);
      onIngested(data.source_id, data.ingested);
    } catch (e: unknown) {
      setStatus("error");
      setMessage(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) upload(file);
  }

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.click()}
      className={`cursor-pointer rounded-xl border-2 border-dashed p-8 text-center transition-colors
        ${isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300 hover:border-blue-400 hover:bg-gray-50"}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf"
        className="hidden"
        onChange={(e) => { const f = e.target.files?.[0]; if (f) upload(f); }}
      />
      <p className="text-sm text-gray-500">
        {status === "idle" && "Drop a PDF here or click to upload"}
        {status === "uploading" && <span className="text-blue-600">{message}</span>}
        {status === "done" && <span className="text-green-600">{message}</span>}
        {status === "error" && <span className="text-red-600">{message}</span>}
      </p>
    </div>
  );
}
