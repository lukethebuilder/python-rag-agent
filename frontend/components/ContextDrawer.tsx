"use client";
import { useState } from "react";

interface Props {
  contexts: string[];
}

export default function ContextDrawer({ contexts }: Props) {
  const [open, setOpen] = useState(false);
  if (contexts.length === 0) return null;

  return (
    <div className="rounded-xl border border-gray-200">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-3 text-sm font-medium text-gray-700 hover:bg-gray-50"
      >
        <span>Retrieved contexts ({contexts.length})</span>
        <span>{open ? "▲" : "▼"}</span>
      </button>
      {open && (
        <div className="divide-y divide-gray-100 border-t border-gray-200">
          {contexts.map((ctx, i) => (
            <div key={i} className="px-4 py-3">
              <p className="mb-1 text-xs font-semibold text-gray-400">Chunk {i + 1}</p>
              <p className="whitespace-pre-wrap text-xs text-gray-600">{ctx}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
