interface Props {
  answer: string;
  sources: string[];
}

export default function AnswerCard({ answer, sources }: Props) {
  return (
    <div className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
      <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-gray-500">Answer</h2>
      <p className="text-gray-800 leading-relaxed">{answer}</p>
      {sources.length > 0 && (
        <div className="mt-4">
          <h3 className="mb-1 text-xs font-semibold uppercase tracking-wide text-gray-400">Sources</h3>
          <ul className="space-y-1">
            {sources.map((src) => (
              <li key={src} className="rounded bg-gray-100 px-2 py-1 font-mono text-xs text-gray-700">
                {src}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}
