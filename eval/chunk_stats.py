"""
Chunking benchmark helpers.

append_chunk_stats() records chunk count and length stats per ingest run so
chunking strategy changes can be compared over time.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from config import RAG_CHUNK_SIZE, RAG_CHUNK_OVERLAP, RAG_CHUNKING_STRATEGY
from data_loader import summarize_chunks

CHUNK_STATS_FILE = Path(__file__).parent / "chunk_stats.jsonl"


def append_chunk_stats(source_id: str, chunks: list[str]) -> dict:
    stats = summarize_chunks(chunks)
    record = {
        "source_id": source_id,
        "chunking_strategy": RAG_CHUNKING_STRATEGY,
        "chunk_size": RAG_CHUNK_SIZE,
        "chunk_overlap": RAG_CHUNK_OVERLAP,
        **stats,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    CHUNK_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CHUNK_STATS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record
