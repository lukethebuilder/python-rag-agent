"""
RAGAS evaluation helpers.

evaluate_response() scores a single RAG response for faithfulness and answer
relevancy, then appends the result to eval/scores.jsonl.

Usage (called from app.py and main.py after a query):
    from eval.evaluate import evaluate_response
    evaluate_response(question, answer, contexts)
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from config import OPENAI_EMBEDDING_MODEL, EMBEDDING_DIMENSION, RAG_CHUNKING_STRATEGY

SCORES_FILE = Path(__file__).parent / "scores.jsonl"


def evaluate_response(
    question: str,
    answer: str,
    contexts: list[str],
    source_filter: str | None = None,
) -> dict:
    """
    Run RAGAS faithfulness + answer_relevancy on a single question/answer/contexts
    triple and append the scored result to eval/scores.jsonl.

    Returns the score dict (keys: faithfulness, answer_relevancy, question,
    answer, source_filter, embedding_model, embedding_dimension,
    chunking_strategy, evaluated_at).
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy
    from datasets import Dataset

    sample = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],
    }

    result = evaluate(
        Dataset.from_dict(sample),
        metrics=[faithfulness, answer_relevancy],
    )
    scores_df = result.to_pandas()

    record = {
        "question": question,
        "answer": answer,
        "source_filter": source_filter,
        "embedding_model": OPENAI_EMBEDDING_MODEL,
        "embedding_dimension": EMBEDDING_DIMENSION,
        "chunking_strategy": RAG_CHUNKING_STRATEGY,
        "faithfulness": round(float(scores_df["faithfulness"].iloc[0]), 4)
            if "faithfulness" in scores_df.columns else None,
        "answer_relevancy": round(float(scores_df["answer_relevancy"].iloc[0]), 4)
            if "answer_relevancy" in scores_df.columns else None,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }

    SCORES_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORES_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record
