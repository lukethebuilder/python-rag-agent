"""Evaluation dashboard — reads eval/scores.jsonl and displays score history."""
import json
import os
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="RAG Eval", layout="wide")
st.title("Evaluation History")

SCORES_FILE = Path(__file__).parent.parent / "eval" / "scores.jsonl"

if not SCORES_FILE.exists() or SCORES_FILE.stat().st_size == 0:
    st.info(
        "No evaluation scores yet. Set `RAG_EVAL_ENABLED=true` in your `.env` "
        "and ask a question from the main page to record your first score."
    )
    st.stop()

records = []
with open(SCORES_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

if not records:
    st.info("No valid score records found in scores.jsonl.")
    st.stop()

import pandas as pd

df = pd.DataFrame(records)

# ── Summary metrics ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("Total evaluations", len(df))
if "faithfulness" in df.columns:
    col2.metric("Avg faithfulness", f"{df['faithfulness'].mean():.3f}")
if "answer_relevancy" in df.columns:
    col3.metric("Avg answer relevancy", f"{df['answer_relevancy'].mean():.3f}")

st.divider()

# ── Score history chart ────────────────────────────────────────────────────────
chart_cols = [c for c in ["faithfulness", "answer_relevancy"] if c in df.columns]
if chart_cols:
    st.subheader("Score history")
    st.line_chart(df[chart_cols])

st.divider()

# ── Full table ─────────────────────────────────────────────────────────────────
st.subheader("All records")
display_cols = [c for c in ["evaluated_at", "question", "faithfulness", "answer_relevancy", "source_filter"] if c in df.columns]
st.dataframe(df[display_cols], use_container_width=True)
