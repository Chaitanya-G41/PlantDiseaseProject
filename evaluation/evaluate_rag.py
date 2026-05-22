"""
evaluation/evaluate_rag.py
--------------------------
RAGAS evaluation of the GuavaScan RAG pipeline.

Runs 3 ablation variants:
  1. Dense only          (BM25 disabled)
  2. Dense + BM25        (default hybrid)
  3. Dense + BM25 + Reranker

Metrics: faithfulness, answer_relevancy, context_precision, context_recall

Results saved to: evaluation/results.csv
Summary ablation table printed to console.

Usage:
    python evaluation/evaluate_rag.py
    python evaluation/evaluate_rag.py --config path/to/config.yaml
"""

import os
import sys
import json
import copy
import argparse
import pandas as pd
from dotenv import load_dotenv

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

load_dotenv(os.path.join(ROOT, ".env"))

import yaml
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from rag.chain import run_rag_chain, load_config
from rag.retriever import get_retriever


# ── Config loader ──────────────────────────────────────────────────────────────
def load_cfg(path=None):
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ── Map disease label to model class name ──────────────────────────────────────
DISEASE_TO_CLASS = {
    "Canker":         "Guava_anthracnose",
    "Healthy":        "Guava_healthy",
    "Insect_bite":    "Guava_insect_bite",
    "Mummification":  "Guava_yld",
    "Multiple":       "Guava_multiple",
    "Rust":           "Guava_anthracnose",   # closest available class; RAG query handles it
    "Scorch":         "Guava_scorch",
}

# For diseases not in the model's class list, override the query directly
DISEASE_QUERY_OVERRIDE = {
    "Canker":        "guava canker anthracnose disease symptoms treatment management",
    "Rust":          "guava rust Puccinia psidii symptoms treatment management",
    "Mummification": "guava mummification fruit shoot disease treatment management",
}


# ── Single evaluation run ──────────────────────────────────────────────────────
def run_evaluation(cfg: dict, variant_name: str, eval_data: list) -> list[dict]:
    """
    Run RAGAS evaluation for one system variant.
    Returns list of result dicts with per-question metrics.
    """
    print(f"\n{'='*60}")
    print(f"Evaluating variant: {variant_name}")
    print(f"{'='*60}")

    retriever = get_retriever(cfg)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set in .env")

    llm = ChatGoogleGenerativeAI(
        model=cfg["llm_model"],
        google_api_key=api_key,
        temperature=0.1,
    )
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )

    questions     = []
    answers       = []
    ground_truths = []
    contexts_list = []
    diseases      = []
    q_types       = []

    for i, item in enumerate(eval_data):
        disease      = item["disease"]
        question     = item["question"]
        q_type       = item["question_type"]
        ground_truth = item["ground_truth"]

        print(f"  [{i+1:02d}/{len(eval_data)}] {disease} — {q_type}")

        # Build retrieval query
        if disease in DISEASE_QUERY_OVERRIDE:
            query = DISEASE_QUERY_OVERRIDE[disease]
        else:
            model_class = DISEASE_TO_CLASS.get(disease, f"Guava_{disease.lower()}")
            query_map = cfg.get("class_query_map", {})
            query = query_map.get(model_class, f"guava {disease.lower()} disease treatment")

        # Append question context to query for better retrieval
        full_query = f"{query} {question}"

        # Retrieve
        chunks = retriever.retrieve(full_query)
        context_texts = [c["content"] for c in chunks]

        # Generate answer using chain
        model_class = DISEASE_TO_CLASS.get(disease, "Guava_anthracnose")
        try:
            result = run_rag_chain(
                predicted_class=model_class,
                confidence=0.95,
                cfg=cfg,
                retriever=retriever,
            )
            answer = result["answer"]
        except Exception as e:
            print(f"    ⚠ Chain error: {e}")
            answer = f"Error generating answer: {e}"

        questions.append(question)
        answers.append(answer)
        ground_truths.append([ground_truth])
        contexts_list.append(context_texts)
        diseases.append(disease)
        q_types.append(q_type)

    # ── RAGAS evaluation ───────────────────────────────────────────────────────
    print(f"\n  Running RAGAS metrics for {variant_name}...")
    dataset = Dataset.from_dict({
        "question":     questions,
        "answer":       answers,
        "contexts":     contexts_list,
        "ground_truth": ground_truths,
    })

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )

    # ── Collect per-row results ────────────────────────────────────────────────
    results = []
    scores_df = ragas_result.to_pandas()

    for i, row in scores_df.iterrows():
        results.append({
            "disease":           diseases[i],
            "question_type":     q_types[i],
            "system_variant":    variant_name,
            "faithfulness":      round(float(row.get("faithfulness", float("nan"))), 4),
            "answer_relevancy":  round(float(row.get("answer_relevancy", float("nan"))), 4),
            "context_precision": round(float(row.get("context_precision", float("nan"))), 4),
            "context_recall":    round(float(row.get("context_recall", float("nan"))), 4),
        })

    return results


# ── Ablation variants ──────────────────────────────────────────────────────────
def build_variants(base_cfg: dict) -> list[tuple[str, dict]]:
    """Return (name, config) pairs for the 3 ablation variants."""
    # Variant 1: Dense only (disable BM25 by setting bm25_top_k to 0)
    cfg_dense = copy.deepcopy(base_cfg)
    cfg_dense["retriever"]["bm25_top_k"] = 0
    cfg_dense["retriever"]["use_reranker"] = False

    # Variant 2: Hybrid dense + BM25 (default)
    cfg_hybrid = copy.deepcopy(base_cfg)
    cfg_hybrid["retriever"]["use_reranker"] = False

    # Variant 3: Hybrid + Reranker
    cfg_reranker = copy.deepcopy(base_cfg)
    cfg_reranker["retriever"]["use_reranker"] = True

    return [
        ("dense_only",      cfg_dense),
        ("hybrid_bm25",     cfg_hybrid),
        ("hybrid_reranker", cfg_reranker),
    ]


# ── Summary table ──────────────────────────────────────────────────────────────
def print_ablation_table(df: pd.DataFrame):
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    summary = (
        df.groupby("system_variant")[metrics]
        .mean()
        .round(4)
        .reset_index()
    )
    print("\n" + "=" * 70)
    print("ABLATION SUMMARY (mean across all 35 test cases)")
    print("=" * 70)
    print(summary.to_string(index=False))
    print("=" * 70)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(config_path=None):
    base_cfg = load_cfg(config_path)

    # Load evaluation dataset
    eval_path = os.path.join(ROOT, "evaluation", "eval_dataset.json")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    print(f"Loaded {len(eval_data)} evaluation cases from {eval_path}")

    variants = build_variants(base_cfg)
    all_results = []

    for variant_name, cfg in variants:
        results = run_evaluation(cfg, variant_name, eval_data)
        all_results.extend(results)

    # Save to CSV
    df = pd.DataFrame(all_results)
    out_path = os.path.join(ROOT, "evaluation", "results.csv")
    df.to_csv(out_path, index=False)
    print(f"\n✅ Results saved to: {out_path}")

    # Print ablation table
    print_ablation_table(df)

    # Per-disease breakdown
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    print("\nPER-DISEASE BREAKDOWN (hybrid_bm25 variant):")
    hybrid_df = df[df["system_variant"] == "hybrid_bm25"]
    per_disease = hybrid_df.groupby("disease")[metrics].mean().round(4)
    print(per_disease.to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuavaScan RAGAS Evaluation")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()
    main(config_path=args.config)
