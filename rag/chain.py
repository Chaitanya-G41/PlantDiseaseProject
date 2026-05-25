"""
rag/chain.py
------------
LangChain LCEL RAG chain for GuavaScan.

Public API:
    build_rag_chain(cfg) -> callable
    run_rag_chain(predicted_class, confidence, cfg, retriever) -> dict
        Returns: {"answer": str, "sources": list[dict]}

The chain is stateless w.r.t. the ViT model — it only receives a string
(predicted class) and a float (confidence).
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from dotenv import load_dotenv

load_dotenv(os.path.join(ROOT, ".env"))



print("=== ENV DEBUG ===")
print(f"ROOT path: {ROOT}")
print(f"ENV file exists: {os.path.exists(os.path.join(ROOT, '.env'))}")
key = os.getenv("GEMINI_API_KEY", "NOT FOUND")
print(f"Key loaded: {key[:10]}..." if key != "NOT FOUND" else "Key: NOT FOUND")
print("=================")

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser

from rag.prompts import DISEASE_PROMPT, HEALTHY_PROMPT
from rag.retriever import GuavaRetriever


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Context formatter ─────────────────────────────────────────────────────────
def _format_context(retrieved_chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context string for the prompt.
    Each chunk is prefixed with its source file and section.
    """
    parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        header = f"[Source {i}: {chunk['source']} — {chunk['section']}]"
        parts.append(f"{header}\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


# ── Query builder ─────────────────────────────────────────────────────────────
def _build_query(predicted_class: str, cfg: dict) -> str:
    """
    Build a rich retrieval query from the predicted class name.
    Uses class_query_map from config.yaml — no hardcoding in Python.
    """
    query_map = cfg.get("class_query_map", {})
    return query_map.get(
        predicted_class,
        f"guava {predicted_class} disease symptoms treatment management control",
    )


# ── Main run function ─────────────────────────────────────────────────────────
def run_rag_chain(
    predicted_class: str,
    confidence: float,
    cfg: dict,
    retriever: GuavaRetriever,
) -> dict:
    """
    Run the full RAG chain for a given prediction.

    Args:
        predicted_class: Model output class name (e.g. "Guava_anthracnose")
        confidence:      Model confidence 0.0–1.0
        cfg:             Loaded config dict
        retriever:       Initialised GuavaRetriever instance

    Returns:
        {
            "answer":  str,           # LLM-generated advisory text
            "sources": list[dict],    # [{"file": str, "section": str}]
        }
    """
    api_key = os.getenv("GEMINI_API_KEY")
    print(f">>> KEY IS: '{api_key}'")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    # ── Build retrieval query ─────────────────────────────────────────────────
    query = _build_query(predicted_class, cfg)

    # ── Retrieve ──────────────────────────────────────────────────────────────
    chunks = retriever.retrieve(query)
    context = _format_context(chunks)

    # ── Select prompt ─────────────────────────────────────────────────────────
    healthy_class = cfg.get("healthy_class", "Guava_healthy")
    is_healthy = predicted_class == healthy_class

    # ── Build LLM ─────────────────────────────────────────────────────────────
    llm = ChatGoogleGenerativeAI(
        model=cfg["llm_model"],
        google_api_key=api_key,
        temperature=0.2,
        max_output_tokens=2048,
    )

    # ── LCEL chain ────────────────────────────────────────────────────────────
    parser = StrOutputParser()

    if is_healthy:
        prompt_value = HEALTHY_PROMPT.format(
            confidence=confidence * 100,
            context=context,
        )
    else:
        display_name = predicted_class.replace("Guava_", "").replace("_", " ").title()
        prompt_value = DISEASE_PROMPT.format(
            disease_name=display_name,
            confidence=confidence * 100,
            context=context,
        )

    # Run chain
    from langchain_core.messages import HumanMessage
    response = llm.invoke([HumanMessage(content=prompt_value)])
    answer = parser.invoke(response)

    # ── Build sources list ────────────────────────────────────────────────────
    sources = [
        {"file": c["source"], "section": c["section"]}
        for c in chunks
    ]
    # Deduplicate sources
    seen = set()
    unique_sources = []
    for s in sources:
        key = (s["file"], s["section"])
        if key not in seen:
            seen.add(key)
            unique_sources.append(s)

    return {
        "answer":  answer,
        "sources": unique_sources,
    }
