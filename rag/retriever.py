"""
rag/retriever.py
----------------
Hybrid retriever for GuavaScan RAG pipeline.

Architecture:
  1. Dense retrieval   — ChromaDB cosine similarity (top dense_top_k)
  2. Sparse retrieval  — BM25 keyword search (top bm25_top_k)
  3. Fusion            — Reciprocal Rank Fusion (RRF) → final_top_k chunks
  4. Reranker          — cross-encoder (ONLY loaded if use_reranker: true)

Public API:
  get_retriever(cfg) -> GuavaRetriever
  retriever.retrieve(query: str) -> list[dict]
      Each dict: {"content": str, "source": str, "section": str, "score": float}
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────
def _rrf_merge(dense_docs, sparse_docs, k: int = 60) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    Returns deduplicated list sorted by RRF score descending.
    """
    scores: dict[str, float] = {}
    content_map: dict[str, object] = {}

    for rank, doc in enumerate(dense_docs):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        content_map[key] = doc

    for rank, doc in enumerate(sparse_docs):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        content_map[key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(content_map[k], scores[k]) for k in sorted_keys]


# ── GuavaRetriever class ───────────────────────────────────────────────────────
class GuavaRetriever:
    """
    Hybrid dense + BM25 retriever with optional cross-encoder reranking.
    Instantiated once and reused across Streamlit reruns.
    """

    def __init__(self, cfg: dict):
        self.cfg         = cfg
        self.dense_top_k = cfg["retriever"]["dense_top_k"]
        self.bm25_top_k  = cfg["retriever"]["bm25_top_k"]
        self.final_top_k = cfg["retriever"]["final_top_k"]
        self.use_reranker= cfg["retriever"]["use_reranker"]

        chroma_dir = os.path.join(ROOT, cfg["chroma_persist_dir"].lstrip("./"))
        collection = cfg["collection_name"]

        # ── Embeddings ────────────────────────────────────────────────────────
        self.embeddings = HuggingFaceEmbeddings(
            model_name=cfg["embedding_model"],
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # ── ChromaDB (dense) ──────────────────────────────────────────────────
        self.vectorstore = Chroma(
            persist_directory=chroma_dir,
            collection_name=collection,
            embedding_function=self.embeddings,
        )

        # ── BM25 (sparse) — built from all stored documents ───────────────────
        self._build_bm25()

        # ── Reranker (only loaded when enabled) ───────────────────────────────
        self.reranker = None
        if self.use_reranker:
            from sentence_transformers.cross_encoder import CrossEncoder
            self.reranker = CrossEncoder(cfg["retriever"]["reranker_model"])

    def _build_bm25(self):
        """Build BM25 index from all documents currently in ChromaDB."""
        from rank_bm25 import BM25Okapi

        # Fetch all stored documents
        stored = self.vectorstore.get(include=["documents", "metadatas"])
        texts     = stored["documents"]
        metadatas = stored["metadatas"]

        # Tokenise (simple whitespace)
        tokenised = [t.lower().split() for t in texts]
        self.bm25       = BM25Okapi(tokenised)
        self.bm25_texts = texts
        self.bm25_metas = metadatas

    def _bm25_search(self, query: str, top_k: int):
        """Run BM25 search and return LangChain-compatible Document objects."""
        from langchain.schema import Document

        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            Document(
                page_content=self.bm25_texts[i],
                metadata=self.bm25_metas[i],
            )
            for i in top_idx
        ]

    def retrieve(self, query: str) -> list[dict]:
        """
        Run hybrid retrieval for a query string.
        If bm25_top_k == 0, dense-only retrieval is used (ablation mode).

        Returns:
            list of dicts: [{"content": str, "source": str, "section": str, "score": float}]
        """
        # 1. Dense retrieval
        dense_docs = self.vectorstore.similarity_search(query, k=self.dense_top_k)

        # 2. Sparse retrieval (skip when bm25_top_k == 0 → dense-only ablation)
        if self.bm25_top_k > 0:
            sparse_docs = self._bm25_search(query, self.bm25_top_k)
            merged = _rrf_merge(dense_docs, sparse_docs)
        else:
            # Dense-only: wrap with dummy score for uniform output format
            merged = [(doc, 1.0 / (60 + i + 1)) for i, doc in enumerate(dense_docs)]

        # 3. Take final_top_k
        top_merged = merged[: self.final_top_k]

        # 4. Optional reranking
        if self.use_reranker and self.reranker is not None:
            pairs     = [(query, doc.page_content) for doc, _ in top_merged]
            ce_scores = self.reranker.predict(pairs)
            ranked    = sorted(zip(top_merged, ce_scores), key=lambda x: x[1], reverse=True)
            top_merged = [item[0] for item, _ in ranked]
        
        # 5. Format output
        results = []
        for item in top_merged:
            if isinstance(item, tuple):
                doc, score = item
            else:
                doc, score = item, 0.0
            results.append({
                "content": doc.page_content,
                "source":  doc.metadata.get("source", "unknown"),
                "section": doc.metadata.get("section", "General"),
                "score":   float(score),
            })

        return results


# ── Factory function ───────────────────────────────────────────────────────────
def get_retriever(cfg: dict) -> GuavaRetriever:
    """
    Instantiate and return a GuavaRetriever.
    Call once and cache the result (e.g. with @st.cache_resource).
    """
    return GuavaRetriever(cfg)
