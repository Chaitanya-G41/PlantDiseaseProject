"""
rag/ingest.py
-------------
Knowledge Base Ingestion Pipeline for GuavaScan.

Loads all .md files from knowledge_base/, chunks them, embeds with
sentence-transformers, and persists to ChromaDB.

Hash guard: only re-indexes when KB content has changed.

Key improvements over v1:
  - Raw text loading (preserves ## headings — UnstructuredMarkdownLoader stripped them)
  - Section metadata attached by scanning backwards in source doc (not just chunk text)
  - chunk_size/overlap read from config (600/80 recommended)

Run:
    python rag/ingest.py
    python rag/ingest.py --force   # force re-index even if hash matches
"""

import os
import sys
import glob
import hashlib
import argparse
import re

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Hash guard ─────────────────────────────────────────────────────────────────
def compute_kb_hash(kb_dir: str) -> str:
    """Compute a combined MD5 hash of all .md files in the KB directory."""
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    hasher = hashlib.md5()
    for filepath in md_files:
        with open(filepath, "rb") as f:
            hasher.update(f.read())
    return hasher.hexdigest()


def get_stored_hash(chroma_dir: str) -> str | None:
    hash_file = os.path.join(chroma_dir, ".index_hash")
    if os.path.exists(hash_file):
        with open(hash_file, "r") as f:
            return f.read().strip()
    return None


def store_hash(chroma_dir: str, hash_val: str):
    os.makedirs(chroma_dir, exist_ok=True)
    hash_file = os.path.join(chroma_dir, ".index_hash")
    with open(hash_file, "w") as f:
        f.write(hash_val)


# ── Section metadata ───────────────────────────────────────────────────────────
def attach_section_metadata(chunks: list, full_texts: dict) -> list:
    """
    For each chunk, find its nearest preceding ## heading in the source document
    by scanning backwards from where the chunk appears in the full text.

    This is far more accurate than extract_nearest_section(chunk.page_content)
    because ## headings are preserved in the raw text but often absent in a chunk
    that starts mid-section.

    Args:
        chunks:     List of LangChain Document objects post-splitting
        full_texts: {source_filename: full_raw_text} lookup built from md_files
    """
    for chunk in chunks:
        source     = chunk.metadata.get("source", "")
        full_text  = full_texts.get(source, "")

        if not full_text:
            chunk.metadata["section"] = "General"
            continue

        # Find where this chunk starts in the source document
        # Use first 80 chars as a fingerprint (avoids false matches on short chunks)
        fingerprint = chunk.page_content[:80].strip()
        chunk_start = full_text.find(fingerprint)

        if chunk_start == -1:
            # Fallback: scan chunk text itself for any ## heading
            headings = re.findall(r"^##\s+(.+)$", chunk.page_content, re.MULTILINE)
            chunk.metadata["section"] = headings[-1].strip() if headings else "General"
            continue

        # Scan backwards from chunk_start — find the last ## heading before it
        preceding  = full_text[:chunk_start]
        headings   = re.findall(r"^##\s+(.+)$", preceding, re.MULTILINE)
        chunk.metadata["section"] = headings[-1].strip() if headings else "General"

    return chunks


# ── Main ingestion ─────────────────────────────────────────────────────────────
def ingest(config_path: str = None, force: bool = False):
    cfg = load_config(config_path)

    kb_dir        = os.path.join(ROOT, cfg["knowledge_base_dir"].lstrip("./"))
    chroma_dir    = os.path.join(ROOT, cfg["chroma_persist_dir"].lstrip("./"))
    chunk_size    = cfg["chunk_size"]
    chunk_overlap = cfg["chunk_overlap"]
    embed_model   = cfg["embedding_model"]
    collection    = cfg["collection_name"]

    # ── Hash guard ─────────────────────────────────────────────────────────────
    current_hash = compute_kb_hash(kb_dir)
    stored_hash  = get_stored_hash(chroma_dir)

    if not force and stored_hash == current_hash and os.path.exists(chroma_dir):
        print(f"[ingest] KB unchanged (hash: {current_hash[:8]}…). Skipping re-index.")
        print("[ingest] Use --force to re-index anyway.")
        return

    print(f"[ingest] KB hash: {current_hash[:8]}…  Indexing...")

    # ── Load markdown files as raw text (preserves ## headings) ───────────────
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in: {kb_dir}")

    print(f"[ingest] Loading {len(md_files)} documents (raw text mode)...")

    raw_docs   = []
    full_texts = {}   # {source_filename: full_raw_text} — used for section lookup

    for filepath in md_files:
        source_name  = os.path.basename(filepath)
        disease_name = os.path.splitext(source_name)[0]

        with open(filepath, "r", encoding="utf-8") as f:
            raw_text = f.read()

        full_texts[source_name] = raw_text   # store for section metadata later

        doc = Document(
            page_content=raw_text,
            metadata={"source": source_name, "disease": disease_name},
        )
        raw_docs.append(doc)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    # Separators prioritise section boundaries first, then paragraphs, then lines
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # ── Attach section metadata (backwards-scan method) ───────────────────────
    chunks = attach_section_metadata(chunks, full_texts)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"[ingest] Total chunks: {len(chunks)}")
    for src in sorted({c.metadata["source"] for c in chunks}):
        n        = sum(1 for c in chunks if c.metadata["source"] == src)
        sections = {c.metadata["section"] for c in chunks if c.metadata["source"] == src}
        print(f"         {src}: {n} chunks | sections: {sorted(sections)}")

    # ── Embed + store ─────────────────────────────────────────────────────────
    print(f"[ingest] Embedding with: {embed_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Clear existing ChromaDB before re-indexing to avoid duplicate chunks
    if os.path.exists(chroma_dir):
        import shutil
        try:
            shutil.rmtree(chroma_dir)
        except PermissionError:
            import time
            time.sleep(1)
            shutil.rmtree(chroma_dir, ignore_errors=True)
        os.makedirs(chroma_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=chroma_dir,
        collection_name=collection,
    )
    # Chroma 0.4+ auto-persists — no manual .persist() call needed

    # ── Store hash ────────────────────────────────────────────────────────────
    store_hash(chroma_dir, current_hash)
    print(f"[ingest] ✓ Indexed {len(chunks)} chunks → ChromaDB at: {chroma_dir}")
    return len(chunks)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuavaScan KB Ingestion")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force",  action="store_true", help="Force re-index")
    args = parser.parse_args()
    ingest(config_path=args.config, force=args.force)