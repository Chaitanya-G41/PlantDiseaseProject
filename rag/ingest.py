"""
rag/ingest.py
-------------
Knowledge Base Ingestion Pipeline for GuavaScan.

Loads all .md files from knowledge_base/, chunks them, embeds with
sentence-transformers, and persists to ChromaDB.

Hash guard: only re-indexes when KB content has changed.

Run:
    python rag/ingest.py
    python rag/ingest.py --force   # force re-index even if hash matches
"""

import os
import sys
import glob
import json
import hashlib
import argparse
import re

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import yaml
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


# ── Config loader ──────────────────────────────────────────────────────────────
def load_config(path: str = None) -> dict:
    if path is None:
        path = os.path.join(ROOT, "config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── Section extractor ──────────────────────────────────────────────────────────
def extract_nearest_section(text: str) -> str:
    """
    Find the last ## heading that appears before the chunk text.
    Returns the heading text, or 'General' if none found.
    """
    headings = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    return headings[-1].strip() if headings else "General"


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


# ── Main ingestion ─────────────────────────────────────────────────────────────
def ingest(config_path: str = None, force: bool = False):
    cfg = load_config(config_path)

    kb_dir       = os.path.join(ROOT, cfg["knowledge_base_dir"].lstrip("./"))
    chroma_dir   = os.path.join(ROOT, cfg["chroma_persist_dir"].lstrip("./"))
    chunk_size   = cfg["chunk_size"]
    chunk_overlap= cfg["chunk_overlap"]
    embed_model  = cfg["embedding_model"]
    collection   = cfg["collection_name"]

    # ── Hash guard ─────────────────────────────────────────────────────────────
    current_hash = compute_kb_hash(kb_dir)
    stored_hash  = get_stored_hash(chroma_dir)

    if not force and stored_hash == current_hash and os.path.exists(chroma_dir):
        print(f"[ingest] KB unchanged (hash: {current_hash[:8]}…). Skipping re-index.")
        print("[ingest] Use --force to re-index anyway.")
        return

    print(f"[ingest] KB hash: {current_hash[:8]}…  Indexing...")

    # ── Load markdown files ────────────────────────────────────────────────────
    md_files = sorted(glob.glob(os.path.join(kb_dir, "*.md")))
    if not md_files:
        raise FileNotFoundError(f"No .md files found in: {kb_dir}")

    print(f"[ingest] Loading {len(md_files)} documents...")
    raw_docs = []
    for filepath in md_files:
        loader = UnstructuredMarkdownLoader(filepath, mode="single")
        docs = loader.load()
        source_name = os.path.basename(filepath)
        for doc in docs:
            doc.metadata["source"] = source_name
            doc.metadata["disease"] = os.path.splitext(source_name)[0]
        raw_docs.extend(docs)

    # ── Chunk ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)

    # Attach section metadata to every chunk
    for chunk in chunks:
        chunk.metadata["section"] = extract_nearest_section(chunk.page_content)

    print(f"[ingest] Total chunks: {len(chunks)}")
    for src in sorted({c.metadata['source'] for c in chunks}):
        n = sum(1 for c in chunks if c.metadata['source'] == src)
        print(f"         {src}: {n} chunks")

    # ── Embed + store ─────────────────────────────────────────────────────────
    print(f"[ingest] Embedding with: {embed_model}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Clear existing ChromaDB if re-indexing
    if os.path.exists(chroma_dir):
        import shutil
        try:
            shutil.rmtree(chroma_dir)
        except PermissionError:
            # Windows may lock SQLite files — try individual removal
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
    # Chroma 0.4+ auto-persists — no manual persist() call needed

    # ── Store hash ────────────────────────────────────────────────────────────
    store_hash(chroma_dir, current_hash)
    print(f"[ingest] DONE. Indexed {len(chunks)} chunks into ChromaDB at: {chroma_dir}")
    return len(chunks)


# ── CLI entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GuavaScan KB Ingestion")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--force", action="store_true", help="Force re-index")
    args = parser.parse_args()
    ingest(config_path=args.config, force=args.force)
