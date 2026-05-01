import os
import pickle
import json
import re
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

INDEX_DIR = Path(__file__).parent.parent / 'data' / 'index'
INDEX_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIRS = {
    'claude':      Path(__file__).parent.parent / 'data' / 'claude',
    'hackerrank':  Path(__file__).parent.parent / 'data' / 'hackerrank',
    'visa':        Path(__file__).parent.parent / 'data' / 'visa',
}

MODEL_NAME  = 'all-MiniLM-L6-v2'
CHUNK_SIZE  = 400   # words
OVERLAP     = 50


def clean_text(text: str) -> str:
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)  # markdown links -> text
    text = re.sub(r'[#*`>_~]', '', text)                   # markdown symbols
    text = re.sub(r'(accept all cookies|privacy policy|skip to content|back to top)',
                  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def chunk(text: str) -> list[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(' '.join(words[start:end]))
        start += CHUNK_SIZE - OVERLAP
    return chunks


def load_markdown_files() -> list[dict]:
    all_chunks = []
    for source, directory in DATA_DIRS.items():
        if not directory.exists():
            print(f"  Warning: {directory} not found")
            continue
        files = list(directory.rglob('*.md'))
        print(f"  {source}: {len(files)} markdown files")
        for fpath in files:
            try:
                raw = fpath.read_text(encoding='utf-8')
            except Exception:
                continue
            cleaned = clean_text(raw)
            title   = fpath.stem.replace('-', ' ').replace('_', ' ')
            for c in chunk(cleaned):
                all_chunks.append({
                    'text':   c,
                    'title':  title,
                    'source': source,
                    'url':    str(fpath)
                })
    return all_chunks


def build_index():
    print("Loading markdown files...")
    chunks = load_markdown_files()
    print(f"  Total chunks: {len(chunks)}")

    texts = [c['text'] for c in chunks]

    # --- BM25 ---
    print("Building BM25 index...")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)
    with open(INDEX_DIR / 'bm25.pkl', 'wb') as f:
        pickle.dump(bm25, f)

    # --- Dense (FAISS) ---
    print(f"Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        texts, batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / 'index.faiss'))

    # --- Shared chunks metadata ---
    with open(INDEX_DIR / 'chunks.pkl', 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Done. Indexes saved to {INDEX_DIR}")


if __name__ == '__main__':
    build_index()