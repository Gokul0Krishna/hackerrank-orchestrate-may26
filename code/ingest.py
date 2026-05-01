import json
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CORPUS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'corpus')
INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'index')
os.makedirs(INDEX_DIR, exist_ok=True)

MODEL_NAME = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 400   # words
CHUNK_OVERLAP = 50


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append(' '.join(words[start:end]))
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def load_corpus() -> list[dict]:
    docs = []
    for fname in ['hackerrank.json', 'claude.json', 'visa.json']:
        path = os.path.join(CORPUS_DIR, fname)
        if not os.path.exists(path):
            print(f"  Warning: {path} not found — run scraper.py first")
            continue
        with open(path) as f:
            articles = json.load(f)
        docs.extend(articles)
        print(f"  Loaded {len(articles)} docs from {fname}")
    return docs


def build_index():
    print("Loading corpus...")
    docs = load_corpus()

    print("Chunking documents...")
    chunks = []
    for doc in docs:
        combined = f"{doc['title']}\n\n{doc['body']}"
        for chunk in chunk_text(combined):
            chunks.append({
                'text': chunk,
                'title': doc['title'],
                'url': doc.get('url', ''),
                'source': doc['source']
            })
    print(f"  Total chunks: {len(chunks)}")

    print(f"Embedding with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [c['text'] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True
    ).astype(np.float32)

    print("Building FAISS index (cosine similarity via inner product)...")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(INDEX_DIR, 'index.faiss'))
    with open(os.path.join(INDEX_DIR, 'chunks.pkl'), 'wb') as f:
        pickle.dump(chunks, f)

    print(f"Index saved to {INDEX_DIR}")
    print(f"  {len(chunks)} chunks | dim={dim}")


if __name__ == '__main__':
    build_index()