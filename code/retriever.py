import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

INDEX_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'index')
MODEL_NAME = 'all-MiniLM-L6-v2'

SOURCE_MAP = {
    'hackerrank': 'hackerrank',
    'claude': 'claude',
    'visa': 'visa',
}


class Retriever:
    def __init__(self):
        index_path = os.path.join(INDEX_DIR, 'index.faiss')
        chunks_path = os.path.join(INDEX_DIR, 'chunks.pkl')

        if not os.path.exists(index_path):
            raise FileNotFoundError("FAISS index not found. Run ingest.py first.")

        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(index_path)
        with open(chunks_path, 'rb') as f:
            self.chunks = pickle.load(f)

    def retrieve(self, query: str, company: str = 'None', top_k: int = 5) -> list[dict]:
        query_emb = self.model.encode(
            [query], normalize_embeddings=True
        ).astype(np.float32)

        # Over-fetch so we can filter by company without running dry
        fetch_k = top_k * 4
        scores, indices = self.index.search(query_emb, fetch_k)

        expected_source = SOURCE_MAP.get(company.lower()) if company and company.lower() != 'none' else None

        results = []
        seen = set()

        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]

            # Filter by company if known
            if expected_source and chunk['source'] != expected_source:
                continue

            # Deduplicate on first 120 chars
            key = chunk['text'][:120]
            if key in seen:
                continue
            seen.add(key)

            results.append({**chunk, 'score': float(score)})
            if len(results) >= top_k:
                break

        # If company-filtered results are empty, fall back to global retrieval
        if not results and expected_source:
            return self.retrieve(query, company='None', top_k=top_k)

        return results