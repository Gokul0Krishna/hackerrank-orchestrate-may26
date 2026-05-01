import os
import pickle
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

INDEX_DIR  = Path(__file__).parent.parent / 'data' / 'index'
MODEL_NAME = 'all-MiniLM-L6-v2'

SOURCE_MAP = {'hackerrank': 'hackerrank', 'claude': 'claude', 'visa': 'visa'}


class Retriever:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)
        self.index = faiss.read_index(str(INDEX_DIR / 'index.faiss'))
        with open(INDEX_DIR / 'chunks.pkl', 'rb') as f:
            self.chunks = pickle.load(f)
        with open(INDEX_DIR / 'bm25.pkl', 'rb') as f:
            self.bm25 = pickle.load(f)

    def retrieve(self, query: str, company: str = 'None', top_k: int = 5) -> list[dict]:
        expected_source = SOURCE_MAP.get(company.lower()) if company.lower() != 'none' else None
        fetch_k = top_k * 6

        # --- Dense scores ---
        qemb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        dense_scores, dense_ids = self.index.search(qemb, fetch_k)
        dense_scores = dense_scores[0]
        dense_ids    = dense_ids[0]

        # Normalize dense scores to [0, 1]
        d_min, d_max = dense_scores.min(), dense_scores.max()
        if d_max > d_min:
            dense_norm = (dense_scores - d_min) / (d_max - d_min)
        else:
            dense_norm = np.ones_like(dense_scores)

        # --- BM25 scores ---
        bm25_scores = self.bm25.get_scores(query.lower().split())
        b_min, b_max = bm25_scores.min(), bm25_scores.max()
        if b_max > b_min:
            bm25_norm = (bm25_scores - b_min) / (b_max - b_min)
        else:
            bm25_norm = np.zeros_like(bm25_scores)

        # --- Fuse: 60% dense + 40% BM25 ---
        ALPHA = 0.6
        combined = {}
        for norm_score, idx in zip(dense_norm, dense_ids):
            if idx == -1:
                continue
            combined[idx] = ALPHA * norm_score + (1 - ALPHA) * bm25_norm[idx]

        sorted_ids = sorted(combined, key=combined.get, reverse=True)

        results = []
        seen = set()
        for idx in sorted_ids:
            chunk = self.chunks[idx]
            if expected_source and chunk['source'] != expected_source:
                continue
            key = chunk['text'][:120]
            if key in seen:
                continue
            seen.add(key)
            results.append({**chunk, 'score': combined[idx]})
            if len(results) >= top_k:
                break

        # Fallback: if company filter returns nothing, go global
        if not results and expected_source:
            return self.retrieve(query, company='None', top_k=top_k)

        return results

    def has_sufficient_coverage(self, results: list[dict], threshold: float = 0.35) -> bool:
        if not results:
            return False
        return results[0]['score'] >= threshold