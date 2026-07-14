"""Vector storage: FAISS for the global embedding, side store for attributes."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False


class VectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        if _HAS_FAISS:
            self.index = faiss.IndexFlatIP(dim)
        else:
            self.index = None
            self._matrix = None
        self.meta = []

    @property
    def n(self) -> int:
        return len(self.meta)

    def add(self, embeddings, metas):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings[None, :]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-9)
        if _HAS_FAISS:
            self.index.add(embeddings)
        else:
            self._matrix = embeddings if self._matrix is None else np.vstack(
                [self._matrix, embeddings])
        self.meta.extend(metas)

    def search(self, query_vec, k: int = 20):
        query_vec = np.asarray(query_vec, dtype=np.float32).reshape(1, -1)
        query_vec /= (np.linalg.norm(query_vec) + 1e-9)
        k = min(k, self.n)
        if _HAS_FAISS:
            scores, idxs = self.index.search(query_vec, k)
            scores, idxs = scores[0], idxs[0]
        else:
            sims = self._matrix @ query_vec.ravel()
            order = np.argsort(-sims)[:k]
            scores, idxs = sims[order], order
        results = []
        for s, i in zip(scores, idxs):
            if i < 0:
                continue
            m = dict(self.meta[i])
            m["score"] = float(s)
            results.append(m)
        return results

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if _HAS_FAISS:
            faiss.write_index(self.index, str(path / "vectors.faiss"))
        else:
            np.save(path / "vectors.npy", self._matrix)
        with open(path / "meta.pkl", "wb") as f:
            pickle.dump(self.meta, f)
        (path / "dim.txt").write_text(str(self.dim))

    @classmethod
    def load(cls, path):
        path = Path(path)
        dim = int((path / "dim.txt").read_text().strip())
        store = cls(dim)
        if (path / "vectors.faiss").exists():
            store.index = faiss.read_index(str(path / "vectors.faiss"))
        else:
            store._matrix = np.load(path / "vectors.npy")
        with open(path / "meta.pkl", "rb") as f:
            store.meta = pickle.load(f)
        return store
