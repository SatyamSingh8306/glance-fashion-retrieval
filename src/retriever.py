"""Part B -- the retriever. Modes: "vanilla" (CLIP/FAISS) and "compositional" (binding + scene/style)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from query_parser import ParsedQuery, get_parser
from taxonomy import COLORS, ALL_GARMENTS, SCENES, STYLES, GLOBAL_VOCAB, VOCAB_INDEX

EPS = 1e-6


@dataclass
class SearchResult:
    image_id: str
    path: str
    global_score: float
    attr_score: float
    scene_score: float
    style_score: float
    final_score: float
    meta: dict
    attr_table: dict | None = None


class Retriever:
    def __init__(self, store, parser=None, recall_k: int = 200,
                 weights: dict | None = None, text_encoder=None):
        self.store = store
        self.parser = parser or get_parser()
        self.recall_k = recall_k
        self.weights = weights or {"global": 0.4, "attr": 0.4,
                                   "scene": 0.1, "style": 0.1}
        self.text_encoder = text_encoder

    def _query_global_vector(self, q: ParsedQuery) -> np.ndarray:
        if self.text_encoder is not None:
            return self.text_encoder(q.raw)
        vec = np.zeros(len(GLOBAL_VOCAB), dtype=np.float32)
        for b in q.bindings:
            if b["color"] in VOCAB_INDEX:
                vec[VOCAB_INDEX[b["color"]]] += 1.0
            if b["garment"] in VOCAB_INDEX:
                vec[VOCAB_INDEX[b["garment"]]] += 1.0
        if q.scene and q.scene in VOCAB_INDEX:
            vec[VOCAB_INDEX[q.scene]] += 1.0
        if q.style and q.style in VOCAB_INDEX:
            vec[VOCAB_INDEX[q.style]] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        return vec

    @staticmethod
    def _attr_score(bindings, meta) -> float:
        attr = meta.get("attr_scores", {})
        if not bindings:
            return 1.0
        scores = []
        for b in bindings:
            c, g = b["color"], b["garment"]
            this = attr.get(f"{c}|{g}", 0.0)
            best = max((v for (kk, v) in attr.items() if kk.endswith(f"|{g}")),
                       default=0.0)
            scores.append(this / (best + EPS))
        return float(np.mean(scores))

    @staticmethod
    def _scene_score(scene, meta) -> float:
        if not scene:
            return 0.5
        sc = meta.get("scene_scores", {})
        if not sc:
            return 0.5
        vals = np.array(list(sc.values()), dtype=float)
        logits = vals - vals.max()
        probs = np.exp(logits) / (np.exp(logits).sum() + EPS)
        return float(probs[list(sc.keys()).index(scene)])

    @staticmethod
    def _style_score(style, meta) -> float:
        if not style:
            return 0.5
        st = meta.get("style_scores", {})
        if not st:
            return 0.5
        vals = np.array(list(st.values()), dtype=float)
        logits = vals - vals.max()
        probs = np.exp(logits) / (np.exp(logits).sum() + EPS)
        return float(probs[list(st.keys()).index(style)])

    def search(self, text, top_k=10, mode="compositional", recall_k=None):
        q = self.parser.parse(text)
        qvec = self._query_global_vector(q)
        candidates = self.store.search(qvec, k=recall_k or self.recall_k)
        if not candidates:
            return [], q

        w = self.weights
        results = []
        for c in candidates:
            g = c["score"]
            a = self._attr_score(q.bindings, c)
            s = self._scene_score(q.scene, c)
            st = self._style_score(q.style, c)
            final = g if mode == "vanilla" else (
                w["global"] * g + w["attr"] * a + w["scene"] * s + w["style"] * st)
            results.append(SearchResult(
                image_id=c["image_id"], path=c["path"], global_score=float(g),
                attr_score=a, scene_score=s, style_score=st,
                final_score=float(final), meta=c.get("meta", {}),
                attr_table=c.get("attr_scores")))

        key = "global_score" if mode == "vanilla" else "final_score"
        results.sort(key=lambda r: getattr(r, key), reverse=True)
        return results[:top_k], q
