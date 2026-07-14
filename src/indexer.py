"""Part A -- the indexer: streams the dataset, extracts features, builds the FAISS + attribute store."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from feature_extractor import FeatureExtractor, SyntheticExtractor
from store import VectorStore


class Indexer:
    def __init__(self, extractor: FeatureExtractor, index_dir):
        self.extractor = extractor
        self.index_dir = Path(index_dir)
        self.store = None

    def build(self, image_dir, labels_path=None, batch_size: int = 64):
        image_dir = Path(image_dir)
        labels = self._load_labels(labels_path)
        use_batch = hasattr(self.extractor, "extract_batch")

        embs, metas, dim = [], [], None
        paths = sorted(p for p in image_dir.rglob("*")
                        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
        from PIL import Image
        if use_batch:
            buf_imgs, buf_lbls = [], []
            for p in paths:
                lbl = labels.get(p.stem, {}) if labels else {}
                lbl = dict(lbl)
                lbl["image_id"] = lbl.get("image_id", p.stem)
                lbl["path"] = str(p)
                buf_imgs.append(np.asarray(Image.open(p).convert("RGB")))
                buf_lbls.append(lbl)
                if len(buf_imgs) >= batch_size:
                    self._ingest(self.extractor.extract_batch(buf_imgs, buf_lbls),
                                embs, metas, dim)
                    dim = dim or (embs[-1].shape[0] if embs else None)
                    buf_imgs, buf_lbls = [], []
            if buf_imgs:
                self._ingest(self.extractor.extract_batch(buf_imgs, buf_lbls),
                             embs, metas, dim)
        else:
            for p in paths:
                lbl = labels.get(p.stem, {}) if labels else {}
                lbl = dict(lbl)
                lbl["image_id"] = lbl.get("image_id", p.stem)
                lbl["path"] = str(p)
                feats = self.extractor.extract(
                    np.asarray(Image.open(p).convert("RGB")), lbl)
                if feats.global_emb is None:
                    continue
                if dim is None:
                    dim = feats.global_emb.shape[0]
                self._append(feats, embs, metas)

        if embs:
            self._flush(embs, metas, dim)
        self.store.save(self.index_dir)
        print(f"[indexer] indexed {self.store.n} images -> {self.index_dir}")
        return self.store

    def _ingest(self, feats_list, embs, metas, dim):
        for feats in feats_list:
            if feats.global_emb is None:
                continue
            if dim is None:
                dim = feats.global_emb.shape[0]
            self._append(feats, embs, metas)

    @staticmethod
    def _append(feats, embs, metas):
        embs.append(feats.global_emb.astype(np.float32))
        metas.append({
            "image_id": feats.image_id, "path": feats.path,
            "attr_scores": feats.to_dict()["attr_scores"],
            "scene_scores": feats.scene_scores,
            "style_scores": feats.style_scores, "meta": feats.meta,
        })

    def _flush(self, embs, metas, dim):
        if self.store is None:
            self.store = VectorStore(dim)
        self.store.add(np.stack(embs), metas)

    @staticmethod
    def _load_labels(labels_path):
        if not labels_path:
            return None
        with open(labels_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return {d.get("image_id", d.get("id", str(i))): d
                    for i, d in enumerate(data)}
        return data
