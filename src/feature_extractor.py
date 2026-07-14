"""Feature extraction: image -> searchable representation (global embedding + attribute table)."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from taxonomy import COLORS, ALL_GARMENTS, SCENES, STYLES, GLOBAL_VOCAB, VOCAB_INDEX


@dataclass
class ImageFeatures:
    image_id: str
    path: str
    global_emb: np.ndarray | None = None
    attr_scores: dict = field(default_factory=dict)
    scene_scores: dict = field(default_factory=dict)
    style_scores: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "image_id": self.image_id, "path": self.path,
            "global_emb": self.global_emb,
            "attr_scores": {f"{c}|{g}": v for (c, g), v in self.attr_scores.items()},
            "scene_scores": self.scene_scores, "style_scores": self.style_scores,
            "meta": self.meta,
        }


class FeatureExtractor:
    def extract(self, image, labels=None) -> ImageFeatures:
        raise NotImplementedError


class SyntheticExtractor(FeatureExtractor):
    def __init__(self, noise: float = 0.05, seed: int = 0):
        self.noise = noise
        self.rng = np.random.default_rng(seed)

    def extract(self, image, labels=None) -> ImageFeatures:
        labels = labels or {}
        items = labels.get("items", [])
        scene = labels.get("scene")
        style = labels.get("style")

        attr_scores = {(c, g): 0.0 for g in ALL_GARMENTS for c in COLORS}
        for it in items:
            key = (it["color"], it["garment"])
            if key in attr_scores:
                attr_scores[key] = 1.0 + self.rng.normal(0, self.noise)

        scene_scores = {s: 0.0 for s in SCENES}
        if scene in scene_scores:
            scene_scores[scene] = 1.0 + self.rng.normal(0, self.noise)
        style_scores = {s: 0.0 for s in STYLES}
        if style in style_scores:
            style_scores[style] = 1.0 + self.rng.normal(0, self.noise)

        vec = np.zeros(len(GLOBAL_VOCAB), dtype=np.float32)
        for it in items:
            if it["color"] in VOCAB_INDEX:
                vec[VOCAB_INDEX[it["color"]]] += 1.0
            if it["garment"] in VOCAB_INDEX:
                vec[VOCAB_INDEX[it["garment"]]] += 1.0
        if scene in VOCAB_INDEX:
            vec[VOCAB_INDEX[scene]] += 1.0
        if style in VOCAB_INDEX:
            vec[VOCAB_INDEX[style]] += 1.0
        vec += self.rng.normal(0, 0.05, vec.shape).astype(np.float32)
        n = np.linalg.norm(vec)
        if n > 0:
            vec /= n

        return ImageFeatures(
            image_id=labels.get("image_id", "unknown"), path=labels.get("path", ""),
            global_emb=vec, attr_scores=attr_scores,
            scene_scores=scene_scores, style_scores=style_scores, meta=labels)


class CLIPTextEncoder:
    def __init__(self, model_name: str = "ViT-B-32", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import open_clip
        self._model, _, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained="openai")
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode(self, text: str) -> np.ndarray:
        import torch
        self._load()
        toks = self._tokenizer([text]).to(self.device)
        with torch.no_grad():
            emb = self._model.encode_text(toks)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return np.asarray(emb.cpu().numpy()[0], dtype=np.float32)

    def __call__(self, text: str) -> np.ndarray:
        return self.encode(text)


class CLIPExtractor(FeatureExtractor):
    def __init__(self, model_name: str = "ViT-B-32", device: str = "cpu"):
        self.device = device
        self.model_name = model_name
        self._model = None
        self._preprocess = None
        self._tokenizer = None
        self._attr_text = None
        self._scene_text = None
        self._style_text = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        import open_clip
        self._model, self._preprocess, _ = open_clip.create_model_and_transforms(
            self.model_name, pretrained="openai")
        self._model = self._model.to(self.device).eval()
        self._tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.no_grad():
            self._attr_text = self._norm(self._model.encode_text(
                self._tokenizer([f"a photo of a {c} {g}" for g in ALL_GARMENTS
                                 for c in COLORS]).to(self.device))).cpu().numpy()
            self._scene_text = self._norm(self._model.encode_text(
                self._tokenizer([f"a photo taken in a {s}" for s in SCENES]).to(self.device))).cpu().numpy()
            self._style_text = self._norm(self._model.encode_text(
                self._tokenizer([f"a person wearing {s} clothes" for s in STYLES]).to(self.device))).cpu().numpy()

    @staticmethod
    def _norm(x):
        import torch
        x = x / x.norm(dim=-1, keepdim=True)
        return x

    def extract(self, image, labels=None) -> ImageFeatures:
        return self.extract_batch([image], [labels or {}])[0]

    def extract_batch(self, images, labels_list) -> list[ImageFeatures]:
        import torch
        from PIL import Image
        self._load()
        tensors = []
        for img in images:
            im = img if isinstance(img, Image.Image) else Image.fromarray(img)
            tensors.append(self._preprocess(im))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            img_emb = self._model.encode_image(batch)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            img_emb = img_emb.cpu().numpy().astype(np.float32)
            attr = (img_emb @ self._attr_text.T)
            scene = (img_emb @ self._scene_text.T)
            style = (img_emb @ self._style_text.T)

        out = []
        garment_order = [(g, c) for g in ALL_GARMENTS for c in COLORS]
        for i, lbl in enumerate(labels_list):
            attr_scores = {gc: float(attr[i, k]) for k, gc in enumerate(garment_order)}
            scene_scores = {s: float(scene[i, k]) for k, s in enumerate(SCENES)}
            style_scores = {s: float(style[i, k]) for k, s in enumerate(STYLES)}
            out.append(ImageFeatures(
                image_id=lbl.get("image_id", "unknown"),
                path=lbl.get("path", ""),
                global_emb=img_emb[i].astype(np.float32),
                attr_scores=attr_scores, scene_scores=scene_scores,
                style_scores=style_scores, meta=lbl))
        return out
