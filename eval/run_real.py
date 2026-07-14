"""Run the 5 assessment queries on the real FashionPedia index (qualitative; no labels)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generator import EVAL_QUERIES
from store import VectorStore
from retriever import Retriever
from feature_extractor import CLIPTextEncoder

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "fashion2020" / "test"
EX_DIR = ROOT / "docs" / "examples"
EX_DIR.mkdir(parents=True, exist_ok=True)


def slug(q: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in q)[:40]


def main():
    store = VectorStore.load(ROOT / "index_real")
    te = CLIPTextEncoder()
    retr = Retriever(store, recall_k=store.n, text_encoder=te)
    out = []
    for q in EVAL_QUERIES:
        van, _ = retr.search(q, top_k=5, mode="vanilla")
        comp, _ = retr.search(q, top_k=5, mode="compositional")
        s = slug(q)
        for mode, res in (("vanilla", van), ("compositional", comp)):
            for i, r in enumerate(res[:3], 1):
                src = IMG_DIR / f"{r.image_id}.jpg"
                if src.exists():
                    shutil.copy(src, EX_DIR / f"{s}_{mode}_{i}.jpg")
        out.append({
            "query": q,
            "vanilla_top5": [r.image_id for r in van],
            "compositional_top5": [r.image_id for r in comp],
        })
        print(f"\n### {q}")
        print("  vanilla      :", [r.image_id for r in van])
        print("  compositional:", [r.image_id for r in comp])

    (ROOT / "eval" / "real_results.json").write_text(json.dumps(out, indent=2))
    print("\n[real] wrote eval/real_results.json and docs/examples/")


if __name__ == "__main__":
    main()
