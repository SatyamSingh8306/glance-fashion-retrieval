"""CLI search.

Usage:
    python scripts/search.py "A red tie and a white shirt in a formal setting" --k 10
    python scripts/search.py "..." --index index_real --clip
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from store import VectorStore
from retriever import Retriever
from feature_extractor import CLIPTextEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--index", default="index")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--mode", choices=["vanilla", "compositional"],
                    default="compositional")
    ap.add_argument("--clip", action="store_true",
                    help="query an index built with CLIP (needs CLIP text encoder)")
    args = ap.parse_args()

    store = VectorStore.load(args.index)
    te = CLIPTextEncoder() if args.clip else None
    retriever = Retriever(store, recall_k=200, text_encoder=te)
    results, pq = retriever.search(args.query, top_k=args.k, mode=args.mode)

    print(f"Parsed : {pq}")
    print(f"Mode   : {args.mode}")
    print(f"{'rank':>4}  {'score':>7}  {'attr':>6}  {'scene':>6}  {'style':>6}  image_id")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:>4}  {r.final_score:7.3f}  {r.attr_score:6.3f}  "
              f"{r.scene_score:6.3f}  {r.style_score:6.3f}  {r.image_id}")


if __name__ == "__main__":
    main()
