"""Build the search index from the dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generator import generate
from feature_extractor import SyntheticExtractor, CLIPExtractor
from indexer import Indexer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--images-dir", default=None,
                    help="explicit image folder (e.g. data/fashion2020/test)")
    ap.add_argument("--labels", default=None, help="optional labels.json")
    ap.add_argument("--index", default="index")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--backend", choices=["synthetic", "clip"], default="synthetic")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.backend == "clip" or args.images_dir:
        img_dir = Path(args.images_dir) if args.images_dir else Path(args.data) / "images"
        lbl_path = args.labels
        extractor = CLIPExtractor() if args.backend == "clip" else \
            SyntheticExtractor(noise=0.05, seed=args.seed)
        Indexer(extractor, args.index).build(img_dir, lbl_path)
        print("[build_index] done.")
        return

    data_dir = Path(args.data)
    if not (data_dir / "labels.json").exists():
        print("[build_index] labels.json missing -> generating synthetic dataset")
        generate(args.n, data_dir, seed=args.seed)
    extractor = SyntheticExtractor(noise=0.05, seed=args.seed)
    Indexer(extractor, args.index).build(data_dir / "images", data_dir / "labels.json")
    print("[build_index] done.")


if __name__ == "__main__":
    main()
