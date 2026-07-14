"""End-to-end evaluation: recall@k (vanilla vs compositional) + compositionality binding probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_generator import EVAL_QUERIES, build_eval_ground_truth, SWAP_PAIRS
from store import VectorStore
from retriever import Retriever


def recall_at_k(ranked_ids, relevant, k):
    if not relevant:
        return 0.0
    return len(set(ranked_ids[:k]) & set(relevant)) / len(relevant)


def main():
    root = Path(__file__).resolve().parent.parent
    store = VectorStore.load(root / "index")
    labels = json.load(open(root / "data" / "labels.json"))
    gt = build_eval_ground_truth(labels)

    ks = [1, 3, 5, 10]
    print("=" * 90)
    print(" (A) ASSESSMENT QUERY RECALL@k   vanilla | compositional")
    print("=" * 90)
    print(f"{'query':48} | #rel |  v@k      |  c@k")
    print("-" * 90)
    van, comp = {k: [] for k in ks}, {k: [] for k in ks}
    detail = []
    for q in EVAL_QUERIES:
        rel = gt[q]
        rv = Retriever(store, recall_k=store.n).search(q, top_k=max(ks),
                                                       mode="vanilla")[0]
        rc = Retriever(store, recall_k=store.n).search(q, top_k=max(ks),
                                                       mode="compositional")[0]
        rv_ids = [r.image_id for r in rv]
        rc_ids = [r.image_id for r in rc]
        v = [round(recall_at_k(rv_ids, rel, k), 2) for k in ks]
        c = [round(recall_at_k(rc_ids, rel, k), 2) for k in ks]
        for d, acc in ((v, van), (c, comp)):
            for k, val in zip(ks, d):
                acc[k].append(val)
        print(f"{q[:48]:48} | {len(rel):4} | "
              + " ".join(f"{y:.2f}" for y in v) + " | "
              + " ".join(f"{y:.2f}" for y in c))
        detail.append({"query": q, "relevant": len(rel),
                       "vanilla_top10": rv_ids[:10],
                       "compositional_top10": rc_ids[:10]})

    print("-" * 90)
    mean = lambda d: " ".join(f"{sum(d[k]) / len(d[k]):.2f}" for k in ks)
    print(f"{'MEAN':48} |     | {mean(van)} | {mean(comp)}")

    print("\n" + "=" * 90)
    print(" (B) COMPOSITIONALITY BINDING PROBE   (lower rank = better)")
    print("=" * 90)
    print(f"{'query':40} | mode | rank(correct) | rank(swapped)")
    print("-" * 90)
    probe_rows = []
    for (a, ag, b, bg) in SWAP_PAIRS:
        qtext = f"a {a} {ag} and a {b} {bg}"
        correct = [l["image_id"] for l in labels
                   if {(a, ag), (b, bg)}.issubset(
                       {(it["color"], it["garment"]) for it in l["items"]})]
        swapped = [l["image_id"] for l in labels
                   if {(b, ag), (a, bg)}.issubset(
                       {(it["color"], it["garment"]) for it in l["items"]})]
        if not correct or not swapped:
            continue
        for mode in ("vanilla", "compositional"):
            r = Retriever(store, recall_k=store.n).search(
                qtext, top_k=store.n, mode=mode)[0]
            order = [x.image_id for x in r]
            rc = _mean_rank(order, correct)
            rs = _mean_rank(order, swapped)
            print(f"{qtext[:40]:40} | {mode:13} | {rc:13.1f} | {rs:13.1f}")
            probe_rows.append({"query": qtext, "mode": mode,
                               "rank_correct": rc, "rank_swapped": rs})
        print()

    (root / "eval" / "results.json").write_text(
        json.dumps({"assessment": detail, "probe": probe_rows}, indent=2))
    print("[eval] wrote eval/results.json")


def _mean_rank(order, ids):
    ranks = [order.index(i) + 1 for i in ids if i in order]
    return sum(ranks) / len(ranks) if ranks else float("inf")


if __name__ == "__main__":
    main()
