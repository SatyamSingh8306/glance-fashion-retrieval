# Glance ML Internship — Multimodal Fashion & Context Retrieval

An attribute-aware, **compositional** image retrieval engine that beats vanilla
CLIP on fashion queries by explicitly modelling `(colour → garment)` bindings,
scene and style — instead of collapsing a query into a bag of words.

## Architecture at a glance
```
natural-language query
        │  query_parser.py  (LLM or rule-based)
        ▼
 ParsedQuery{ scene, style, bindings[(color,garment)...], free_text }
        │
   ┌────┴─────────────────────────────────────────────┐
   │  Part A: Indexer (indexer.py)                     │
   │   images ──feature_extractor.py──▶ FAISS (global) │
   │                                  └▶ attribute tables│
   └────┬─────────────────────────────────────────────┘
        │  Part B: Retriever (retriever.py)
        ▼
  FAISS recall (top-N)  ──▶  compositional re-rank
        (global sim)    (binding + scene + style scores)
        │
        ▼
   top-k images
```

## Run the demo end-to-end (no GPU / torch needed)
```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

python scripts/build_index.py --data data --index index --n 600
python eval/run_evaluation.py          # vanilla vs compositional + binding probe
python scripts/search.py "A red tie and a white shirt in a formal setting" --k 5
```

## Use real data (FashionPedia / your own photos)
The repo includes a real run on the FashionPedia `test` split (val_test2020.zip,
3,200 images). Unzip it under `data/fashion2020/test` and:
```bash
pip install torch open_clip_torch        # CPU torch: pip install torch --index-url https://download.pytorch.org/whl/cpu
python scripts/build_index.py --images-dir data/fashion2020/test --index index_real --backend clip
python eval/run_real.py
python scripts/search.py "A red tie and a white shirt in a formal setting" --index index_real --clip
```
The same indexer/retriever runs unchanged; only the extractor backend differs
(`CLIPExtractor` instead of `SyntheticExtractor`). On real photos the
compositional re-rank moves less than on the labelled synthetic set because the
attribute signal comes from zero-shot CLIP (weak on fine-grained colour); swap
in a fine-tuned fashion detector or BLIP captions to recover the gain.

## Why this beats vanilla CLIP
Vanilla CLIP encodes a query like *"red shirt + blue pants"* and an image into
single global vectors; similarity is largely a **bag-of-words** match, so the
*red shirt / blue pants* image and the *blue shirt / red pants* image score
almost identically (our probe confirms vanilla ranks them ~tied).  We instead
keep a **disentangled attribute table** scoring each `(colour, garment)` pair
independently, then re-rank the FAISS candidates by a compositional score that
enforces the correct colour→garment binding.  See `docs/report.pdf`.

## Repo layout
```
src/        taxonomy, query_parser, feature_extractor, store, indexer, retriever,
            data_generator
scripts/    build_index.py, search.py
eval/       run_evaluation.py, results.json
data/       synthetic images + labels (generated)
index/      FAISS index + metadata (generated)
```
