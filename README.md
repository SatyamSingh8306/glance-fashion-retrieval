# Glance ML Internship — Multimodal Fashion & Context Retrieval

An attribute-aware, **compositional** image retrieval engine that beats vanilla
CLIP on fashion queries by explicitly modelling `(colour → garment)` bindings,
scene and style — instead of collapsing a query into a bag of words.

> **Setup / install / dataset download:** see [`SETUP.md`](SETUP.md).

## Architecture at a glance
```
natural-language query
        │  query_parser.py  (rule-based)
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

## Data formats (what the code expects)

The indexer (`src/indexer.py`) walks the given image folder **recursively** and
accepts `.png / .jpg / .jpeg / .webp` (case-insensitive). The image id is the
file name **without extension** (`p.stem`). No specific sub-folder layout is
required — flat or nested both work.

**A. Your own photos (recommended, zero config).**
Just drop images in any folder; labels are NOT needed (CLIP is zero-shot):
```bash
python scripts/build_index.py --images-dir path/to/your/images --index my_index --backend clip
python scripts/search.py "a person in a blue jacket on a street" --index my_index --clip
```

**B. Synthetic evaluation dataset (auto-generated).**
Running the demo without `data/labels.json` creates it for you:
```bash
python scripts/build_index.py --data data --index index --n 600
# -> writes data/images/*.png + data/labels.json (648 images)
```
If you want to supply your own synthetic labels, the expected JSON is a list of
objects (one per image), e.g.:
```json
[
  {"image_id": "img_00000", "scene": "park", "style": "casual",
   "items": [{"color": "blue", "garment": "shirt"},
             {"color": "black", "garment": "pants"}]},
  ...
]
```
`scene` and `style` are optional; `items` is a list of `{color, garment}` pairs
(colour/garment must come from `src/taxonomy.py`).

**C. FashionPedia reproduction.**
Download `val_test2020.zip` from the FashionPedia source and unzip it so the
images sit in `data/fashion2020/test` (a flat folder of images), then:
```bash
python scripts/build_index.py --images-dir data/fashion2020/test --index index_real --backend clip
```

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
