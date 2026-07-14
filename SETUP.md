# SETUP — Glance ML Internship: Fashion & Context Retrieval

Step-by-step guide to get the repository running, from a clean clone to the
real-dataset demo and regenerating the submission PDF.

---

## 1. Prerequisites

- **Python 3.10+** (developed and tested on 3.11)
- **Git**
- ~**4 GB free disk** only if you build the real FashionPedia index
  (the synthetic demo needs almost none)
- Internet access for the first `pip install` and the optional dataset download

> The synthetic pipeline runs fully **CPU-only** and needs **no download**.
> The real pipeline uses `torch` + `open_clip_torch` (CPU build is fine).

---

## 2. Clone & install

```bash
git clone https://github.com/SatyamSingh8306/glance-fashion-retrieval
cd glance-fashion-retrieval

# virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` already includes `torch` and `open_clip_torch`. If you want a
smaller **CPU-only** torch install instead, replace the torch line with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install open_clip_torch
```

---

## 3. Quick start — synthetic demo (no downloads)

This builds a 648-image synthetic dataset (auto-generated), indexes it, runs the
evaluation, and does a sample search. Nothing external is required.

```bash
python scripts/build_index.py --data data --index index --n 600
python eval/run_evaluation.py
python scripts/search.py "A red tie and a white shirt in a formal setting" --k 5
```

Expected: `run_evaluation.py` prints mean **R@10 vanilla 0.34 → compositional
0.55** and a compositionality binding probe showing vanilla cannot separate
correct vs swapped outfits while the compositional retriever can.

---

## 4. Real dataset — FashionPedia `test` split (3,200 images)

### 4.1 Download

**Dataset URL:** https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip

```bash
# from the project root
curl -L -o val_test2020.zip https://s3.amazonaws.com/ifashionist-dataset/images/val_test2020.zip
# (or paste the URL above into a browser and download manually)
```

### 4.2 Unzip into the expected location

The indexer walks the image folder recursively and uses each file's name
(without extension) as the image id, so a **flat** folder of images is expected:

```bash
mkdir -p data/fashion2020
unzip val_test2020.zip -d data/fashion2020
# result: data/fashion2020/test/*.jpg   (3,200 images)
```

> If your unzip puts the images one level deeper, just point `--images-dir` at
> the folder that actually contains the `.jpg` files.

### 4.3 Build the CLIP index

```bash
python scripts/build_index.py --images-dir data/fashion2020/test --index index_real --backend clip
```

This writes `index_real/` (FAISS index + attribute metadata). It is git-ignored.

### 4.4 Run the real evaluation & examples

```bash
python eval/run_real.py
```

Writes `eval/real_results.json` and copies the top-3 retrieved thumbnails per
query into `docs/examples/` for the report.

### 4.5 Search the real index

```bash
python scripts/search.py "A person in a bright yellow raincoat" --index index_real --clip --k 5
python scripts/search.py "A red tie and a white shirt in a formal setting" --index index_real --clip --mode vanilla
```

On real photos the compositional re-rank moves little versus vanilla CLIP — this
is expected and documented in the report (§3.3): zero-shot CLIP is weak on
fine-grained colour. The architecture fix (fine-tuned detector / BLIP captions)
is in Future Work.

---

## 5. Data formats the code expects

The indexer (`src/indexer.py`) is backend-agnostic:

| Mode | What to provide | Command |
|------|-----------------|---------|
| **Your own photos** | Any folder of `.png/.jpg/.jpeg/.webp` (flat or nested). **No labels needed** (CLIP is zero-shot). | `python scripts/build_index.py --images-dir <FOLDER> --index <IDX> --backend clip` |
| **Synthetic eval** | Nothing — auto-generated into `data/` if `data/labels.json` is missing. | `python scripts/build_index.py --data data --index index --n 600` |
| **FashionPedia** | Images in `data/fashion2020/test` (see §4). | `python scripts/build_index.py --images-dir data/fashion2020/test --index index_real --backend clip` |

Optional custom synthetic labels: a JSON **list** of objects, e.g.

```json
[
  {"image_id": "img_00000", "scene": "park", "style": "casual",
   "items": [{"color": "blue", "garment": "shirt"},
             {"color": "black", "garment": "pants"}]}
]
```

`color` / `garment` values must come from `src/taxonomy.py`. `scene` and `style`
are optional.

---

## 6. Reproduce the submission PDF (`docs/report.pdf`)

`docs/build_report.py` reads the live eval outputs, so run both pipelines first:

```bash
python eval/run_evaluation.py      # -> eval/results.json
python eval/run_real.py            # -> eval/real_results.json + docs/examples/
python docs/build_report.py         # -> docs/report.pdf
```

The PDF contains: approaches + trade-offs, the chosen architecture, the codebase
link, and future work (locations/weather + precision).

---

## 7. Project layout

```
src/        taxonomy, query_parser, feature_extractor, store, indexer,
            retriever, data_generator
scripts/    build_index.py, search.py
eval/       run_evaluation.py, run_real.py, results.json, real_results.json
docs/       build_report.py, report.pdf, examples/
data/       synthetic images + labels  (generated, git-ignored)
index/      synthetic FAISS index       (generated, git-ignored)
index_real/ real FAISS index            (generated, git-ignored)
```

Large artifacts (`data/`, `index/`, `index_real/`, `.venv/`) are git-ignored and
regenerated from the commands above — the repo stays small and self-contained.
