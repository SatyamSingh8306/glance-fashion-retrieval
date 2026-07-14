"""Synthetic dataset generator for the runnable demo (labels stand in for detector output)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from taxonomy import COLORS, GARMENT_SLOTS

COLOR_RGB = {
    "red": (200, 30, 30), "blue": (30, 80, 200), "yellow": (240, 210, 40),
    "green": (30, 160, 60), "orange": (240, 130, 30), "pink": (230, 120, 170),
    "purple": (140, 60, 200), "white": (235, 235, 235), "black": (30, 30, 30),
    "gray": (130, 130, 130), "grey": (130, 130, 130), "brown": (130, 80, 40),
    "navy": (20, 30, 90), "beige": (210, 190, 150), "cyan": (40, 200, 220),
    "maroon": (120, 20, 40),
}

SCENE_BG = {
    "office": (205, 205, 210), "street": (150, 150, 155),
    "park": (120, 190, 120), "home": (220, 200, 170),
}

SLOT_CHOICES = {
    "top": ["shirt", "tshirt", "hoodie", "blazer", "sweater"],
    "bottom": ["pants", "jeans", "skirt"],
    "outerwear": ["raincoat", "coat", "jacket"],
    "accessory": ["tie", "scarf"],
}


def _sample_outfit(rng, force_bindings=None):
    items = []
    items.append({"slot": "top", "garment": rng.choice(SLOT_CHOICES["top"]),
                  "color": rng.choice(COLORS)})
    if rng.random() < 0.9:
        items.append({"slot": "bottom", "garment": rng.choice(SLOT_CHOICES["bottom"]),
                      "color": rng.choice(COLORS)})
    if rng.random() < 0.35:
        items.append({"slot": "outerwear", "garment": rng.choice(SLOT_CHOICES["outerwear"]),
                      "color": rng.choice(COLORS)})
    if rng.random() < 0.3:
        items.append({"slot": "accessory", "garment": rng.choice(SLOT_CHOICES["accessory"]),
                      "color": rng.choice(COLORS)})
    if force_bindings:
        for fb in force_bindings:
            replaced = False
            for it in items:
                if it["garment"] == fb["garment"]:
                    it["color"] = fb["color"]
                    replaced = True
            if not replaced:
                slot = next(s for s, gs in GARMENT_SLOTS.items() if fb["garment"] in gs)
                items.append({"slot": slot, "garment": fb["garment"], "color": fb["color"]})
    return items


def _render(items, scene, size=(224, 224)):
    from taxonomy import GARMENT_TO_SLOT
    img = Image.new("RGB", size, SCENE_BG.get(scene, (180, 180, 180)))
    d = ImageDraw.Draw(img)
    rng = random.Random(hash((scene, tuple(sorted((i["garment"], i["color"])
                                             for i in items)))) % (2**31))
    for _ in range(6):
        x0, y0 = rng.randint(0, size[0]), rng.randint(0, size[1])
        w, h = rng.randint(20, 60), rng.randint(20, 60)
        shade = tuple(max(0, min(255, int(c + rng.randint(-25, 25))))
                      for c in SCENE_BG.get(scene, (180, 180, 180)))
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=shade)
    by = int(size[1] * 0.30)
    slot_of = lambda g: GARMENT_TO_SLOT.get(g)
    outer = next((i for i in items if slot_of(i["garment"]) == "outerwear"), None)
    if outer:
        d.rectangle([int(size[0]*0.30), by, int(size[0]*0.70), int(size[1]*0.92)],
                    fill=COLOR_RGB.get(outer["color"], (120, 120, 120)))
    top = next((i for i in items if slot_of(i["garment"]) == "top"), None)
    if top:
        d.rectangle([int(size[0]*0.35), by, int(size[0]*0.65), int(size[1]*0.62)],
                    fill=COLOR_RGB.get(top["color"], (120, 120, 120)))
    bot = next((i for i in items if slot_of(i["garment"]) == "bottom"), None)
    if bot:
        d.rectangle([int(size[0]*0.37), int(size[1]*0.62), int(size[0]*0.63),
                     int(size[1]*0.92)], fill=COLOR_RGB.get(bot["color"], (120, 120, 120)))
    acc = next((i for i in items if slot_of(i["garment"]) == "accessory"), None)
    if acc:
        d.rectangle([int(size[0]*0.47), by + 6, int(size[0]*0.53), int(size[1]*0.55)],
                    fill=COLOR_RGB.get(acc["color"], (120, 120, 120)))
    return img


def _make_label(image_id, scene, style, items):
    return {"image_id": image_id, "scene": scene, "style": style,
            "items": [{"garment": it["garment"], "color": it["color"]} for it in items]}


EVAL_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]

SWAP_PAIRS = [
    ("red", "shirt", "blue", "pants"),
    ("blue", "shirt", "red", "pants"),
    ("white", "shirt", "black", "pants"),
    ("black", "shirt", "white", "pants"),
    ("red", "tie", "white", "shirt"),
    ("white", "tie", "red", "shirt"),
]


def generate(n: int = 600, out_dir="data", seed: int = 42):
    out_dir = Path(out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    labels = []
    scene_choices = list(SCENE_BG.keys())
    style_choices = ["formal", "business", "casual", "weekend"]

    n_random = int(n * 0.8)
    for i in range(n_random):
        labels.append(_make_label(f"img_{i:05d}", rng.choice(scene_choices),
                                  rng.choice(style_choices), _sample_outfit(rng)))

    n_swap = n - n_random
    for j in range(n_swap):
        a, ag, b, bg = rng.choice(SWAP_PAIRS)
        labels.append(_make_label(
            f"img_{n_random + j:05d}", rng.choice(scene_choices),
            rng.choice(style_choices),
            _sample_outfit(rng, force_bindings=[
                {"color": a, "garment": ag}, {"color": b, "garment": bg}])))

    seed_specs = [
        {"n": 8, "scene": "street", "style": "weather",
         "bind": [("yellow", "raincoat"), ("blue", "pants")]},
        {"n": 8, "scene": "office", "style": "business",
         "bind": [("navy", "blazer"), ("gray", "pants"), ("red", "tie")]},
        {"n": 8, "scene": "park", "style": "casual",
         "bind": [("blue", "shirt"), ("green", "pants")]},
        {"n": 8, "scene": "street", "style": "weekend",
         "bind": [("gray", "hoodie"), ("blue", "jeans")]},
        {"n": 8, "scene": "office", "style": "formal",
         "bind": [("white", "shirt"), ("red", "tie"), ("black", "pants")]},
    ]
    seed_idx = n_random + n_swap
    for spec in seed_specs:
        for _ in range(spec["n"]):
            labels.append(_make_label(
                f"img_{seed_idx:05d}", spec["scene"], spec["style"],
                _sample_outfit(rng, force_bindings=[
                    {"color": c, "garment": g} for c, g in spec["bind"]])))
            seed_idx += 1
    for _ in range(8):
        labels.append(_make_label(
            f"img_{seed_idx:05d}", "office", "formal",
            _sample_outfit(rng, force_bindings=[
                {"color": "red", "garment": "shirt"},
                {"color": "white", "garment": "tie"},
                {"color": "black", "garment": "pants"}])))
        seed_idx += 1

    for lbl in labels:
        _render(lbl["items"], lbl["scene"]).save(img_dir / f"{lbl['image_id']}.png")
    with open(out_dir / "labels.json", "w") as f:
        json.dump(labels, f)
    print(f"[generator] wrote {len(labels)} images + labels -> {out_dir}")
    return labels


def build_eval_ground_truth(labels, queries=EVAL_QUERIES):
    from query_parser import RuleBasedParser
    parser = RuleBasedParser()
    gt = {}
    for q in queries:
        pq = parser.parse(q)
        gt[q] = [lbl["image_id"] for lbl in labels if _satisfies(pq, lbl)]
    return gt


def _satisfies(pq, lbl):
    bound = {(it["color"], it["garment"]) for it in lbl["items"]}
    for b in pq.bindings:
        if (b["color"], b["garment"]) not in bound:
            return False
    if pq.scene and pq.scene != lbl["scene"]:
        return False
    if pq.style and pq.style != lbl["style"]:
        return False
    return True


if __name__ == "__main__":
    labels = generate(600)
    gt = build_eval_ground_truth(labels)
    (Path("data") / "eval_ground_truth.json").write_text(json.dumps(gt, indent=2))
    for q, m in gt.items():
        print(f"[{len(m):3d}] {q}")
