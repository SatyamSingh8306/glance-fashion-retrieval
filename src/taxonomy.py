"""Fashion / context taxonomy shared by the indexer and retriever."""

GARMENT_SLOTS = {
    "top": ["shirt", "tshirt", "t-shirt", "blouse", "sweater", "hoodie", "tank_top", "polo"],
    "bottom": ["pants", "trousers", "jeans", "shorts", "skirt", "leggings"],
    "outerwear": ["raincoat", "coat", "jacket", "blazer", "windbreaker", "parka"],
    "dress": ["dress", "gown", "jumpsuit"],
    "accessory": ["tie", "scarf", "hat", "belt", "sunglasses", "gloves"],
}

ALL_GARMENTS = [g for slot in GARMENT_SLOTS.values() for g in slot]
GARMENT_TO_SLOT = {g: slot for slot, gs in GARMENT_SLOTS.items() for g in gs}

GARMENT_ALIASES = {
    "t-shirt": "tshirt", "t shirt": "tshirt", "tee": "tshirt", "top": "shirt",
    "jumper": "sweater", "suit": "blazer", "trousers": "pants",
}

COLORS = [
    "red", "blue", "yellow", "green", "orange", "pink", "purple", "white",
    "black", "gray", "grey", "brown", "navy", "beige", "cyan", "maroon",
]
COLOR_ALIASES = {
    "grey": "gray", "turquoise": "cyan", "crimson": "red", "scarlet": "red",
    "azure": "blue", "olive": "green",
}

SCENES = ["office", "street", "park", "home", "outdoor", "indoor"]
SCENE_ALIASES = {
    "modern office": "office", "office interior": "office", "workplace": "office",
    "city": "street", "urban": "street", "sidewalk": "street", "bench": "park",
    "garden": "park", "living room": "home", "kitchen": "home",
    "indoors": "indoor", "outdoors": "outdoor",
}

STYLES = ["formal", "business", "casual", "sporty", "weekend", "elegant", "weather"]
STYLE_ALIASES = {
    "professional": "business", "business attire": "business",
    "business casual": "business", "smart": "formal", "dressed up": "formal",
    "relaxed": "casual", "everyday": "casual", "rainy": "weather", "cold": "weather",
}

BRIGHT_MODIFIERS = {"bright", "vibrant", "neon", "light", "pale"}
DARK_MODIFIERS = {"dark", "deep", "navy", "black"}

GLOBAL_VOCAB = COLORS + ALL_GARMENTS + SCENES + STYLES
VOCAB_INDEX = {tok: i for i, tok in enumerate(GLOBAL_VOCAB)}


def canonical_color(c: str) -> str:
    return COLOR_ALIASES.get(c.lower().strip(), c.lower().strip())


def canonical_garment(g: str) -> str:
    g = g.lower().strip()
    g = GARMENT_ALIASES.get(g, g)
    if g.endswith("s") and g[:-1] in ALL_GARMENTS:
        g = g[:-1]
    if g in ALL_GARMENTS:
        return g
    for cand in ALL_GARMENTS:
        if cand in g or g in cand:
            return cand
    return g


def canonical_scene(s: str) -> str:
    return SCENE_ALIASES.get(s.lower().strip(), s.lower().strip())


def canonical_style(s: str) -> str:
    return STYLE_ALIASES.get(s.lower().strip(), s.lower().strip())
