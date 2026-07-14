"""Query parsing into structured slots; the key output is paired (colour, garment) bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from taxonomy import (ALL_GARMENTS, BRIGHT_MODIFIERS, COLORS, GARMENT_TO_SLOT,
                      SCENES, STYLES, SCENE_ALIASES, STYLE_ALIASES,
                      canonical_color, canonical_garment,
                      canonical_scene, canonical_style)


@dataclass
class ParsedQuery:
    raw: str
    scene: str | None = None
    style: str | None = None
    bindings: list = field(default_factory=list)
    free_text: str = ""


class RuleBasedParser:
    _STOP = {"a", "an", "the", "in", "on", "with", "and", "of", "inside",
             "person", "people", "someone", "wearing", "wear", "sitting",
             "standing", "for", "to", "at", "by", "is", "are", "who", "that",
             "walk", "walking", "outfit", "attire", "setting", "place",
             "city", "modern", "bench", "weekend"}

    def parse(self, text: str) -> ParsedQuery:
        raw = text
        t = re.sub(r"[^a-z0-9\s]", " ", text.lower())
        tokens = t.split()
        q = ParsedQuery(raw=raw)

        for alias, canon in sorted(SCENE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if alias in t:
                q.scene = canon
                break
        if q.scene is None:
            for s in SCENES:
                if re.search(rf"\b{s}\b", t):
                    q.scene = s
                    break
        for alias, canon in sorted(STYLE_ALIASES.items(), key=lambda kv: -len(kv[0])):
            if alias in t:
                q.style = canon
                break
        if q.style is None:
            for s in STYLES:
                if re.search(rf"\b{s}\b", t):
                    q.style = s
                    break

        used = set()
        for i, tok in enumerate(tokens):
            modifier = None
            if tok in BRIGHT_MODIFIERS and i + 1 < len(tokens) and tokens[i + 1] in COLORS:
                modifier = tok
                color = canonical_color(tokens[i + 1])
                start = i + 1
            elif tok in COLORS:
                color = canonical_color(tok)
                start = i
            else:
                continue
            garment = None
            for j in range(start + 1, min(start + 3, len(tokens))):
                g = _match_garment(tokens[j])
                if g:
                    garment = g
                    used.update({i, j})
                    if modifier:
                        used.add(i + 1)
                    break
            if garment:
                q.bindings.append({"color": color, "garment": garment, "modifier": modifier})
                used.add(start)

        q.free_text = " ".join(tok for i, tok in enumerate(tokens)
                               if i not in used and tok not in self._STOP)
        return q


def _match_garment(tok: str):
    tok = tok.rstrip("s")
    if tok in ALL_GARMENTS:
        return tok
    for g in ALL_GARMENTS:
        if g in tok or tok in g:
            return g
    return None


def get_parser():
    return RuleBasedParser()
