from __future__ import annotations

import random

# Expanded palette: each problem samples two disjoint color-triples
# (one for the memory target, one for the correct perception analog),
# so we need at least 6 colors available. We use 10 common, easily
# distinguishable color names.
COLORS: list[str] = [
    "blue", "red", "green", "yellow", "purple",
    "orange", "pink", "brown", "black", "cyan",
]

POSITIONS: list[str] = ["top", "bottom", "side", "front", "back"]

# The "button-slot" positions used for every 3-button object in the
# redesigned generators. Kept as a canonical ordering so that every
# object has a button at each of these slots — equalizing button counts
# across perception objects (prevents count-based shortcuts).
BUTTON_SLOTS: list[str] = ["top", "side", "bottom"]

NONSENSE_WORDS: list[str] = [
    "florp", "greeble", "wix", "zop", "quib", "mek", "tarn", "blix", "drog", "pell",
    "krink", "vorp", "snig", "thorp", "glim", "rask", "plon", "twix", "vask", "grup",
    "sploof", "zink", "brall", "muk", "flen", "jorp", "skiv", "trob", "yup", "nop",
    "wren", "clop", "fip", "hask", "lurn", "moob", "nurp", "oxim", "prit", "quan",
    "rast", "scob", "tunk", "ulmp", "vro", "woz", "xav", "yib", "zant", "arn",
    "bep", "corp", "dask", "emp", "fro", "gox", "hint", "ikk", "jax", "klon",
]


def draw_nonsense_words(rng: random.Random, n: int) -> list[str]:
    return rng.sample(NONSENSE_WORDS, n)


def draw_colors(rng: random.Random, n: int) -> list[str]:
    return rng.sample(COLORS, n)


def draw_disjoint_color_triples(rng: random.Random, k: int) -> list[list[str]]:
    """Sample k disjoint 3-color lists from the palette.

    Requires k * 3 <= len(COLORS). With 10 colors we can get up to 3 triples.
    """
    needed = k * 3
    assert needed <= len(COLORS), f"need {needed} disjoint colors, have {len(COLORS)}"
    pool = rng.sample(COLORS, needed)
    return [pool[i * 3:(i + 1) * 3] for i in range(k)]
