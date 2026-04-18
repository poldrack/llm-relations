from __future__ import annotations

import random

COLORS: list[str] = ["blue", "red", "green", "yellow", "purple"]

POSITIONS: list[str] = ["top", "bottom", "side", "front", "back"]

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
