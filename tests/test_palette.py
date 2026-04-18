from llm_relations.palette import (
    COLORS,
    POSITIONS,
    NONSENSE_WORDS,
    draw_nonsense_words,
    draw_colors,
)


def test_pools_are_nonempty_and_unique():
    assert len(COLORS) >= 5 and len(set(COLORS)) == len(COLORS)
    assert len(POSITIONS) >= 5 and len(set(POSITIONS)) == len(POSITIONS)
    assert len(NONSENSE_WORDS) >= 50 and len(set(NONSENSE_WORDS)) == len(NONSENSE_WORDS)


def test_draw_nonsense_words_is_seeded_and_unique():
    import random

    rng = random.Random(42)
    a = draw_nonsense_words(rng, n=6)
    rng2 = random.Random(42)
    b = draw_nonsense_words(rng2, n=6)
    assert a == b
    assert len(a) == 6 and len(set(a)) == 6


def test_draw_colors_returns_requested_count():
    import random

    rng = random.Random(0)
    colors = draw_colors(rng, n=3)
    assert len(colors) == 3
    assert all(c in COLORS for c in colors)
