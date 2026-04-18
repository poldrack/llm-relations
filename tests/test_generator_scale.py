import pytest

from llm_relations.generator.scale import generate_scale


@pytest.mark.parametrize("n", [4, 5, 6, 7, 8])
def test_scale_generates_n_objects(n: int):
    p = generate_scale(seed=1, index=0, n_objects=n)
    assert p.variant == "scale"
    assert p.metadata["n_objects"] == n
    # Prompt should reference all n memory + n perception objects
    # (we verify by checking the memory scenario mentions n distinct nonsense words)
    memory_section = p.prompt_text.split("Perception scenario")[0]
    # Count nonsense-word mentions by checking each metadata word
    memory_words = p.metadata["memory_words"]
    assert len(memory_words) == n
    for w in memory_words:
        assert w in memory_section


def test_scale_correct_analog_in_prompt():
    p = generate_scale(seed=2, index=0, n_objects=5)
    assert p.correct_answer["analog"] in p.prompt_text


def test_scale_is_seed_reproducible():
    a = generate_scale(seed=3, index=0, n_objects=6)
    b = generate_scale(seed=3, index=0, n_objects=6)
    assert a == b
