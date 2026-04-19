import itertools

import pytest

from llm_relations.generator.cross_domain import generate_cross_domain, DOMAINS, _DOMAINS


def test_domains_list_has_five_entries():
    assert set(DOMAINS) == {"org_chart", "garden", "building", "enclosure", "vehicle_lot"}


@pytest.mark.parametrize(
    "mem_dom,perc_dom",
    [(a, b) for a, b in itertools.product(sorted(DOMAINS), repeat=2) if a != b][:6],
)
def test_cross_domain_generates_for_each_pair(mem_dom: str, perc_dom: str):
    p = generate_cross_domain(
        seed=1,
        index=0,
        memory_domain=mem_dom,
        perception_domain=perc_dom,
        correct_slot_index=1,
        feature_distractor_slot=0,
    )
    assert p.variant == "cross_domain"
    assert p.metadata["memory_domain"] == mem_dom
    assert p.metadata["perception_domain"] == perc_dom
    assert p.correct_answer["analog"] in p.prompt_text

    # The memory scenario must use memory-domain predicates; the perception
    # scenario must use perception-domain predicates.
    m_dom = _DOMAINS[mem_dom]
    p_dom = _DOMAINS[perc_dom]
    mem_section = p.prompt_text.split("Perception scenario:")[0]
    perc_section = "Perception scenario:" + p.prompt_text.split("Perception scenario:")[1]
    assert m_dom.relation_vertical in mem_section or m_dom.relation_horizontal in mem_section
    assert p_dom.relation_vertical in perc_section or p_dom.relation_horizontal in perc_section
    # Memory-domain predicates should NOT appear in perception and vice versa
    # (this is the "no string-matching predicate names" guarantee).
    assert m_dom.relation_vertical not in perc_section
    assert m_dom.relation_horizontal not in perc_section
    assert p_dom.relation_vertical not in mem_section
    assert p_dom.relation_horizontal not in mem_section


def test_cross_domain_rejects_same_domain():
    with pytest.raises(ValueError):
        generate_cross_domain(
            seed=1,
            index=0,
            memory_domain="garden",
            perception_domain="garden",
            correct_slot_index=1,
            feature_distractor_slot=0,
        )


def test_cross_domain_forces_different_slot_orders():
    # Every generated cross_domain problem should have memory's slot order
    # differ from perception's — enforced by force_different_slot_orders=True.
    for seed in [11, 22, 33, 44, 55]:
        p = generate_cross_domain(
            seed=seed,
            index=0,
            memory_domain="org_chart",
            perception_domain="garden",
            correct_slot_index=1,
            feature_distractor_slot=0,
        )
        m = tuple(p.metadata["memory_slot_order"])
        q = tuple(p.metadata["perception_slot_order"])
        assert m != q, f"seed {seed}: slot orders should differ ({m} vs {q})"
        assert p.metadata["slot_orders_aligned"] is False


def test_cross_domain_is_seed_reproducible():
    a = generate_cross_domain(
        seed=5,
        index=0,
        memory_domain="garden",
        perception_domain="building",
        correct_slot_index=2,
        feature_distractor_slot=1,
    )
    b = generate_cross_domain(
        seed=5,
        index=0,
        memory_domain="garden",
        perception_domain="building",
        correct_slot_index=2,
        feature_distractor_slot=1,
    )
    assert a == b


def test_cross_domain_uses_perception_vocab_in_question():
    """The question text must ask about the perception domain's noun, not memory's."""
    p = generate_cross_domain(
        seed=7,
        index=0,
        memory_domain="org_chart",      # feature_noun='specialty', category='employee'
        perception_domain="garden",     # feature_noun='leaf', category='plant'
        correct_slot_index=1,
        feature_distractor_slot=0,
    )
    # Find the question (last paragraph)
    question = p.prompt_text.rsplit("\n\n", 1)[-1]
    assert "plant" in question, f"question should ask about perception's category 'plant': {question}"
    assert "leaf" in question, f"question should ask about perception's feature 'leaf': {question}"
