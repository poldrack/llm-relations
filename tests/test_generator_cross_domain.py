import pytest

from llm_relations.generator.cross_domain import generate_cross_domain, DOMAINS


def test_domains_list_has_five_entries():
    assert set(DOMAINS) == {"org_chart", "garden", "building", "enclosure", "vehicle_lot"}


@pytest.mark.parametrize("domain", sorted(["org_chart", "garden", "building", "enclosure", "vehicle_lot"]))
def test_cross_domain_generates_for_each_domain(domain: str):
    p = generate_cross_domain(seed=1, index=0, domain=domain, correct_slot_index=1, feature_distractor_slot=0)
    assert p.variant == "cross_domain"
    assert p.metadata["domain"] == domain
    assert p.correct_answer["analog"] in p.prompt_text


def test_cross_domain_is_seed_reproducible():
    a = generate_cross_domain(seed=5, index=0, domain="garden", correct_slot_index=2, feature_distractor_slot=1)
    b = generate_cross_domain(seed=5, index=0, domain="garden", correct_slot_index=2, feature_distractor_slot=1)
    assert a == b
