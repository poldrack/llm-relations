from llm_relations.generator.feature_misleading import generate_feature_misleading


def test_feature_misleading_returns_well_formed_problem():
    p = generate_feature_misleading(seed=2, index=0, correct_slot_index=1, feature_distractor_slot=0)
    assert p.variant == "feature_misleading"
    assert p.problem_id == "feature_misleading_00"
    assert p.metadata["n_objects"] == 3


def test_feature_misleading_adds_salience_cue_to_feature_twin():
    """The feature-twin should be given an extra salience cue in its description."""
    p = generate_feature_misleading(seed=9, index=0, correct_slot_index=2, feature_distractor_slot=0)
    twin = p.metadata["feature_match_answer"]["analog"]
    assert f"The {twin}'s buttons are arranged in the same layout as a typical object." in p.prompt_text


def test_feature_misleading_is_seed_reproducible():
    a = generate_feature_misleading(seed=1, index=0, correct_slot_index=1, feature_distractor_slot=2)
    b = generate_feature_misleading(seed=1, index=0, correct_slot_index=1, feature_distractor_slot=2)
    assert a == b
