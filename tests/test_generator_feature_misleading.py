from llm_relations.generator.feature_misleading import generate_feature_misleading


def test_feature_misleading_returns_well_formed_problem():
    p = generate_feature_misleading(seed=2, index=0, correct_slot_index=1, feature_distractor_slot=0)
    assert p.variant == "feature_misleading"
    assert p.problem_id == "feature_misleading_00"
    assert p.metadata["n_objects"] == 3


def test_feature_misleading_distractor_has_two_target_color_buttons():
    p = generate_feature_misleading(seed=9, index=0, correct_slot_index=2, feature_distractor_slot=0)
    color = p.correct_answer["button_color"]
    distractor = p.metadata["feature_match_answer"]["analog"]
    # Find the distractor's button-description sentence; target color should appear >=2 times.
    marker = f"The {distractor} has"
    line_start = p.prompt_text.index(marker)
    line_end = p.prompt_text.index(".", line_start)
    segment = p.prompt_text[line_start:line_end]
    assert segment.count(color) >= 2


def test_feature_misleading_is_seed_reproducible():
    a = generate_feature_misleading(seed=1, index=0, correct_slot_index=0, feature_distractor_slot=1)
    b = generate_feature_misleading(seed=1, index=0, correct_slot_index=0, feature_distractor_slot=1)
    assert a == b
