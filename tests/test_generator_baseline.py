from llm_relations.generator.baseline import generate_baseline


def test_baseline_returns_well_formed_problem():
    p = generate_baseline(seed=1, index=0, correct_slot_index=2, feature_distractor_slot=0)
    assert p.variant == "baseline"
    assert p.problem_id == "baseline_00"
    assert p.metadata["n_objects"] == 3
    assert p.correct_answer["analog"] and p.correct_answer["button_color"]
    assert p.metadata["feature_match_answer"]["analog"] != p.correct_answer["analog"]
    assert p.metadata["positional_match_answer"]["analog"] != p.correct_answer["analog"]


def test_baseline_is_seed_reproducible():
    a = generate_baseline(seed=7, index=0, correct_slot_index=1, feature_distractor_slot=2)
    b = generate_baseline(seed=7, index=0, correct_slot_index=1, feature_distractor_slot=2)
    assert a == b


def test_baseline_correct_analog_appears_in_prompt_with_target_button():
    p = generate_baseline(seed=3, index=0, correct_slot_index=0, feature_distractor_slot=1)
    analog = p.correct_answer["analog"]
    color = p.correct_answer["button_color"]
    assert analog in p.prompt_text
    # The correct analog should have the target color on top
    assert f"The {analog} has a {color} button on top" in p.prompt_text


def test_baseline_feature_distractor_has_target_color_button():
    p = generate_baseline(seed=5, index=0, correct_slot_index=2, feature_distractor_slot=0)
    distractor = p.metadata["feature_match_answer"]["analog"]
    color = p.correct_answer["button_color"]
    # Find the distractor's button-description sentence and check the target color is in it.
    marker = f"The {distractor} has"
    line_start = p.prompt_text.index(marker)
    line_end = p.prompt_text.index(".", line_start)
    assert color in p.prompt_text[line_start:line_end]
