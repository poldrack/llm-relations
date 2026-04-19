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


def test_baseline_activation_position_is_in_prompt():
    """Activation is now by position, not color."""
    p = generate_baseline(seed=3, index=0, correct_slot_index=1, feature_distractor_slot=2)
    pos = p.metadata["activation_position"]
    target = p.metadata["memory_target_name"]
    assert f"Pressing the button on the {pos} of the {target}" in p.prompt_text


def test_baseline_correct_analog_has_answer_color_at_activation_position():
    """The correct answer's color should appear at the activation position on the correct analog."""
    p = generate_baseline(seed=3, index=0, correct_slot_index=1, feature_distractor_slot=2)
    analog = p.correct_answer["analog"]
    color = p.correct_answer["button_color"]
    pos = p.metadata["activation_position"]
    assert f"The {analog} has" in p.prompt_text
    # Somewhere in the analog's description, the answer color should be at the activation position.
    marker = f"The {analog} has"
    start = p.prompt_text.index(marker)
    end = p.prompt_text.index(".", start)
    segment = p.prompt_text[start:end]
    assert f"{color} button on {pos}" in segment


def test_baseline_feature_twin_is_exact_copy_of_memory_target():
    """The feature-twin distractor should have EXACTLY the same (color, position) buttons as m_target."""
    p = generate_baseline(seed=5, index=0, correct_slot_index=2, feature_distractor_slot=0)
    twin = p.metadata["feature_match_answer"]["analog"]
    target = p.metadata["memory_target_name"]

    def _extract_buttons(name: str) -> str:
        marker = f"The {name} has"
        start = p.prompt_text.index(marker)
        end = p.prompt_text.index(".", start)
        return p.prompt_text[start:end]

    twin_desc = _extract_buttons(twin)
    target_desc = _extract_buttons(target)
    # Strip the leading "The X has" portion and compare the button-list tails.
    twin_tail = twin_desc.split("has", 1)[1].strip()
    target_tail = target_desc.split("has", 1)[1].strip()
    assert twin_tail == target_tail, (
        f"feature-twin should exactly copy m_target's buttons\n"
        f"  twin:   {twin_tail}\n"
        f"  target: {target_tail}"
    )


def test_baseline_correct_analog_colors_disjoint_from_target():
    """The correct analog's colors should be disjoint from m_target's, so that
    feature-matching on m_target's colors gives the wrong object."""
    p = generate_baseline(seed=11, index=0, correct_slot_index=1, feature_distractor_slot=2)
    analog = p.correct_answer["analog"]
    target = p.metadata["memory_target_name"]

    def _colors(name: str) -> set[str]:
        marker = f"The {name} has"
        start = p.prompt_text.index(marker)
        end = p.prompt_text.index(".", start)
        segment = p.prompt_text[start:end]
        # Parse "a X button on Y" chunks.
        import re
        return set(re.findall(r"a (\w+) button on \w+", segment))

    target_colors = _colors(target)
    analog_colors = _colors(analog)
    assert target_colors.isdisjoint(analog_colors), (
        f"correct analog colors {analog_colors} overlap with target colors {target_colors}"
    )
