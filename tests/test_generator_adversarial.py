from llm_relations.generator.adversarial import generate_adversarial, LINGUISTIC_DECOYS


def test_adversarial_has_linguistic_decoy_sentence():
    p = generate_adversarial(seed=1, index=0, correct_slot_index=1, feature_distractor_slot=0, decoy_index=0)
    distractor = p.metadata["feature_match_answer"]["analog"]
    # The decoy uses the feature-twin's own activation-position button color.
    twin_color = p.metadata["feature_match_answer"]["button_color"]
    assert any(
        template.format(name=distractor, color=twin_color) in p.prompt_text
        for template in LINGUISTIC_DECOYS
    )


def test_adversarial_correct_answer_is_structure_match_not_distractor():
    p = generate_adversarial(seed=2, index=0, correct_slot_index=2, feature_distractor_slot=0, decoy_index=1)
    assert p.correct_answer["analog"] != p.metadata["feature_match_answer"]["analog"]


def test_adversarial_is_seed_reproducible():
    a = generate_adversarial(seed=9, index=0, correct_slot_index=1, feature_distractor_slot=2, decoy_index=3)
    b = generate_adversarial(seed=9, index=0, correct_slot_index=1, feature_distractor_slot=2, decoy_index=3)
    assert a == b
