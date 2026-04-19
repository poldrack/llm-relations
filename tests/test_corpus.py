from pathlib import Path

import pytest

from llm_relations.problem import load_problem

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"

EXPECTED_VARIANTS = {
    "baseline",
    "feature_misleading",
    "scale",
    "cross_domain",
    "adversarial",
    "control",
}


def _all_problems():
    files = sorted(PROBLEMS_DIR.glob("*.json"))
    return [load_problem(f) for f in files]


def test_corpus_has_expected_problem_count():
    if not PROBLEMS_DIR.exists() or not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen — run scripts/freeze_corpus.py")
    # 5 instances per variant * 6 variants = 30
    assert len(list(PROBLEMS_DIR.glob("*.json"))) == 5 * len(EXPECTED_VARIANTS)


def test_corpus_covers_all_variants():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    problems = _all_problems()
    variants = {p.variant for p in problems}
    assert variants == EXPECTED_VARIANTS
    for variant in variants:
        assert sum(1 for p in problems if p.variant == variant) == 5


def test_each_problem_has_internally_consistent_ground_truth():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    for p in _all_problems():
        if p.variant == "control":
            # Control has no structurally-determined answer; sentinel is "AMBIGUOUS".
            assert p.correct_answer["analog"] == "AMBIGUOUS"
            # But the STRUCTURAL answer (what would be correct if relations were present)
            # should still be a real name in the prompt.
            structural = p.metadata["structural_correct_analog"]
            assert structural in p.prompt_text
        else:
            assert p.correct_answer["analog"] in p.prompt_text, f"{p.problem_id}: analog not in prompt"

        # Distractors must differ from the correct answer in non-control variants.
        fma = p.metadata["feature_match_answer"]["analog"]
        pma = p.metadata["positional_match_answer"]["analog"]
        if p.variant != "control":
            assert fma != p.correct_answer["analog"], f"{p.problem_id}: feature_match == correct"
            assert pma != p.correct_answer["analog"], f"{p.problem_id}: positional_match == correct"


def test_each_problem_has_unique_id():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    ids = [p.problem_id for p in _all_problems()]
    assert len(ids) == len(set(ids))


def test_no_problem_permits_feature_twin_shortcut():
    """The correct answer's analog must NOT have the same button colors as m_target.

    If it did, a feature-matcher would get the right answer — which was
    the original shortcut the redesign was meant to remove.
    """
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    for p in _all_problems():
        if p.variant == "control":
            continue
        # The correct answer's button color should DIFFER from the memory
        # target's activating button color (which is what a naive model
        # would report if it just echoed the memory).
        correct_color = p.correct_answer["button_color"]
        # The feature match answer gives m_target's activating color.
        feature_color = p.metadata["feature_match_answer"]["button_color"]
        assert correct_color != feature_color, (
            f"{p.problem_id}: correct color == feature-match color "
            f"— feature-matching would give the right color"
        )
