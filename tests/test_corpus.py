from pathlib import Path

import pytest

from llm_relations.problem import load_problem

PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"


def _all_problems():
    files = sorted(PROBLEMS_DIR.glob("*.json"))
    return [load_problem(f) for f in files]


def test_corpus_has_25_problems():
    if not PROBLEMS_DIR.exists() or not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen — run scripts/freeze_corpus.py")
    assert len(list(PROBLEMS_DIR.glob("*.json"))) == 25


def test_corpus_covers_all_five_variants():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    problems = _all_problems()
    variants = {p.variant for p in problems}
    assert variants == {"baseline", "feature_misleading", "scale", "cross_domain", "adversarial"}
    for variant in variants:
        assert sum(1 for p in problems if p.variant == variant) == 5


def test_each_problem_has_internally_consistent_ground_truth():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    for p in _all_problems():
        # The correct analog must appear in the prompt text.
        assert p.correct_answer["analog"] in p.prompt_text, f"{p.problem_id}: analog not in prompt"
        # Distractors must differ from the correct answer.
        fma = p.metadata["feature_match_answer"]["analog"]
        pma = p.metadata["positional_match_answer"]["analog"]
        assert fma != p.correct_answer["analog"], f"{p.problem_id}: feature_match == correct"
        assert pma != p.correct_answer["analog"], f"{p.problem_id}: positional_match == correct"


def test_each_problem_has_unique_id():
    if not list(PROBLEMS_DIR.glob("*.json")):
        pytest.skip("Corpus not yet frozen")
    ids = [p.problem_id for p in _all_problems()]
    assert len(ids) == len(set(ids))
