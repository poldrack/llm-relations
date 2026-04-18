from llm_relations.problem import Problem
from llm_relations.scorer import score_answer, ScoreResult


def _make_problem() -> Problem:
    return Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="...",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "quib", "button_color": "green"},
        },
    )


def test_correct_answer_scored_as_correct():
    p = _make_problem()
    r = score_answer(p, {"analog": "mek", "button_color": "blue"})
    assert r == ScoreResult(is_correct=True, error_type=None)


def test_feature_match_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "zop", "button_color": "blue"})
    assert r == ScoreResult(is_correct=False, error_type="feature_match")


def test_positional_match_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "quib", "button_color": "green"})
    assert r == ScoreResult(is_correct=False, error_type="positional_match")


def test_other_error_classified():
    p = _make_problem()
    r = score_answer(p, {"analog": "mek", "button_color": "red"})
    assert r.is_correct is False
    assert r.error_type == "other"


def test_parse_error_classified():
    p = _make_problem()
    r = score_answer(p, None)
    assert r == ScoreResult(is_correct=False, error_type="parse_error")
