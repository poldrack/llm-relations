import json
from pathlib import Path

from llm_relations.problem import Problem, load_problem, save_problem


def test_problem_constructs_with_required_fields():
    p = Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="...",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={"n_objects": 3, "seed": 42},
    )
    assert p.variant == "baseline"
    assert p.correct_answer["analog"] == "mek"


def test_problem_round_trips_through_json(tmp_path: Path):
    p = Problem(
        problem_id="baseline_00",
        variant="baseline",
        prompt_text="prompt",
        correct_answer={"analog": "mek", "button_color": "blue"},
        metadata={
            "n_objects": 3,
            "seed": 42,
            "feature_match_answer": {"analog": "zop", "button_color": "blue"},
            "positional_match_answer": {"analog": "zop", "button_color": "green"},
        },
    )
    path = tmp_path / "p.json"
    save_problem(p, path)
    loaded = load_problem(path)
    assert loaded == p


def test_problem_rejects_missing_correct_answer_keys():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Problem(
            problem_id="x",
            variant="baseline",
            prompt_text="p",
            correct_answer={"analog": "mek"},  # missing button_color
            metadata={},
        )
