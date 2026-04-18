from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from llm_relations.problem import Problem


ErrorType = Literal["feature_match", "positional_match", "other", "parse_error"]


@dataclass(frozen=True)
class ScoreResult:
    is_correct: bool
    error_type: Optional[ErrorType]


def score_answer(problem: Problem, answer: Optional[dict[str, str]]) -> ScoreResult:
    """Score a parsed model answer against the problem's ground truth.

    `answer=None` indicates a parse failure upstream.
    Error types correspond to Hummel & Heaton's predicted failure modes:
    - feature_match: picked the object with matching surface features
    - positional_match: picked the object in the same list position
    - other: some other wrong answer
    - parse_error: model did not emit a parseable answer
    """
    if answer is None:
        return ScoreResult(is_correct=False, error_type="parse_error")

    if answer == problem.correct_answer:
        return ScoreResult(is_correct=True, error_type=None)

    if answer == problem.metadata.get("feature_match_answer"):
        return ScoreResult(is_correct=False, error_type="feature_match")

    if answer == problem.metadata.get("positional_match_answer"):
        return ScoreResult(is_correct=False, error_type="positional_match")

    return ScoreResult(is_correct=False, error_type="other")
