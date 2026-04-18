from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Answer(BaseModel):
    analog: str
    button_color: str


class Problem(BaseModel):
    problem_id: str
    variant: str
    prompt_text: str
    correct_answer: dict[str, str]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("correct_answer")
    @classmethod
    def _check_answer_keys(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v.keys()) != {"analog", "button_color"}:
            raise ValueError("correct_answer must have exactly keys: analog, button_color")
        return v


def save_problem(problem: Problem, path: Path) -> None:
    path.write_text(json.dumps(problem.model_dump(), indent=2, sort_keys=True))


def load_problem(path: Path) -> Problem:
    return Problem.model_validate_json(path.read_text())
