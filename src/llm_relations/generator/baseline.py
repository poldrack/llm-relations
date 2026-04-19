"""Baseline variant of the redesigned relational-reasoning task.

3-object memory and perception scenarios with position-based activation,
a color-disjoint correct analog, and a feature-twin distractor. See
_common.py for the full design rationale.
"""
from __future__ import annotations

from typing import Optional

from llm_relations.generator._common import build_problem_3
from llm_relations.problem import Problem


def generate_baseline(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
    target_role: Optional[int] = None,
    activation_position: Optional[str] = None,
) -> Problem:
    return build_problem_3(
        seed=seed,
        index=index,
        variant="baseline",
        problem_id=f"baseline_{index:02d}",
        correct_slot_index=correct_slot_index,
        feature_distractor_slot=feature_distractor_slot,
        target_role=target_role,
        activation_position=activation_position,
    )
