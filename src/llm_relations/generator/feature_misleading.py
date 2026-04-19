"""Feature-misleading variant.

Same shared schema as baseline, but with an extra salience cue attached
to the feature-twin's description to strengthen the feature lure. The
correct answer is still determined purely by relational structure; a
feature-matching model is even more tempted to pick the twin.
"""
from __future__ import annotations

from typing import Optional

from llm_relations.generator._common import build_problem_3
from llm_relations.problem import Problem


def generate_feature_misleading(
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
        variant="feature_misleading",
        problem_id=f"feature_misleading_{index:02d}",
        correct_slot_index=correct_slot_index,
        feature_distractor_slot=feature_distractor_slot,
        target_role=target_role,
        activation_position=activation_position,
        make_feature_twin_more_tempting=True,
    )
