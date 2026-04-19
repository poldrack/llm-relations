"""Adversarial variant.

Shared schema, plus a linguistic decoy sentence appended to the
perception scenario that primes the feature-twin as the answer.
"""
from __future__ import annotations

from typing import Optional

from llm_relations.generator._common import build_problem_3
from llm_relations.problem import Problem


LINGUISTIC_DECOYS: list[str] = [
    "Notably, the {name}'s {color} button glows softly when anyone walks by.",
    "The {name}'s {color} button is described in the catalog as its primary control.",
    "People who have interacted with the {name} report that its {color} button is the one you press first.",
    "The {name} is famous for its prominent {color} button.",
    "The {name}'s {color} button is the one usually associated with activation in similar objects.",
]


def generate_adversarial(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
    decoy_index: int,
    target_role: Optional[int] = None,
    activation_position: Optional[str] = None,
) -> Problem:
    assert 0 <= decoy_index < len(LINGUISTIC_DECOYS)

    return build_problem_3(
        seed=seed,
        index=index,
        variant="adversarial",
        problem_id=f"adversarial_{index:02d}",
        correct_slot_index=correct_slot_index,
        feature_distractor_slot=feature_distractor_slot,
        target_role=target_role,
        activation_position=activation_position,
        feature_twin_decoy_template=LINGUISTIC_DECOYS[decoy_index],
        extra_metadata={"decoy_index": decoy_index},
    )
