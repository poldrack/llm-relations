"""Instantiate all 25 problems with fixed seeds and write them to problems/."""
from __future__ import annotations

from pathlib import Path

from llm_relations.problem import save_problem
from llm_relations.generator.baseline import generate_baseline
from llm_relations.generator.feature_misleading import generate_feature_misleading
from llm_relations.generator.scale import generate_scale
from llm_relations.generator.cross_domain import generate_cross_domain
from llm_relations.generator.adversarial import generate_adversarial


PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"


# (seed, correct_slot_index, feature_distractor_slot) per instance.
# IMPORTANT: correct_slot_index must be 1 or 2 (never 0). When it is 0,
# the positional-match distractor coincides with the correct answer, and
# we lose the ability to classify positional errors on that instance.
# 5 distinct configs using slot indices in {1, 2}.
BASELINE_CONFIGS = [
    (1001, 1, 0),
    (1002, 2, 0),
    (1003, 1, 2),
    (1004, 2, 1),
    (1005, 1, 0),  # varied by seed only
]

FEATURE_MISLEADING_CONFIGS = [
    (2001, 1, 0),
    (2002, 2, 1),
    (2003, 1, 2),
    (2004, 2, 0),
    (2005, 1, 2),
]

# Scale test: one instance per size.
# Seeds 3001 and 3002 produced degenerate instances where the RNG shuffle
# placed the correct analog first in list_order, making positional_match == correct.
# Seeds 3002 (n=4) and 3000 (n=5) are verified non-degenerate replacements.
SCALE_CONFIGS = [
    (3002, 4),
    (3000, 5),
    (3003, 6),
    (3004, 7),
    (3005, 8),
]

# One instance per domain. correct_slot_index in {1, 2}.
CROSS_DOMAIN_CONFIGS = [
    (4001, "org_chart", 2, 0),
    (4002, "garden", 1, 2),
    (4003, "building", 2, 1),
    (4004, "enclosure", 2, 1),
    (4005, "vehicle_lot", 1, 0),
]

# Five linguistic decoys (one each). correct_slot_index in {1, 2}.
ADVERSARIAL_CONFIGS = [
    (5001, 1, 0, 0),
    (5002, 2, 1, 1),
    (5003, 2, 0, 2),
    (5004, 2, 0, 3),
    (5005, 1, 2, 4),
]


def main() -> None:
    PROBLEMS_DIR.mkdir(exist_ok=True)

    for i, (seed, correct, distractor) in enumerate(BASELINE_CONFIGS):
        p = generate_baseline(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor) in enumerate(FEATURE_MISLEADING_CONFIGS):
        p = generate_feature_misleading(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, n) in enumerate(SCALE_CONFIGS):
        p = generate_scale(seed=seed, index=i, n_objects=n)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, domain, correct, distractor) in enumerate(CROSS_DOMAIN_CONFIGS):
        p = generate_cross_domain(seed=seed, index=i, domain=domain, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor, decoy) in enumerate(ADVERSARIAL_CONFIGS):
        p = generate_adversarial(
            seed=seed, index=i, correct_slot_index=correct,
            feature_distractor_slot=distractor, decoy_index=decoy,
        )
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    print(f"Wrote {len(list(PROBLEMS_DIR.glob('*.json')))} problems to {PROBLEMS_DIR}")


if __name__ == "__main__":
    main()
