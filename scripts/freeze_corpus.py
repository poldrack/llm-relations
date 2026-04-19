"""Instantiate all problems with fixed seeds and write them to problems/.

The redesigned test suite includes six variants:
- baseline: 3 objects, position-based activation, color-disjoint correct
  analog, feature-twin distractor.
- feature_misleading: same as baseline + salience cue on feature twin.
- adversarial: same as baseline + linguistic decoy priming the twin.
- cross_domain: same relational schema over a different surface domain.
- scale: 3-object schema extended to n = 4..8 objects.
- control: same as baseline but with perception relations REMOVED —
  there is no structurally-correct answer. A relational solver should
  refuse or hedge; a feature-matcher will confidently pick the twin.
"""
from __future__ import annotations

from pathlib import Path

from llm_relations.problem import save_problem
from llm_relations.generator.baseline import generate_baseline
from llm_relations.generator.feature_misleading import generate_feature_misleading
from llm_relations.generator.scale import generate_scale
from llm_relations.generator.cross_domain import generate_cross_domain
from llm_relations.generator.adversarial import generate_adversarial
from llm_relations.generator.control import generate_control


PROBLEMS_DIR = Path(__file__).resolve().parent.parent / "problems"


# Config conventions:
# - correct_slot_index and feature_distractor_slot in {0, 1, 2} with
#   correct_slot_index > 0 so the positional-match distractor (= first
#   listed) never coincides with the correct answer.
# - feature_distractor_slot != correct_slot_index.
# - target_role is sampled by the generator from the RNG seed, so it
#   varies across instances.

BASELINE_CONFIGS = [
    # (seed, correct_slot_index, feature_distractor_slot)
    (1001, 1, 0),
    (1002, 2, 0),
    (1003, 1, 2),
    (1004, 2, 1),
    (1005, 1, 0),
]

FEATURE_MISLEADING_CONFIGS = [
    (2001, 1, 0),
    (2002, 2, 1),
    (2003, 1, 2),
    (2004, 2, 0),
    (2005, 1, 2),
]

SCALE_CONFIGS = [
    # (seed, n_objects)
    (3001, 4),
    (3002, 5),
    (3003, 6),
    (3004, 7),
    (3005, 8),
]

CROSS_DOMAIN_CONFIGS = [
    # (seed, memory_domain, perception_domain, correct_slot_index, feature_distractor_slot)
    # Each row pairs TWO distinct domains so the memory scenario and
    # perception scenario have different surface vocabulary and different
    # relational predicates. The generator additionally forces the two
    # scenarios to use different slot orders for their predicates.
    (4001, "org_chart", "garden", 2, 0),
    (4002, "garden", "building", 1, 2),
    (4003, "building", "enclosure", 2, 1),
    (4004, "enclosure", "vehicle_lot", 2, 1),
    (4005, "vehicle_lot", "org_chart", 1, 0),
]

ADVERSARIAL_CONFIGS = [
    # (seed, correct_slot_index, feature_distractor_slot, decoy_index)
    (5001, 1, 0, 0),
    (5002, 2, 1, 1),
    (5003, 2, 0, 2),
    (5004, 2, 0, 3),
    (5005, 1, 2, 4),
]

CONTROL_CONFIGS = [
    (6001, 1, 0),
    (6002, 2, 0),
    (6003, 1, 2),
    (6004, 2, 1),
    (6005, 1, 0),
]


def main() -> None:
    PROBLEMS_DIR.mkdir(exist_ok=True)

    # Clean out any stale JSONs from the previous schema. If deletion
    # fails (e.g. read-only mount), we still overwrite below; any
    # leftover names from the old schema will just sit alongside.
    for p in PROBLEMS_DIR.glob("*.json"):
        try:
            p.unlink()
        except OSError:
            pass

    for i, (seed, correct, distractor) in enumerate(BASELINE_CONFIGS):
        p = generate_baseline(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor) in enumerate(FEATURE_MISLEADING_CONFIGS):
        p = generate_feature_misleading(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, n) in enumerate(SCALE_CONFIGS):
        p = generate_scale(seed=seed, index=i, n_objects=n)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, memory_dom, perception_dom, correct, distractor) in enumerate(CROSS_DOMAIN_CONFIGS):
        p = generate_cross_domain(
            seed=seed,
            index=i,
            memory_domain=memory_dom,
            perception_domain=perception_dom,
            correct_slot_index=correct,
            feature_distractor_slot=distractor,
        )
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor, decoy) in enumerate(ADVERSARIAL_CONFIGS):
        p = generate_adversarial(
            seed=seed, index=i, correct_slot_index=correct,
            feature_distractor_slot=distractor, decoy_index=decoy,
        )
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    for i, (seed, correct, distractor) in enumerate(CONTROL_CONFIGS):
        p = generate_control(seed=seed, index=i, correct_slot_index=correct, feature_distractor_slot=distractor)
        save_problem(p, PROBLEMS_DIR / f"{p.problem_id}.json")

    print(f"Wrote {len(list(PROBLEMS_DIR.glob('*.json')))} problems to {PROBLEMS_DIR}")


if __name__ == "__main__":
    main()
