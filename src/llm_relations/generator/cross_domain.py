from __future__ import annotations

import random
from dataclasses import dataclass

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


@dataclass(frozen=True)
class Domain:
    name: str
    container: str                # "on a table"
    category: str                 # "objects"
    feature_noun: str             # "button"
    feature_prefix: str           # what modifier describes the feature (e.g., a color)
    relation_vertical: str        # "is underneath"
    relation_horizontal: str      # "is to-the-left-of"
    activation_phrase: str        # "Pressing the {feature} {feature_noun} on the {name}"


_DOMAINS: dict[str, Domain] = {
    "org_chart": Domain(
        name="org_chart",
        container="in an organization",
        category="employees",
        feature_noun="specialty",
        feature_prefix="color-coded",
        relation_vertical="reports-to",
        relation_horizontal="sits-beside",
        activation_phrase="Assigning the {feature} {feature_noun} to the {name}",
    ),
    "garden": Domain(
        name="garden",
        container="in a garden",
        category="plants",
        feature_noun="leaf",
        feature_prefix="colored",
        relation_vertical="is-growing-under",
        relation_horizontal="is-to-the-left-of",
        activation_phrase="Clipping the {feature} {feature_noun} from the {name}",
    ),
    "building": Domain(
        name="building",
        container="in a building",
        category="rooms",
        feature_noun="fixture",
        feature_prefix="colored",
        relation_vertical="is-directly-below",
        relation_horizontal="is-adjacent-to",
        activation_phrase="Operating the {feature} {feature_noun} in the {name}",
    ),
    "enclosure": Domain(
        name="enclosure",
        container="in an enclosure",
        category="animals",
        feature_noun="marking",
        feature_prefix="colored",
        relation_vertical="is-beneath",
        relation_horizontal="is-to-the-left-of",
        activation_phrase="Touching the {feature} {feature_noun} on the {name}",
    ),
    "vehicle_lot": Domain(
        name="vehicle_lot",
        container="in a lot",
        category="vehicles",
        feature_noun="panel",
        feature_prefix="colored",
        relation_vertical="is-parked-beneath",
        relation_horizontal="is-parked-to-the-left-of",
        activation_phrase="Switching the {feature} {feature_noun} on the {name}",
    ),
}

DOMAINS = list(_DOMAINS.keys())

_POSITIONS = ["top", "bottom", "side"]


def _describe(name: str, feature_noun: str, feature_prefix: str, features: list[tuple[str, str]]) -> str:
    parts = [f"a {feat} {feature_prefix} {feature_noun} on {pos}" for feat, pos in features]
    if len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_cross_domain(
    seed: int,
    index: int,
    domain: str,
    correct_slot_index: int,
    feature_distractor_slot: int,
) -> Problem:
    assert domain in _DOMAINS, f"unknown domain: {domain}"
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot

    d = _DOMAINS[domain]
    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    m0_feats = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_feats = [(other1, "top"), (target_color, "bottom")]
    m2_feats = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three {d.category} {d.container}: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} {d.relation_vertical} the {m1}.",
        f"The {m1} {d.relation_horizontal} the {m2}.",
        _describe(m0, d.feature_noun, d.feature_prefix, m0_feats),
        _describe(m1, d.feature_noun, d.feature_prefix, m1_feats),
        _describe(m2, d.feature_noun, d.feature_prefix, m2_feats),
        d.activation_phrase.format(feature=target_color, feature_noun=d.feature_noun, name=m0) + " activates it.",
    ])

    correct_analog = perception_words[0]
    list_order: list[str | None] = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    distractor = remaining[0]
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    perception_feats = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(other1, "top"), (target_color, "side")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    perception_text = " ".join([
        f"Perception scenario: There are three {d.category} {d.container}: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} {d.relation_vertical} the {perception_words[1]}.",
        f"The {perception_words[1]} {d.relation_horizontal} the {perception_words[2]}.",
        _describe(list_order[0], d.feature_noun, d.feature_prefix, perception_feats[list_order[0]]),
        _describe(list_order[1], d.feature_noun, d.feature_prefix, perception_feats[list_order[1]]),
        _describe(list_order[2], d.feature_noun, d.feature_prefix, perception_feats[list_order[2]]),
    ])

    question = (
        f"Which {d.category[:-1]} in the perception scenario is the {m0}-analog, "
        f"and which {d.feature_prefix} {d.feature_noun} activates it?"
    )

    instruction = (
        f"I'm going to describe two scenarios. In the memory scenario, a novel {d.category[:-1]} called "
        f"a {m0} has a property: it can be activated by one of its {d.feature_prefix} {d.feature_noun}s. "
        f"Your job is to figure out which {d.category[:-1]} in the perception scenario is the {m0}-analog, "
        f"and therefore which {d.feature_prefix} {d.feature_noun} on it can be activated."
    )

    prompt = "\n\n".join([instruction, memory_text, perception_text, question])

    positional_analog = list_order[0]
    positional_feature = next(c for c, p in perception_feats[positional_analog] if p == "top")

    return Problem(
        problem_id=f"cross_domain_{index:02d}",
        variant="cross_domain",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "domain": domain,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_feature},
        },
    )
