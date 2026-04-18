from __future__ import annotations

import random

from llm_relations.problem import Problem
from llm_relations.palette import draw_nonsense_words, draw_colors


_INSTRUCTION = (
    "I'm going to describe two scenarios. In the memory scenario, a novel object called "
    "a {target_memory} has a property: it can be activated by pressing one of its buttons. "
    "Your job is to figure out which object in the perception scenario is the {target_memory}-analog, "
    "and therefore which button on it can be activated."
)

LINGUISTIC_DECOYS: list[str] = [
    "Notably, the {name}'s {color} button glows softly when anyone walks by.",
    "The {name}'s {color} button is described in the catalog as its primary control.",
    "People who have interacted with the {name} report that its {color} button is the one you press first.",
    "The {name} is famous for its prominent {color} button.",
    "The {name}'s {color} button is the one usually associated with activation in similar objects.",
]


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def generate_adversarial(
    seed: int,
    index: int,
    correct_slot_index: int,
    feature_distractor_slot: int,
    decoy_index: int,
) -> Problem:
    assert 0 <= correct_slot_index < 3
    assert 0 <= feature_distractor_slot < 3
    assert correct_slot_index != feature_distractor_slot
    assert 0 <= decoy_index < len(LINGUISTIC_DECOYS)

    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=6)
    m0, m1, m2 = words[:3]
    perception_words = words[3:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    m0_buttons = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    m1_buttons = [(other1, "top"), (target_color, "bottom")]
    m2_buttons = [(other2, "top"), (target_color, "side")]

    memory_text = " ".join([
        f"Memory scenario: There are three objects on a table: a {m0}, a {m1}, and a {m2}.",
        f"The {m0} is underneath the {m1}.",
        f"The {m1} is to-the-left-of the {m2}.",
        _describe_object(m0, m0_buttons),
        _describe_object(m1, m1_buttons),
        _describe_object(m2, m2_buttons),
        f"Pressing the {target_color} button on the {m0} activates it.",
    ])

    correct_analog = perception_words[0]
    list_order = [None, None, None]
    list_order[correct_slot_index] = correct_analog
    remaining = [perception_words[1], perception_words[2]]
    distractor = remaining[0]
    list_order[feature_distractor_slot] = distractor
    other_slot = [i for i in range(3) if list_order[i] is None][0]
    list_order[other_slot] = remaining[1]

    perception_buttons = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(target_color, "top"), (target_color, "side"), (other1, "bottom")],
        remaining[1]: [(other2, "top"), (other1, "bottom")],
    }

    decoy_sentence = LINGUISTIC_DECOYS[decoy_index].format(name=distractor, color=target_color)

    perception_text = " ".join([
        f"Perception scenario: There are three objects on a shelf: "
        f"a {list_order[0]}, a {list_order[1]}, and a {list_order[2]}.",
        f"The {perception_words[0]} is underneath the {perception_words[1]}.",
        f"The {perception_words[1]} is to-the-left-of the {perception_words[2]}.",
        _describe_object(list_order[0], perception_buttons[list_order[0]]),
        _describe_object(list_order[1], perception_buttons[list_order[1]]),
        _describe_object(list_order[2], perception_buttons[list_order[2]]),
        decoy_sentence,
    ])

    question = f"Which object in the perception scenario is the {m0}-analog, and which button activates it?"

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=m0),
        memory_text,
        perception_text,
        question,
    ])

    positional_analog = list_order[0]
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"adversarial_{index:02d}",
        variant="adversarial",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": 3,
            "seed": seed,
            "correct_slot_index": correct_slot_index,
            "feature_distractor_slot": feature_distractor_slot,
            "decoy_index": decoy_index,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
