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

_RELATION_CYCLE = [
    "is underneath",
    "is to-the-left-of",
    "is above",
    "is to-the-right-of",
]


def _describe_object(name: str, buttons: list[tuple[str, str]]) -> str:
    parts = [f"a {c} button on {p}" for c, p in buttons]
    if len(parts) == 1:
        body = parts[0]
    elif len(parts) == 2:
        body = f"{parts[0]} and {parts[1]}"
    else:
        body = ", ".join(parts[:-1]) + f", and {parts[-1]}"
    return f"The {name} has {body}."


def _chain_relations(names: list[str]) -> list[str]:
    out = []
    for i in range(len(names) - 1):
        rel = _RELATION_CYCLE[i % len(_RELATION_CYCLE)]
        out.append(f"The {names[i]} {rel} the {names[i+1]}.")
    return out


def _build_memory_buttons(
    rng: random.Random,
    words: list[str],
    target_color: str,
    other1: str,
    other2: str,
) -> dict[str, list[tuple[str, str]]]:
    buttons: dict[str, list[tuple[str, str]]] = {
        words[0]: [(target_color, "top"), (other1, "side"), (other2, "bottom")]
    }
    for w in words[1:]:
        buttons[w] = [
            (rng.choice([other1, other2]), "top"),
            (target_color, rng.choice(["bottom", "side", "front", "back"])),
        ]
    return buttons


def _build_perception_buttons(
    rng: random.Random,
    words: list[str],
    target_color: str,
    other1: str,
    other2: str,
) -> dict[str, list[tuple[str, str]]]:
    correct_analog = words[0]
    distractor = words[1]
    buttons: dict[str, list[tuple[str, str]]] = {
        correct_analog: [(target_color, "top"), (other1, "side"), (other2, "bottom")],
        distractor: [(other1, "top"), (target_color, "side"), (other2, "bottom")],
    }
    for w in words[2:]:
        buttons[w] = [
            (rng.choice([other1, other2]), "top"),
            (rng.choice([other1, other2]), "bottom"),
        ]
    return buttons


def _build_memory_text(words: list[str], buttons: dict[str, list[tuple[str, str]]]) -> str:
    n = len(words)
    listing = (
        ", ".join(f"a {w}" for w in words[:-1]) + f", and a {words[-1]}"
    )
    parts = [
        f"Memory scenario: There are {n} objects on a table: {listing}.",
        *_chain_relations(words),
        *(_describe_object(w, buttons[w]) for w in words),
        f"Pressing the {buttons[words[0]][0][0]} button on the {words[0]} activates it.",
    ]
    return " ".join(parts)


def _build_perception_text(
    words: list[str],
    list_order: list[str],
    buttons: dict[str, list[tuple[str, str]]],
) -> str:
    n = len(words)
    listing = (
        ", ".join(f"a {w}" for w in list_order[:-1]) + f", and a {list_order[-1]}"
    )
    parts = [
        f"Perception scenario: There are {n} objects on a shelf: {listing}.",
        *_chain_relations(words),
        *(_describe_object(w, buttons[w]) for w in list_order),
    ]
    return " ".join(parts)


def generate_scale(seed: int, index: int, n_objects: int) -> Problem:
    assert n_objects >= 3
    rng = random.Random(seed)
    words = draw_nonsense_words(rng, n=2 * n_objects)
    memory_words = words[:n_objects]
    perception_words = words[n_objects:]

    colors = draw_colors(rng, n=3)
    target_color, other1, other2 = colors

    memory_buttons = _build_memory_buttons(rng, memory_words, target_color, other1, other2)
    perception_buttons = _build_perception_buttons(rng, perception_words, target_color, other1, other2)

    perception_list_order = perception_words[:]
    rng.shuffle(perception_list_order)

    memory_text = _build_memory_text(memory_words, memory_buttons)
    perception_text = _build_perception_text(perception_words, perception_list_order, perception_buttons)

    correct_analog = perception_words[0]
    distractor = perception_words[1]
    question = (
        f"Which object in the perception scenario is the {memory_words[0]}-analog, "
        "and which button activates it?"
    )

    prompt = "\n\n".join([
        _INSTRUCTION.format(target_memory=memory_words[0]),
        memory_text,
        perception_text,
        question,
    ])

    positional_analog = perception_list_order[0]
    positional_button = next(c for c, p in perception_buttons[positional_analog] if p == "top")

    return Problem(
        problem_id=f"scale_{index:02d}",
        variant="scale",
        prompt_text=prompt,
        correct_answer={"analog": correct_analog, "button_color": target_color},
        metadata={
            "n_objects": n_objects,
            "seed": seed,
            "memory_words": memory_words,
            "perception_words": perception_words,
            "perception_list_order": perception_list_order,
            "feature_match_answer": {"analog": distractor, "button_color": target_color},
            "positional_match_answer": {"analog": positional_analog, "button_color": positional_button},
        },
    )
