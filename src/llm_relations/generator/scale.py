"""Scale variant.

Extends the shared schema to n_objects >= 4, chaining relations across
the sequence so that each object has a distinct relational role. The
same invariants hold as in the 3-object variants: all perception objects
have 3 buttons at canonical slots, the correct analog uses a disjoint
color palette from the memory target, and a feature-twin distractor
exactly copies the target's buttons while sitting at a different role.
"""
from __future__ import annotations

import random
from typing import Optional

from llm_relations.generator._common import (
    BUTTON_SLOTS,
    buttons_from_triple,
    describe_object,
    draw_three_color_triples,
    pick_feature_twin_role,
    pick_target_role,
)
from llm_relations.palette import draw_nonsense_words
from llm_relations.problem import Problem


# A cycle of relations chained along the role sequence.
_RELATION_CYCLE = [
    "is underneath",
    "is to-the-left-of",
    "is above",
    "is to-the-right-of",
]


def _chain_relation_sentences(names_by_role: list[str]) -> list[str]:
    out = []
    for i in range(len(names_by_role) - 1):
        rel = _RELATION_CYCLE[i % len(_RELATION_CYCLE)]
        out.append(f"The {names_by_role[i]} {rel} the {names_by_role[i+1]}.")
    return out


def generate_scale(
    seed: int,
    index: int,
    n_objects: int,
    target_role: Optional[int] = None,
    activation_position: Optional[str] = None,
) -> Problem:
    assert n_objects >= 4
    rng = random.Random(seed)

    words = draw_nonsense_words(rng, n=2 * n_objects)
    memory_names = words[:n_objects]
    perception_names = words[n_objects:]

    target_colors, correct_colors, other_colors = draw_three_color_triples(rng)

    if target_role is None:
        target_role = pick_target_role(rng, n_objects)
    assert 0 <= target_role < n_objects

    feature_twin_role = pick_feature_twin_role(rng, n_objects, target_role)

    if activation_position is None:
        activation_position = rng.choice(list(BUTTON_SLOTS))
    assert activation_position in BUTTON_SLOTS

    # Memory buttons. Non-target memory objects mix BOTH correct_colors
    # and other_colors so that neither palette is "novel" relative to
    # memory — breaking the "find perception object with colors never
    # mentioned in memory" shortcut.
    m_target_buttons = buttons_from_triple(rng, target_colors)
    memory_buttons = {memory_names[target_role]: m_target_buttons}
    non_target_roles = [r for r in range(n_objects) if r != target_role]
    for i, r in enumerate(non_target_roles):
        pool = correct_colors if (i % 2 == 0) else other_colors
        memory_buttons[memory_names[r]] = buttons_from_triple(rng, pool)
    # Shuffle which non-target roles get which palette so it's not
    # always the same alternation.
    rng.shuffle(non_target_roles)
    for i, r in enumerate(non_target_roles):
        pool = correct_colors if (i % 2 == 0) else other_colors
        memory_buttons[memory_names[r]] = buttons_from_triple(rng, pool)

    # Perception buttons: correct analog (disjoint colors),
    # feature twin (exact copy), all others (other_colors, independently shuffled)
    p_correct_name = perception_names[target_role]
    p_twin_name = perception_names[feature_twin_role]
    p_correct_buttons = buttons_from_triple(rng, correct_colors)
    p_twin_buttons = m_target_buttons

    perception_buttons = {}
    for r in range(n_objects):
        name = perception_names[r]
        if r == target_role:
            perception_buttons[name] = p_correct_buttons
        elif r == feature_twin_role:
            perception_buttons[name] = p_twin_buttons
        else:
            perception_buttons[name] = buttons_from_triple(rng, other_colors)

    # Shuffle perception list order. Guarantee the correct analog is
    # NOT listed first (so the positional-match heuristic doesn't
    # coincide with the correct answer).
    list_order = perception_names[:]
    rng.shuffle(list_order)
    # If the correct analog ended up first, swap it with index 1.
    if list_order[0] == p_correct_name:
        list_order[0], list_order[1] = list_order[1], list_order[0]

    # Memory text
    m_listing = ", ".join(f"a {n}" for n in memory_names[:-1]) + f", and a {memory_names[-1]}"
    memory_sentences = [
        f"Memory scenario: There are {n_objects} objects on a table: {m_listing}.",
        *_chain_relation_sentences(memory_names),
        *(describe_object(name, memory_buttons[name]) for name in memory_names),
        f"Pressing the button on the {activation_position} of the {memory_names[target_role]} activates it.",
    ]
    memory_text = " ".join(memory_sentences)

    # Perception text
    p_listing = ", ".join(f"a {n}" for n in list_order[:-1]) + f", and a {list_order[-1]}"
    perception_sentences = [
        f"Perception scenario: There are {n_objects} objects on a shelf: {p_listing}.",
        *_chain_relation_sentences(perception_names),
        *(describe_object(name, perception_buttons[name]) for name in list_order),
    ]
    perception_text = " ".join(perception_sentences)

    m_target_name = memory_names[target_role]
    instruction = (
        f"I'm going to describe two scenarios. In the memory scenario, a novel object called "
        f"a {m_target_name} has a property: it can be activated by pressing the button at a "
        f"specific position on it. Your job is to figure out which object in the perception "
        f"scenario is the {m_target_name}-analog, and therefore which button on it activates it."
    )
    question = (
        f"Which object in the perception scenario is the {m_target_name}-analog, "
        "and which button activates it? "
        "Answer as a fenced JSON block with keys `analog` and `button_color`."
    )

    prompt = "\n\n".join([instruction, memory_text, perception_text, question])

    correct_answer = {
        "analog": p_correct_name,
        "button_color": p_correct_buttons.color_at(activation_position),
    }
    feature_match_answer = {
        "analog": p_twin_name,
        "button_color": p_twin_buttons.color_at(activation_position),
    }
    positional_analog = list_order[0]
    positional_match_answer = {
        "analog": positional_analog,
        "button_color": perception_buttons[positional_analog].color_at(activation_position),
    }

    return Problem(
        problem_id=f"scale_{index:02d}",
        variant="scale",
        prompt_text=prompt,
        correct_answer=correct_answer,
        metadata={
            "n_objects": n_objects,
            "seed": seed,
            "memory_words": memory_names,
            "perception_words": perception_names,
            "perception_list_order": list_order,
            "target_role": target_role,
            "feature_twin_role": feature_twin_role,
            "activation_position": activation_position,
            "feature_match_answer": feature_match_answer,
            "positional_match_answer": positional_match_answer,
            "memory_target_name": m_target_name,
            "structural_correct_analog": p_correct_name,
            "structural_correct_button_color": p_correct_buttons.color_at(activation_position),
        },
    )
