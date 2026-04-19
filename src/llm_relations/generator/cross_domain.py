"""Cross-domain variant.

A *true* cross-domain analogy test: the memory scenario is narrated in
one surface domain (e.g. an organization chart with employees and
specialties) while the perception scenario is narrated in a different
surface domain (e.g. a garden of plants with leaves). The structural
graph is identical on both sides, but the surface vocabulary,
category nouns, feature nouns, and relation predicates all differ.

Two anti-shortcut mechanisms are enabled here:

1. ``memory_domain`` and ``perception_domain`` must differ. A model
   cannot rely on string-matching predicate names between the two
   scenarios to recover the mapping.

2. The two relation predicates of each DomainSpec (a "vertical"-ish
   and a "horizontal"-ish one) are assigned to template slots in an
   order that is *guaranteed to differ* between the two scenarios
   (``force_different_slot_orders=True``). This ensures memory's
   vertical-semantic predicate is paired with perception's
   horizontal-semantic predicate by template position. A model using
   predicate-semantic alignment ("match the hierarchical-meaning
   sentence to the hierarchical-meaning sentence") gets the role
   mapping wrong; only a model that uses structural (template-position)
   role identification succeeds.

If ``memory_domain == perception_domain``, a ``ValueError`` is raised.
"""
from __future__ import annotations

from typing import Optional

from llm_relations.generator._common import DomainSpec, build_problem_3
from llm_relations.problem import Problem


_DOMAINS: dict[str, DomainSpec] = {
    "org_chart": DomainSpec(
        memory_container_phrase="in an organization",
        perception_container_phrase="in another organization",
        category_singular="employee",
        category_plural="employees",
        feature_noun="specialty",
        feature_prefix="color-coded",
        relation_vertical="reports-to",
        relation_horizontal="sits-beside",
        activation_phrase="Assigning the {feature_noun} at the {position} position of the {name}",
        instruction_verb="assigning",
    ),
    "garden": DomainSpec(
        memory_container_phrase="in a garden",
        perception_container_phrase="in another garden",
        category_singular="plant",
        category_plural="plants",
        feature_noun="leaf",
        feature_prefix="colored",
        relation_vertical="is-growing-under",
        relation_horizontal="grows-beside",
        activation_phrase="Clipping the {feature_noun} at the {position} of the {name}",
        instruction_verb="clipping",
    ),
    "building": DomainSpec(
        memory_container_phrase="in a building",
        perception_container_phrase="in another building",
        category_singular="room",
        category_plural="rooms",
        feature_noun="fixture",
        feature_prefix="colored",
        relation_vertical="is-directly-below",
        relation_horizontal="is-adjacent-to",
        activation_phrase="Operating the {feature_noun} at the {position} of the {name}",
        instruction_verb="operating",
    ),
    "enclosure": DomainSpec(
        memory_container_phrase="in an enclosure",
        perception_container_phrase="in another enclosure",
        category_singular="animal",
        category_plural="animals",
        feature_noun="marking",
        feature_prefix="colored",
        relation_vertical="lies-beneath",
        relation_horizontal="roams-beside",
        activation_phrase="Touching the {feature_noun} at the {position} of the {name}",
        instruction_verb="touching",
    ),
    "vehicle_lot": DomainSpec(
        memory_container_phrase="in a lot",
        perception_container_phrase="in another lot",
        category_singular="vehicle",
        category_plural="vehicles",
        feature_noun="panel",
        feature_prefix="colored",
        relation_vertical="is-stacked-below",
        relation_horizontal="is-parked-to-the-left-of",
        activation_phrase="Switching the {feature_noun} at the {position} of the {name}",
        instruction_verb="switching",
    ),
}

DOMAINS = list(_DOMAINS.keys())


def generate_cross_domain(
    seed: int,
    index: int,
    memory_domain: str,
    perception_domain: str,
    correct_slot_index: int,
    feature_distractor_slot: int,
    target_role: Optional[int] = None,
    activation_position: Optional[str] = None,
) -> Problem:
    assert memory_domain in _DOMAINS, f"unknown memory_domain: {memory_domain}"
    assert perception_domain in _DOMAINS, f"unknown perception_domain: {perception_domain}"
    if memory_domain == perception_domain:
        raise ValueError(
            f"cross_domain requires distinct domains; got {memory_domain!r} for both scenarios"
        )

    m_dom = _DOMAINS[memory_domain]
    p_dom = _DOMAINS[perception_domain]

    return build_problem_3(
        seed=seed,
        index=index,
        variant="cross_domain",
        problem_id=f"cross_domain_{index:02d}",
        correct_slot_index=correct_slot_index,
        feature_distractor_slot=feature_distractor_slot,
        target_role=target_role,
        activation_position=activation_position,
        memory_domain=m_dom,
        perception_domain=p_dom,
        scramble_relation_slot_order=True,
        force_different_slot_orders=True,
        extra_metadata={
            "memory_domain": memory_domain,
            "perception_domain": perception_domain,
        },
    )
