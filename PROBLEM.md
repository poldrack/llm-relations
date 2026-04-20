# On Hummel & Heaton's Argument About LLMs and Symbolic Thought

This is a thoughtful paper, and the authors are making a more careful argument than a quick read might suggest. Let me separate what I think is right, what I think is questionable, and what remains genuinely open.

## What the paper actually claims

First, a clarification: the authors don't quite claim that there is a definite set of tasks LLMs cannot perform. Their claim is more measured. They argue that (1) dynamic binding, multiplace predicates, and structure mapping are jointly the minimal computational requirements for symbolic thought in a neural architecture, (2) LLMs appear to lack genuine implementations of these, and (3) this predicts a specific failure mode: LLMs should struggle when tasks require binding familiar elements into statistically unusual configurations, because their representations are "entangled" with training distributions. Their coffee-maker simulations and the seagull/plane/tent image example are meant to illustrate this.

## What I find reasonably compelling

The theoretical core is well-motivated. The distinction between single-place predicates like `can-walk-on(surface)` and genuine multiplace relations like `can-fit-inside(x,y)` where both arguments vary is a real computational distinction, not just a notational one. Their point that egocentric affordances can be "degenerate" relations (collapsible to single-place predicates) while allocentric spatial relations cannot, is a sharp observation that matters.

The entanglement prediction is also empirically testable and seems to hold up. The seagull/plane/tent example isn't cherry-picked in the damning way it might appear: the broader literature on compositional generalization (Lake & Baroni 2018, which they cite, plus many follow-ups) has repeatedly found that neural networks struggle in predictable ways when asked to bind familiar elements in unfamiliar configurations. The sample efficiency gap between humans and LLMs is also real and large — children do learn words from two or three exposures, while LLMs require corpora that would take millennia to read.

The observation that architectures with structure mapping cannot compensate for lacking multiplace predicates (their MO failing on the RO task, and vice versa) is a genuinely interesting result. If it holds, it means these two capacities are computationally independent, which has real implications for how we think about cognitive architectures.

## Where I think the argument is weaker

The move from "LISA demonstrates what these capacities do" to "LLMs lack these capacities" is doing a lot of work and deserves more scrutiny than the paper gives it. Their footnote 8 acknowledges that transformer attention involves dynamic weight modulation but dismisses it as not "fully flexible dynamic binding." This is asserted rather than argued. There's genuine ongoing research on whether attention mechanisms implement something functionally equivalent to dynamic binding, or whether in-context learning constitutes a form of structure mapping. The paper would be stronger if it engaged with this rather than defining it away.

The seagull/plane/tent demonstration is suggestive but not decisive. It shows current frontier models struggle with a particular class of unusual spatial compositions in image generation. But image generation models have specific architectural constraints (diffusion processes, CLIP-based conditioning) that may not reflect limits of symbolic reasoning in LLMs more broadly. Text-based compositional tests would be more directly relevant to the symbolic-thought claim. And the authors themselves note that such errors are a "moving target" as training expands — which raises the question of whether this is a fundamental architectural limit or an empirical gap that shrinks with scale and training strategy.

Webb, Holyoak, and Lu (2023), which the authors cite, showed GPT-3 solving analogies that seem to require structure mapping. The authors wave this off by suggesting the model might be exploiting training statistics, but this is the same move critics of symbolic cognition make against behavioral evidence in animals. You can't have it both ways: if surface-level success on analogy tasks doesn't demonstrate structure mapping in LLMs, it's not obvious why similar success would demonstrate it in humans or chimpanzees.

There's also a definitional issue lurking. The authors are fairly strict about what counts as "real" symbolic thought — it requires specific neural-level implementations via synchrony-based binding. But this risks making the claim unfalsifiable. If an LLM solves every task the authors throw at it, they can always say "but it's doing so via entangled representations rather than true dynamic binding." That's a reasonable theoretical stance but it means the empirical question becomes: can we find tasks where the underlying mechanism produces different observable behavior? The entanglement/convex-hull prediction is one such test, and to their credit they make it, but a single class of examples is thin evidence for a paper making a claim this broad.

## My overall assessment

The argument is reasonable in a limited form: LLMs likely do not implement symbolic cognition in the specific way biological brains do, and they exhibit characteristic failure modes (entanglement, sample inefficiency, compositional brittleness) that are consistent with lacking genuine dynamic binding and relational structure. This is a defensible position and the paper contributes real insight about what hierarchical integration adds to dynamic binding.

The argument overreaches when it suggests LLMs categorically cannot perform tasks requiring symbolic thought. The evidence is that they perform such tasks unreliably and inefficiently, which is different. Whether that gap is a fundamental architectural ceiling or something that narrows with scale, better training objectives, or hybrid neurosymbolic approaches is an open empirical question the paper doesn't really settle.

So: a reasonable argument about biological cognition with a somewhat overstated extension to LLMs. The core theoretical contribution — isolating multiplace predicates and structure mapping as independent, both-necessary capacities — is valuable regardless of how the LLM debate shakes out.


# Designing a Novel Test

The key constraints, based on their argument:

1. **The surface features must be uninformative** — you can't solve it by pattern-matching to familiar scenarios
2. **The task must require binding familiar elements into statistically unusual relational configurations**
3. **It must not be solvable by analogy to common training examples** (so: no coffee makers, no solar-system/atom analogies, no standard analogical-reasoning benchmarks)
4. **The correct answer must depend on hierarchical relational structure**, not on which elements "feel right" together
5. **It should be novel enough that the specific configuration almost certainly isn't in training data**, while using elements individually common enough that the LLM has representations of them

Here's a task I'd propose. It's designed to be a direct analog of their R&M coffee-maker task but in a domain where the statistical priors actively mislead.

---

## The Task: "Inverted Kitchen"

**Setup you give the LLM:**

> I'm going to describe two scenarios. In the memory scenario, a novel object called a *florp* has a property: it can be activated by pressing its blue button. Your job is to figure out which object in the perception scenario is the florp-analog, and therefore which button on it can be activated.
>
> **Memory scenario:** There are three objects on a table: a florp, a greeble, and a wix. The florp is *underneath* the greeble. The greeble is *to-the-left-of* the wix. The florp has a blue button on top, a red button on the side, and a green button on the bottom. The greeble has a red button on top and a blue button on the bottom. The wix has a green button on top and a blue button on the side. **Pressing the blue button on the florp activates it.**
>
> **Perception scenario:** There are three objects on a shelf: a zop, a quib, and a mek. The mek is *underneath* the quib. The quib is *to-the-left-of* the zop. The zop has a green button on top and a blue button on the side. The quib has a red button on top and a blue button on the bottom. The mek has a blue button on top, a red button on the side, and a green button on the bottom.
>
> Which object in the perception scenario is the florp-analog, and which button activates it?

**Correct answer:** The *mek* is the florp-analog (because it occupies the same relational position — underneath the middle object — and has the same button configuration). Press its *blue button* (on top).

---

## Why this is a good test of their hypothesis

**It requires multiplace predicates.** You cannot solve this by single-place predicates alone. "Underneath" and "to-the-left-of" are genuine two-place relations where both arguments vary. Collapsing them to single-place predicates like `is-below(x)` loses the information about *what* x is below.

**It requires structure mapping.** You must map the relational structure of perception onto memory: mek↔florp, quib↔greeble, zop↔wix. The mapping is determined by relational position, not by object identity (the names are nonsense words) or shared features.

**Surface statistics actively mislead.** The perception scenario lists objects in the order zop, quib, mek — but the florp-analog is the mek (mentioned last). An LLM relying on positional heuristics or first-mentioned-is-most-important patterns will get this wrong. The memory scenario lists florp first; a naive mapping would align florp with zop (first-with-first), which is the wrong answer.

**Feature-matching is insufficient and misleading.** The zop and the florp both have blue buttons prominently. A feature-matching strategy would suggest zop is the florp-analog. Only relational-structure mapping gives the right answer.

**The domain is novel.** "Florp/greeble/wix/zop/quib/mek with colored buttons in spatial relations" is almost certainly not a configuration in any training corpus. The individual elements (colors, buttons, spatial prepositions) are familiar; their specific binding into this problem is not.

## How to interpret the results

If the LLM gets it right with clean reasoning, that's evidence *against* the strong version of Hummel & Heaton's claim — it would suggest transformers can implement something functionally equivalent to structure mapping, at least for small problems.

If the LLM gets it wrong, examine *how*. Their theory predicts specific errors: the model should be drawn to feature-matching (picking zop because of the blue button) or positional matching (picking zop because florp is mentioned first), rather than relational-structure matching.

## Strengthening the test

To make it more rigorous, run several variants:

1. **Baseline:** The version above.
2. **Feature-misleading variant:** Make the feature-matching answer even more tempting (e.g., give the wrong object *two* blue buttons).
3. **Scale test:** Increase to 4 or 5 objects with more relations. Their theory predicts performance should degrade sharply with relational complexity; humans solve these with effort but don't hit a wall.
4. **Cross-domain transfer:** After the object version, give a structurally identical problem with different content (e.g., people in an organizational hierarchy with different skills) and see if performance transfers. True structure mapping should transfer; pattern matching on surface features should not.
5. **Adversarial variant:** Construct a version where the *wrong* answer is statistically likely (e.g., make the feature-matched distractor share more surface tokens with typical "correct answer" patterns from training). This directly tests the entanglement prediction.

One honest caveat: even if an LLM passes all of these, Hummel & Heaton could argue it's still not doing "real" dynamic binding — just a good-enough functional approximation. That's a legitimate theoretical stance, but at that point the empirical question becomes whether the distinction matters for any observable behavior. If the approximation holds across arbitrarily complex compositional tests, the architectural claim starts looking more like an interpretive preference than an empirical constraint.

If you run this, I'd be genuinely curious what you find. The scale-test variant especially — that's where I'd expect to see the clearest signal if their theory is right.