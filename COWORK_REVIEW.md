### Claude Cowork review of initial problem set

> this directory is for a project that is trying to test ideas presented in a paper by hummel, which can be found at hummel_heaton.pdf.  I asked claude code to create a set of tests to check whether LLMs have the relational reasoning capacity that Hummel claims they should not have - this is described in PROBLEMS.md.   I ran the tests on LLMS including Haiku, Sonnet, and Opus, and all of them were able to perform the task perfectly. This has me worried that there might be somethign subtly wrong with the examples that is making it too easy for the models.  can you please examine the problems (in problems/) and identify whether there are any issues that might be allowing the LLM to solve the problem without true relational reasoning?

I've examined the problems carefully and found several serious shortcuts that let LLMs solve the task without any relational reasoning. The short version: **the correct answer is uniquely identifiable by its button configuration alone**. Here are the specific issues, with evidence from the generator code.

## The fatal flaw: exact button-signature matching

In every variant (`baseline`, `feature_misleading`, `adversarial`, `cross_domain`, `scale`), the memory target `m0` and the correct perception analog have **identical** button lists:

From `baseline.py` (lines 47, 80):
```
m0_buttons     = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
correct_analog = [(target_color, "top"), (other1, "side"), (other2, "bottom")]
```

That same pattern appears verbatim in `feature_misleading.py` (lines 46, 71), `adversarial.py` (lines 54, 78), `cross_domain.py` (lines 110, 134), and `scale.py` (lines 51, 71).

An LLM can solve every problem with a trivial non-relational heuristic:

1. Read the memory target's button list.
2. Find the perception object whose button list matches it color-by-color and position-by-position.
3. Report the target color that was "on top" in the memory target.

No spatial relations, no structure mapping, no binding needed. Just a lookup on an exact-match feature vector.

## Additional shortcuts that compound the problem

**Button-count tell (baseline & cross_domain).** The target has 3 buttons; the correct analog has 3; both distractors have only 2. So even a weaker heuristic — "pick the perception object with the same number of buttons" — solves the problem uniquely. Example: in `baseline_00.json`, only `ikk` has three buttons among `{snig, ikk, mek}`. Same in `cross_domain_00.json` (only `trob`).

**The activation color is trivially recoverable.** The memory says "Pressing the `target_color` button on the `m0` activates it," and the correct analog always has `target_color` on top. So once the object is picked (by any method, including the wrong one), guessing the color is almost free. Worse: naïvely echoing the same color named in the memory scenario gets the color right almost every time — in the baseline, feature_misleading, cross_domain, adversarial, and scale variants, `correct_answer.button_color == target_color` by construction. A model that just answers "the one that activates it was X in memory, so it's X here" scores 100% on the color part without thinking about perception at all.

**"Feature-misleading" doesn't remove the feature shortcut.** The supposed hard variant only makes the distractor's colors somewhat more `target_color`-heavy. The correct analog still has the exact same full signature as `m0`, while the distractor has a different signature (`feature_misleading.py` line 72: `[(target_color, "top"), (target_color, "side"), (other1, "bottom")]`). Exact-signature matching still wins, so this variant doesn't stress the feature-matching strategy at all. The "adversarial" variant inherits the same issue and just adds a linguistic decoy sentence.

**Scale variant is the same story.** `scale.py` lines 71–72 give the correct analog `[(target_color, "top"), (other1, "side"), (other2, "bottom")]` and the distractor `[(other1, "top"), (target_color, "side"), (other2, "bottom")]` — a permutation, but still uniquely distinguishable by exact signature. All non-distractor filler objects in perception get only 2 buttons (line 77), preserving the button-count tell.

**The correct analog is always `perception_words[0]` in relation order.** This isn't a shortcut the model can exploit from the text (the list order is shuffled), but it is a diversity issue: the relational role of the answer never varies (always the "underneath" object, never the middle or right one). If a model happens to learn "answer the one in the bottom-left relational slot," it's right every time. Hummel & Heaton's prediction would be better tested if the target's relational role varied across problems.

## Why this matches the symptom you saw

Hummel & Heaton's hypothesis predicts failure precisely when *features are uninformative* and only relational structure gives the answer. Your problems accidentally make the full feature signature **maximally** informative: the correct answer is literally the object whose buttons match the target's buttons exactly. So Haiku/Sonnet/Opus scoring perfectly is consistent with them doing nothing but lexical/feature matching — it does not tell you whether they can do relational reasoning.

## Concrete fixes to make the test actually diagnostic

The tests need to **break the correlation between feature signature and relational role**. A few suggestions:

1. **Give the correct analog a different button set than `m0`.** The relational mapping should say "the object in the same relational position," but its buttons should not be a clone of `m0`'s. For instance, give `m0` buttons `[blue/top, red/side, green/bottom]` and the correct analog buttons `[red/top, green/side, blue/bottom]`. Then ask which button *in the perception object's own color vocabulary* corresponds to the one activated in memory — requiring a genuine mapping from relational position, not color identity.

2. **Make the feature-match distractor a true twin of `m0`.** If the distractor has `m0`'s exact button signature while the correct answer (identified purely by relational position) has a different signature, feature-matching now gives the *wrong* answer. That's the variant the PROBLEM.md says the feature_misleading condition was supposed to implement.

3. **Equalize button counts.** Every perception object should have the same number of buttons, so count alone carries no signal.

4. **Vary which relational role is the target.** Sometimes the answer should be the "underneath" object, sometimes the "middle," sometimes the "rightmost," so no positional default works.

5. **Decouple the activating color from anything nameable in memory.** E.g., ask "which button activates the analog?" where the answer depends on structural role (say, "the button on top of the object whose relational role matches `m0`'s") rather than a color name that was literally stated in the memory scenario.

6. **Stress-test with a no-relational-structure control.** If you strip out all the spatial-relation sentences from the perception scenario, a truly relational solver should fail. If LLMs still answer correctly, you know they were using feature shortcuts all along. This is a cheap but powerful diagnostic to add.

If you'd like, I can patch the generators to implement fix #1 and fix #2, regenerate the problems, and rerun — that would give you a much cleaner signal about whether the models are actually doing structure mapping or just feature-matching.