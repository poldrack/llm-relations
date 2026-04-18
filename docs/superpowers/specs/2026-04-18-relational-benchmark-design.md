# Relational Reasoning Benchmark — Design

## Goal

Build a benchmark that tests whether Claude models (Opus 4.7, Sonnet 4.6, Haiku 4.5) succeed at relational structure mapping as described in Hummel & Heaton (2025), then quantify per-model, per-variant performance and error modes.

The benchmark implements the "Inverted Kitchen" task from `PROBLEM.md` and its four strengthening variants. Each run produces accuracy numbers and — more importantly — error-type classifications that test Hummel & Heaton's specific prediction that failure modes should be feature-match or positional-match errors rather than random.

## Scope

- 5 variants × 5 instances = 25 committed problem instances
- 3 models × 25 problems × 5 samples = 375 API calls per full run
- Structural variation within each variant (not just surface reskinning)
- Outputs: raw response logs, summary CSV, Jupyter notebook report

Out of scope: multi-turn prompts, tool use, fine-tuning, model-as-judge scoring.

## Architecture

Four-layer Python package:

```
llm-relations/
├── generator/              # one module per variant; emits Problem objects
│   ├── baseline.py
│   ├── feature_misleading.py
│   ├── scale.py
│   ├── cross_domain.py
│   └── adversarial.py
├── problems/               # frozen corpus (committed JSON)
│   └── {variant}_{nn}.json
├── runner/                 # API client, sampling, result logging
├── analysis/               # scorer, summary CSV, report.ipynb
├── tests/
└── results/                # raw responses + summary (gitignored except summary)
```

**Generators → corpus → runner → analysis.** Generators are seeded; their output is hand-reviewed once, frozen to `problems/*.json`, and that is what the runner reads. Regenerating with the same seed reproduces the instances, but the committed JSON is ground truth.

## Problem schema

A `Problem` is a dataclass serialized to JSON:

```json
{
  "problem_id": "baseline_02",
  "variant": "baseline",
  "prompt_text": "I'm going to describe two scenarios...",
  "correct_answer": {"analog": "mek", "button_color": "blue"},
  "metadata": {
    "n_objects": 3,
    "feature_match_answer": {"analog": "zop", "button_color": "blue"},
    "positional_match_answer": {"analog": "zop", "button_color": "green"},
    "correct_slot_index": 2,
    "seed": 4201
  }
}
```

`feature_match_answer` and `positional_match_answer` are the predicted wrong answers per Hummel & Heaton's theory. The scorer uses them to classify error types.

## Variants

### 1. Baseline (n=3 objects, 2 relations)
Direct implementation of the Inverted Kitchen task. Structural variation: each of 5 instances places the correct analog at a different relational slot, so naive positional heuristics (first-mentioned, last-mentioned) can succeed on at most one instance.

### 2. Feature-misleading (n=3)
Same structure as baseline, but the feature-match distractor is strengthened — the wrong object gets two matching colored buttons, not one. Variation: which object is the distractor and how the misleading features are arranged.

### 3. Scale test
One instance each at n=4, 5, 6, 7, 8 objects, with 2, 3, 3, 4, 4 relations respectively. Variation is size. Predicts a performance curve; per Hummel & Heaton, performance should degrade sharply with relational complexity.

### 4. Cross-domain transfer
Same n=3 baseline structure, 5 non-object domains: (a) people in an org chart with skills, (b) plants in a garden with colored leaves, (c) rooms in a building with fixtures, (d) animals in an enclosure with markings, (e) vehicles in a lot with labeled panels. Each domain preserves the two-place relations and button-equivalent feature.

### 5. Adversarial
Feature-match distractor plus phrasing that mimics training-data patterns for "activated by X" (e.g., "the distractor's blue button lights up when pressed"). Variation: specific linguistic decoy. Directly tests the entanglement prediction.

**Shared palette.** All variants draw features from a fixed pool (colors: blue/red/green/yellow/purple; positions: top/bottom/side/front/back) and nonsense words from a fixed list so surface content stays comparable across variants.

## Prompt and response

**System prompt** (cached, identical across all problems):

> You are solving relational reasoning problems. Each problem has a memory scenario and a perception scenario. Your task is to map objects in the perception scenario to objects in the memory scenario based on their relational structure (how they relate to each other), then answer a specific question.
>
> Think step by step: first identify the relations in each scenario, then find the mapping that preserves relational structure, then answer.
>
> End your response with a JSON block in this exact format:
> ```json
> {"analog": "<object_name>", "button_color": "<color>"}
> ```

**User prompt** = `problem.prompt_text`.

**Parsing.** Extract the last fenced JSON block. On parse failure, mark `parse_error=true, is_correct=false` and log the raw response.

**API parameters.**
- Model IDs: `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
- `max_tokens`: 4096
- `temperature`: 1.0
- Prompt caching on system prompt
- Retry-with-backoff on 429/529

## Results storage

Raw per-sample JSON at `results/raw/{model}/{problem_id}/sample_{n}.json`:

```json
{
  "problem_id": "baseline_02",
  "model": "claude-opus-4-7",
  "sample": 3,
  "variant": "baseline",
  "prompt": "...",
  "response_text": "...full model output...",
  "parsed_answer": {"analog": "mek", "button_color": "blue"},
  "correct_answer": {"analog": "mek", "button_color": "blue"},
  "is_correct": true,
  "error_type": null,
  "parse_error": false,
  "input_tokens": 812,
  "output_tokens": 1047,
  "latency_ms": 3420,
  "timestamp": "2026-04-18T14:22:01Z"
}
```

`error_type` is `null` when correct, otherwise one of `feature_match | positional_match | other | parse_error`.

`results/summary.csv` — one row per (model, variant, problem_id): mean accuracy over n=5, n_correct, error-type counts, mean tokens, mean latency.

## Analysis notebook

`analysis/report.ipynb` loads `summary.csv` and `results/raw/` and renders:

1. Per-variant × per-model accuracy table (mean ± SE across 5 instances × 5 samples)
2. Scale-test curve: matplotlib line plot, accuracy vs. n_objects, one line per model with error bars
3. Error-type stacked bar chart per model (feature-match / positional / other / parse error)
4. Per-problem drill-down DataFrame, with a helper to render the full prompt + response for any chosen (model, problem_id, sample)

Committed with outputs cleared; regenerated after each run.

## Testing

pytest suite:
- Each generator emits a well-formed `Problem` with consistent metadata
- Frozen corpus is valid (JSON loads, schema matches, ground truth is internally consistent — e.g., the correct analog actually occupies the claimed relational position)
- Scorer correctly classifies error types on hand-constructed response examples
- JSON parser handles malformed responses (missing block, extra fields, wrong types) gracefully

Runner/API is mocked in unit tests. A separate opt-in smoke test hits the real API with a single Haiku call.

TDD: tests written before implementation for each component.

## Cost estimate

At n=5 samples, full run is 375 calls. Per prior estimate: ~$16–25 total, dominated by Opus. No real budget concern.

## Non-goals

- No model-as-judge scoring. Correctness is exact-match against ground truth; error classification is rule-based from metadata.
- No adaptive sampling or early stopping — fixed 5 samples per (model, problem) for simplicity.
- No comparison against non-Claude models in this iteration.
