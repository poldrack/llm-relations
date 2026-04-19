# Prompt Variants — Design

**Goal:** Let benchmark runs select among several named system-prompt variants (e.g. `cot`, `no_cot`, `graphical_model`, …) so we can compare how different prompt instructions affect relational-reasoning accuracy. Replace the boolean `use_cot` / `--no-cot` toggle with a single `--prompt-variant NAME` flag backed by a code-level registry.

## Motivation

Today the system prompt has exactly two forms: with or without a "think step by step" instruction. We want to try other instruction styles — starting with an explicit "build a graphical model of the relational structure" variant — and track results per-variant in the same on-disk and CSV layout already used for `cot` / `no_cot`.

A registry in code (rather than a free-form CLI string) keeps every instruction text checked into git, guarantees reproducibility, and makes each new experiment a one-line addition.

## Architecture

### Prompt registry (in `src/llm_relations/runner/client.py`)

```python
PROMPT_VARIANTS: dict[str, str] = {
    "cot": _COT_INSTRUCTION,
    "no_cot": "",
    "graphical_model": _GRAPHICAL_MODEL_INSTRUCTION,
}
```

Each value is the *middle* instruction block that gets sandwiched between the fixed `_TASK_DESCRIPTION` and `_ANSWER_FORMAT` sections. Empty string means "no extra instruction" (that's what `no_cot` is).

### `build_system_prompt` signature

```python
def build_system_prompt(prompt_variant: str = "cot") -> str:
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(
            f"Unknown prompt_variant {prompt_variant!r}. "
            f"Valid variants: {sorted(PROMPT_VARIANTS)}"
        )
    instruction = PROMPT_VARIANTS[prompt_variant]
    parts = [_TASK_DESCRIPTION]
    if instruction:
        parts.append(instruction)
    parts.append(_ANSWER_FORMAT)
    return "\n\n".join(parts)
```

The existing module-level `SYSTEM_PROMPT = build_system_prompt(use_cot=True)` becomes `SYSTEM_PROMPT = build_system_prompt("cot")` (keeping the same text, so the client's default behavior is unchanged when callers don't pass a system prompt).

### New `_GRAPHICAL_MODEL_INSTRUCTION`

```
Before answering, build an explicit graphical model of each scenario.
Represent objects as nodes and the relations between them (e.g.
*underneath*, *to-the-left-of*) as labeled directed edges. Write out
both graphs. Then find the mapping between the perception graph and
the memory graph that preserves edge labels and direction, and use
that mapping to answer.
```

### `run_benchmark` signature

```python
def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    model_specs: list[ModelSpec],
    n_samples: int,
    prompt_variant: str = "cot",
    variants: Optional[list[str]] = None,
) -> None:
    ...
    system_prompt = build_system_prompt(prompt_variant)
    # prompt_variant flows straight through — no "cot"/"no_cot" derivation.
```

Validation of `prompt_variant` happens inside `build_system_prompt`, so an unknown name raises `ValueError` *before* any client call is made.

### CLI (`scripts/run_benchmark.py`)

Replace the `--no-cot` flag with:

```python
parser.add_argument(
    "--prompt-variant",
    default="cot",
    help=(
        "System-prompt variant. Registered variants: "
        "cot, no_cot, graphical_model. Default: cot."
    ),
)
```

And call `run_benchmark(..., prompt_variant=args.prompt_variant, ...)`.

## What doesn't change

- On-disk layout: `results/raw/{prompt_variant}/{safe_model}/{problem_id}/sample_N.json`. Existing `raw/cot/` and `raw/no_cot/` trees remain valid and continue to be read back by `_load_samples_on_disk` and `rebuild_summary_from_disk`.
- `SampleRecord.prompt_variant` field and the `prompt_variant` column in `summary.csv`.
- `_TASK_DESCRIPTION` and `_ANSWER_FORMAT` text — only the middle instruction block varies.
- `build_model_specs`, LMStudio integration, `--variants` filter, sample appending logic.

## What breaks (no back-compat)

- `run_benchmark(use_cot=...)` — callers must migrate to `prompt_variant=`.
- `scripts/run_benchmark.py --no-cot` — users must migrate to `--prompt-variant no_cot`.
- `build_system_prompt(use_cot=...)` — callers must migrate to positional/named `prompt_variant`.

Acceptable because there are no external consumers of this module and the CLI is only used locally.

## Testing (TDD order)

### `tests/test_runner_client.py` (new/updated tests)

1. **`test_build_system_prompt_graphical_model_includes_graph_instruction`** — `build_system_prompt("graphical_model")` contains the phrase "graphical model" (or "graph") and does *not* contain the "think step by step" wording.
2. **`test_build_system_prompt_no_cot_excludes_both_instructions`** — `build_system_prompt("no_cot")` contains neither the CoT phrase nor the graphical-model phrase, but still contains the answer-format JSON block.
3. **`test_build_system_prompt_unknown_variant_raises`** — `build_system_prompt("bogus")` raises `ValueError` whose message lists the registered variant names.
4. **Update existing tests** — `build_system_prompt(use_cot=True)` → `build_system_prompt("cot")`, etc. `SYSTEM_PROMPT == build_system_prompt("cot")`.

### `tests/test_runner_benchmark.py` (new/updated tests)

5. **`test_run_benchmark_writes_samples_under_graphical_model_variant`** — run with `prompt_variant="graphical_model"`; assert samples appear under `raw/graphical_model/...` and `SampleRecord.prompt_variant == "graphical_model"` in the JSON and the summary CSV.
6. **`test_run_benchmark_rejects_unknown_prompt_variant_before_calling_client`** — run with `prompt_variant="bogus"`; assert `ValueError` and assert the mock client's `call` was never invoked.
7. **Update existing tests** — every `use_cot=True/False` becomes `prompt_variant="cot"/"no_cot"`.

## File changes

- **Modify:** `src/llm_relations/runner/client.py` — add `_GRAPHICAL_MODEL_INSTRUCTION`, `PROMPT_VARIANTS` registry, rewrite `build_system_prompt` signature, update `SYSTEM_PROMPT`.
- **Modify:** `src/llm_relations/runner/benchmark.py` — `use_cot: bool` → `prompt_variant: str` in `run_benchmark`; pass through.
- **Modify:** `scripts/run_benchmark.py` — replace `--no-cot` with `--prompt-variant`.
- **Modify:** `tests/test_runner_client.py` — add new tests, migrate existing `use_cot` calls.
- **Modify:** `tests/test_runner_benchmark.py` — add new tests, migrate existing `use_cot` calls.

## Out of scope

- Composing/stacking variants (e.g. "graphical_model + cot"). If wanted later, add a combined variant as a separate registry entry.
- Per-problem or per-model prompt variants.
- User-supplied ad-hoc instructions via a free-form CLI flag.
- Modifying the *user* prompt (the problem text itself). Only the system prompt is under experimental control here.
