# Prompt Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the boolean `use_cot` / `--no-cot` toggle with a named `prompt_variant` parameter backed by a registry, and ship an initial `graphical_model` variant that asks the model to build a relational graph before answering.

**Architecture:** Add `PROMPT_VARIANTS: dict[str, str]` in `src/llm_relations/runner/client.py`. `build_system_prompt` takes `prompt_variant: str`, looks up the instruction block, composes `_TASK_DESCRIPTION + instruction + _ANSWER_FORMAT`, and raises `ValueError` on unknown names. `run_benchmark`'s `use_cot: bool` becomes `prompt_variant: str`; the CLI's `--no-cot` becomes `--prompt-variant NAME`. The value flows straight through to `SampleRecord.prompt_variant`, the `results/raw/{prompt_variant}/...` path, and the `prompt_variant` column in `summary.csv` — no new on-disk layout.

**Tech Stack:** Python 3.11+, pytest, uv, anthropic SDK (existing).

---

## File Structure

- **Modify:** `src/llm_relations/runner/client.py` — add `_GRAPHICAL_MODEL_INSTRUCTION` constant, `PROMPT_VARIANTS` dict, rewrite `build_system_prompt` signature, update module-level `SYSTEM_PROMPT`.
- **Modify:** `src/llm_relations/runner/benchmark.py` — change `run_benchmark`'s `use_cot: bool = True` parameter to `prompt_variant: str = "cot"`; pass it through unchanged.
- **Modify:** `scripts/run_benchmark.py` — replace `--no-cot` flag with `--prompt-variant`.
- **Modify:** `tests/test_runner_client.py` — add 3 new tests, migrate 3 existing tests.
- **Modify:** `tests/test_runner_benchmark.py` — add 2 new tests, migrate ~10 existing tests.

No new files. No files deleted.

---

## Task 1: Tests — RED

Write all new tests and migrate existing `use_cot=` call sites to the new `prompt_variant=` / string-argument API. Commit the red tests first (per TDD discipline in this repo's CLAUDE.md). The test suite will be red until Task 2 lands the implementation.

**Files:**
- Modify: `tests/test_runner_client.py`
- Modify: `tests/test_runner_benchmark.py`

- [ ] **Step 1: Replace the three `build_system_prompt` tests in `tests/test_runner_client.py`**

Find the three existing tests (`test_build_system_prompt_with_cot_includes_think_step_by_step`, `test_build_system_prompt_without_cot_omits_think_step_by_step`, `test_default_system_prompt_matches_cot_variant`) near lines 46–62 and replace them with the following. This covers the migrated cases PLUS the three new tests from the spec.

```python
def test_build_system_prompt_cot_includes_think_step_by_step():
    prompt = build_system_prompt("cot")
    assert "Think step by step" in prompt
    # Still includes the JSON format instruction.
    assert "```json" in prompt


def test_build_system_prompt_no_cot_excludes_both_instructions():
    prompt = build_system_prompt("no_cot")
    assert "Think step by step" not in prompt
    assert "step by step" not in prompt.lower()
    assert "graphical model" not in prompt.lower()
    assert "graph" not in prompt.lower()
    # Still includes the JSON format instruction.
    assert "```json" in prompt


def test_build_system_prompt_graphical_model_includes_graph_instruction():
    prompt = build_system_prompt("graphical_model")
    # Names the technique explicitly.
    assert "graphical model" in prompt.lower() or "graph" in prompt.lower()
    # Does NOT include the CoT "Think step by step" instruction —
    # graphical_model is an alternative, not an addition.
    assert "Think step by step" not in prompt
    # Still includes the JSON format instruction.
    assert "```json" in prompt


def test_build_system_prompt_unknown_variant_raises():
    import pytest
    with pytest.raises(ValueError) as excinfo:
        build_system_prompt("bogus")
    msg = str(excinfo.value)
    # Message names the bad variant and lists the valid ones.
    assert "bogus" in msg
    assert "cot" in msg
    assert "no_cot" in msg
    assert "graphical_model" in msg


def test_default_system_prompt_matches_cot_variant():
    assert SYSTEM_PROMPT == build_system_prompt("cot")
```

- [ ] **Step 2: Migrate `use_cot=` call sites in `tests/test_runner_benchmark.py`**

Search for every `use_cot=` and replace per the mapping below. Keep each test's other assertions identical. Exact line numbers (verified in current file):

| Line  | Current | Replacement |
|-------|---------|-------------|
| 114   | `use_cot=False,` | `prompt_variant="no_cot",` |
| 140   | `use_cot=False,` | `prompt_variant="no_cot",` |
| 163   | `model_specs=[spec], n_samples=1, use_cot=True,` | `model_specs=[spec], n_samples=1, prompt_variant="cot",` |
| 167   | `model_specs=[spec], n_samples=1, use_cot=False,` | `model_specs=[spec], n_samples=1, prompt_variant="no_cot",` |
| 193   | `use_cot=False,` | `prompt_variant="no_cot",` |
| 362   | `model_specs=[spec], n_samples=2, use_cot=True,` | `model_specs=[spec], n_samples=2, prompt_variant="cot",` |
| 367   | `model_specs=[spec], n_samples=2, use_cot=True,` | `model_specs=[spec], n_samples=2, prompt_variant="cot",` |
| 401   | `model_specs=[spec], n_samples=2, use_cot=True,` | `model_specs=[spec], n_samples=2, prompt_variant="cot",` |
| 405   | `model_specs=[spec], n_samples=2, use_cot=True,` | `model_specs=[spec], n_samples=2, prompt_variant="cot",` |
| 632   | `n_samples=1, use_cot=True,` | `n_samples=1, prompt_variant="cot",` |

Use an editor find-and-replace for safety: replace `use_cot=True` → `prompt_variant="cot"` and `use_cot=False` → `prompt_variant="no_cot"` across the whole file. These are the only two forms that appear.

After the edits, verify with ripgrep:

```bash
uv run rg "use_cot" tests/
```

Expected: no matches in `tests/test_runner_benchmark.py` or `tests/test_runner_client.py`.

- [ ] **Step 3: Also migrate the test-name strings in `tests/test_runner_benchmark.py` that mention `use_cot`**

The functions `test_run_benchmark_passes_no_cot_system_prompt_to_client` and `test_run_benchmark_defaults_to_cot_system_prompt` (lines ~177 and ~201) describe `no_cot`/`cot` and remain valid — keep their names as-is. Only the call-site args change.

- [ ] **Step 4: Add two new benchmark tests in `tests/test_runner_benchmark.py`**

Append these two tests at the end of the file. They assert behavior that requires the new `prompt_variant` parameter and the validation path.

```python
def test_run_benchmark_writes_samples_under_graphical_model_variant(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir,
        results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1,
        prompt_variant="graphical_model",
    )

    # Directory tree uses the variant name verbatim.
    sample_path = (
        results_dir / "raw" / "graphical_model" / "claude-haiku-4-5-20251001"
        / p.problem_id / "sample_0.json"
    )
    assert sample_path.exists()
    rec = json.loads(sample_path.read_text())
    assert rec["prompt_variant"] == "graphical_model"

    # The graphical-model instruction was sent in the system prompt.
    sys_prompt = client.call.call_args.kwargs["system_prompt"]
    assert "graph" in sys_prompt.lower()

    # Summary CSV carries the variant too.
    summary = (results_dir / "summary.csv").read_text()
    assert "graphical_model" in summary


def test_run_benchmark_rejects_unknown_prompt_variant_before_calling_client(
    tmp_path: Path,
):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning("unused")

    import pytest
    with pytest.raises(ValueError, match="bogus"):
        run_benchmark(
            problems_dir=problems_dir,
            results_dir=results_dir,
            model_specs=[_spec("claude-haiku-4-5-20251001", client)],
            n_samples=1,
            prompt_variant="bogus",
        )

    # No API calls were made — validation happened first.
    assert client.call.call_count == 0
```

- [ ] **Step 5: Run the test suite and confirm the failures are the expected ones**

Run:

```bash
uv run pytest tests/test_runner_client.py tests/test_runner_benchmark.py -v
```

Expected failures (at minimum — exact error messages will vary):

- `test_build_system_prompt_cot_includes_think_step_by_step` — `TypeError: build_system_prompt() got an unexpected keyword argument` or `TypeError: build_system_prompt() takes 0 or 1 positional arguments` depending on how it resolves the string; the old signature is `build_system_prompt(use_cot: bool = True)`, and `build_system_prompt("cot")` passes the string as the bool — `"cot"` is truthy so this specific call *accidentally* returns the cot prompt and passes. That's fine. The three genuinely new tests should fail:
  - `test_build_system_prompt_no_cot_excludes_both_instructions` — fails because `build_system_prompt("no_cot")` treats `"no_cot"` as truthy → returns the cot prompt → `"Think step by step"` is present.
  - `test_build_system_prompt_graphical_model_includes_graph_instruction` — same reason, plus no graph-related text in the returned prompt.
  - `test_build_system_prompt_unknown_variant_raises` — old signature does not validate, so `ValueError` is never raised.
- All `prompt_variant=`-passing `run_benchmark` calls — fail with `TypeError: run_benchmark() got an unexpected keyword argument 'prompt_variant'`.
- `test_run_benchmark_writes_samples_under_graphical_model_variant` — same `TypeError`.
- `test_run_benchmark_rejects_unknown_prompt_variant_before_calling_client` — same `TypeError`, though it's wrapped in `pytest.raises(ValueError)` so the test fails because `TypeError` is raised instead of `ValueError`.

This is the intended RED state. Do not fix anything yet.

- [ ] **Step 6: Commit the red tests**

```bash
git add tests/test_runner_client.py tests/test_runner_benchmark.py
git commit -m "$(cat <<'EOF'
test: RED - prompt-variant registry (graphical_model, unknown-variant validation)

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Implementation — GREEN

Land `PROMPT_VARIANTS` in `client.py` and rename the `run_benchmark` parameter. After this task the test suite is green.

**Files:**
- Modify: `src/llm_relations/runner/client.py`
- Modify: `src/llm_relations/runner/benchmark.py`

- [ ] **Step 1: Replace the top of `src/llm_relations/runner/client.py` up through `SYSTEM_PROMPT`**

Find the block from line 9 to line 37 (the three `_TASK_DESCRIPTION`/`_COT_INSTRUCTION`/`_ANSWER_FORMAT` constants, the `build_system_prompt` function, and the `SYSTEM_PROMPT` assignment). Replace it with:

```python
_TASK_DESCRIPTION = (
    "You are solving relational reasoning problems. Each problem has a memory scenario "
    "and a perception scenario. Your task is to map objects in the perception scenario "
    "to objects in the memory scenario based on their relational structure (how they "
    "relate to each other), then answer a specific question."
)

_COT_INSTRUCTION = (
    "Think step by step: first identify the relations in each scenario, then find the "
    "mapping that preserves relational structure, then answer."
)

_GRAPHICAL_MODEL_INSTRUCTION = (
    "Before answering, build an explicit graphical model of each scenario. Represent "
    "objects as nodes and the relations between them (e.g. 'underneath', "
    "'to-the-left-of') as labeled directed edges. Write out both graphs. Then find "
    "the mapping between the perception graph and the memory graph that preserves "
    "edge labels and direction, and use that mapping to answer."
)

_ANSWER_FORMAT = (
    "End your response with a JSON block in this exact format:\n"
    "```json\n"
    '{"analog": "<object_name>", "button_color": "<color>"}\n'
    "```"
)


PROMPT_VARIANTS: dict[str, str] = {
    "cot": _COT_INSTRUCTION,
    "no_cot": "",
    "graphical_model": _GRAPHICAL_MODEL_INSTRUCTION,
}


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


SYSTEM_PROMPT = build_system_prompt("cot")
```

Leave the rest of `client.py` (the `CallResult` dataclass, the `ClaudeClient` class, the retry loop) untouched — the `call()` method already accepts `system_prompt: str = SYSTEM_PROMPT`, which works unchanged.

- [ ] **Step 2: Update `run_benchmark` in `src/llm_relations/runner/benchmark.py`**

Change the `run_benchmark` signature and the two lines inside that derive / use `prompt_variant`.

Current (lines 304–328):

```python
def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    model_specs: list[ModelSpec],
    n_samples: int,
    use_cot: bool = True,
    variants: Optional[list[str]] = None,
) -> None:
    problems = [load_problem(f) for f in sorted(problems_dir.glob("*.json"))]
    if variants is not None:
        ...
    system_prompt = build_system_prompt(use_cot=use_cot)
    prompt_variant = "cot" if use_cot else "no_cot"
```

Replace with:

```python
def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    model_specs: list[ModelSpec],
    n_samples: int,
    prompt_variant: str = "cot",
    variants: Optional[list[str]] = None,
) -> None:
    problems = [load_problem(f) for f in sorted(problems_dir.glob("*.json"))]
    if variants is not None:
        ...
    system_prompt = build_system_prompt(prompt_variant)
```

(Keep the `if variants is not None:` block identical — it validates `--variants`, not `--prompt-variant`.)

The `prompt_variant` local binding now comes from the parameter itself rather than being derived from `use_cot`. The rest of the function (the inner loop, the `_next_sample_index` call, `_run_one_sample` call, `_write_summary` call) is unchanged.

- [ ] **Step 3: Run the full test suite**

```bash
uv run pytest tests/test_runner_client.py tests/test_runner_benchmark.py -v
```

Expected: all tests pass, including the five new ones added in Task 1. If a test still fails, the most likely cause is a stray `use_cot=` call site that Task 1 Step 2 missed — re-run `uv run rg "use_cot" tests/` and fix.

- [ ] **Step 4: Run the whole repo test suite to catch other regressions**

```bash
uv run pytest -v
```

Expected: all tests pass. This catches any other caller of `build_system_prompt(use_cot=...)` or `run_benchmark(use_cot=...)` that the plan missed.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/client.py src/llm_relations/runner/benchmark.py
git commit -m "$(cat <<'EOF'
feat: replace use_cot bool with prompt_variant registry; add graphical_model

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: CLI — expose `--prompt-variant`

Swap the `--no-cot` flag for `--prompt-variant` in `scripts/run_benchmark.py`.

**Files:**
- Modify: `scripts/run_benchmark.py`

- [ ] **Step 1: Replace the `--no-cot` argument definition**

In `scripts/run_benchmark.py`, find this block (lines 48–52):

```python
    parser.add_argument(
        "--no-cot",
        action="store_true",
        help="Omit the 'think step by step' instruction from the system prompt.",
    )
```

Replace with:

```python
    parser.add_argument(
        "--prompt-variant",
        default="cot",
        help=(
            "System-prompt variant. One of: cot (default), no_cot, "
            "graphical_model. See src/llm_relations/runner/client.py "
            "for the registry."
        ),
    )
```

- [ ] **Step 2: Update the `run_benchmark(...)` call in the same file**

Find this block (lines 66–73):

```python
    run_benchmark(
        problems_dir=args.problems_dir,
        results_dir=args.results_dir,
        model_specs=specs,
        n_samples=args.n_samples,
        use_cot=not args.no_cot,
        variants=args.variants,
    )
```

Replace with:

```python
    run_benchmark(
        problems_dir=args.problems_dir,
        results_dir=args.results_dir,
        model_specs=specs,
        n_samples=args.n_samples,
        prompt_variant=args.prompt_variant,
        variants=args.variants,
    )
```

- [ ] **Step 3: Verify the CLI help reflects the new flag**

```bash
uv run python scripts/run_benchmark.py --help
```

Expected: help output lists `--prompt-variant PROMPT_VARIANT` with the description from Step 1, and does NOT list `--no-cot`.

- [ ] **Step 4: Verify unknown-variant validation surfaces as a CLI error**

```bash
uv run python scripts/run_benchmark.py --prompt-variant bogus --problems-dir problems --results-dir /tmp/llmr_bogus_$$ --n-samples 1 --models claude-haiku-4-5-20251001
```

Expected: the script exits with a `ValueError` traceback naming `'bogus'` and listing the valid variants — the `ValueError` is raised by `build_system_prompt` inside `run_benchmark` before any API call. No sample files are written to `/tmp/llmr_bogus_$$`. (An `ANTHROPIC_API_KEY` does not need to be set because validation fails before the client is invoked.)

Clean up afterward:

```bash
rm -rf /tmp/llmr_bogus_$$
```

- [ ] **Step 5: Run the whole repo test suite one more time**

```bash
uv run pytest -v
```

Expected: all tests pass. (The CLI script is not directly covered by the test suite, but this catches any incidental breakage from the earlier tasks.)

- [ ] **Step 6: Commit**

```bash
git add scripts/run_benchmark.py
git commit -m "$(cat <<'EOF'
feat(cli): replace --no-cot with --prompt-variant NAME

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Definition of Done

- `uv run pytest` passes with five new tests added (three in `test_runner_client.py`, two in `test_runner_benchmark.py`) and all existing tests migrated.
- `uv run rg "use_cot" src/ tests/ scripts/` returns no matches — the parameter name is gone everywhere.
- `uv run python scripts/run_benchmark.py --help` lists `--prompt-variant` and does not list `--no-cot`.
- A dry run with `--prompt-variant graphical_model` (when the user chooses to execute one) creates `results/raw/graphical_model/...` and records `graphical_model` in both the JSON samples and `summary.csv`.
- Passing `--prompt-variant bogus` raises a `ValueError` naming the bad variant and listing the valid ones, and does not perform any API calls.
