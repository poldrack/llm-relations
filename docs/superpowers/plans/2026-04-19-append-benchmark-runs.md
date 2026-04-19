# Append Benchmark Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `run_benchmark` append new samples to any existing on-disk samples (per prompt_variant × model × problem_id) instead of overwriting from index 0, and make `summary.csv` reflect the combined count.

**Architecture:** Add a `_next_sample_index` helper that scans the target directory for `sample_*.json` files and returns `max(N)+1`. The `run_benchmark` loop queries this helper per (spec, problem) and numbers new samples from there. `_write_summary` re-aggregates from disk for any key touched by the current run, so the CSV includes both prior and new samples.

**Tech Stack:** Python 3, `uv`, pytest. Existing module: `src/llm_relations/runner/benchmark.py`. Existing tests: `tests/test_runner_benchmark.py`.

**Spec:** `docs/superpowers/specs/2026-04-19-append-benchmark-runs-design.md`

---

## File Structure

- **Modify** `src/llm_relations/runner/benchmark.py` — add `_next_sample_index` helper; change the sample loop in `run_benchmark`; change `_write_summary` to re-aggregate from disk for touched keys.
- **Modify** `tests/test_runner_benchmark.py` — add 3 new tests (append, summary re-aggregation, gap handling).
- No new files. No CLI changes.

---

## Task 1: RED — test that re-running appends samples

**Files:**
- Modify: `tests/test_runner_benchmark.py` (append to end of file)

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_runner_benchmark.py`:

```python
def test_run_benchmark_appends_samples_when_prior_exist(tmp_path: Path):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )
    spec = _spec("claude-haiku-4-5-20251001", client)

    # First run: writes sample_0.json and sample_1.json
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )
    # Second run: should write sample_2.json and sample_3.json (not overwrite)
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )

    target_dir = (
        results_dir / "raw" / "cot" / "claude-haiku-4-5-20251001" / p.problem_id
    )
    files = sorted(f.name for f in target_dir.glob("sample_*.json"))
    assert files == [
        "sample_0.json", "sample_1.json", "sample_2.json", "sample_3.json"
    ], files

    # Each record's `sample` field matches its filename index.
    for name in files:
        idx = int(name.removeprefix("sample_").removesuffix(".json"))
        rec = json.loads((target_dir / name).read_text())
        assert rec["sample"] == idx
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_benchmark.py::test_run_benchmark_appends_samples_when_prior_exist -v`

Expected: FAIL. The second run currently overwrites `sample_0.json` and `sample_1.json`, so the directory ends up with only those two files (not four), and the assertion on the sorted filename list fails.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_runner_benchmark.py
git commit -m "test: RED - re-running run_benchmark should append samples"
```

---

## Task 2: GREEN — append samples via `_next_sample_index`

**Files:**
- Modify: `src/llm_relations/runner/benchmark.py`

- [ ] **Step 1: Add the `_next_sample_index` helper**

Add this helper to `src/llm_relations/runner/benchmark.py`, placed directly above `_write_sample` (so related file-layout logic is grouped):

```python
def _next_sample_index(
    results_dir: Path,
    prompt_variant: str,
    display_name: str,
    problem_id: str,
) -> int:
    """Return the next unused sample index for a given (variant, model, problem).

    Scans `results/raw/{prompt_variant}/{safe_model}/{problem_id}/` for files
    named `sample_N.json` and returns `max(N) + 1`. Returns 0 if the directory
    does not exist or contains no well-formed sample files. Gaps are not
    filled — new samples always land at `max + 1`.
    """
    safe_model = display_name.replace("/", "_")
    out_dir = results_dir / "raw" / prompt_variant / safe_model / problem_id
    if not out_dir.exists():
        return 0
    max_idx = -1
    for f in out_dir.glob("sample_*.json"):
        try:
            idx = int(f.stem.removeprefix("sample_"))
        except ValueError:
            continue
        if idx > max_idx:
            max_idx = idx
    return max_idx + 1
```

- [ ] **Step 2: Use the helper in `run_benchmark`**

Replace the inner sample loop in `run_benchmark` (currently the triple-nested loop at the bottom of the file).

Find this block:

```python
    for spec in model_specs:
        for problem in problems:
            for s in range(n_samples):
                record = _run_one_sample(
                    spec, s, problem, system_prompt, prompt_variant
                )
                _write_sample(results_dir, record)
                records.append(record)
```

Replace with:

```python
    for spec in model_specs:
        for problem in problems:
            start = _next_sample_index(
                results_dir, prompt_variant, spec.display_name, problem.problem_id
            )
            for i in range(n_samples):
                record = _run_one_sample(
                    spec, start + i, problem, system_prompt, prompt_variant
                )
                _write_sample(results_dir, record)
                records.append(record)
```

- [ ] **Step 3: Run the new test to verify it passes**

Run: `uv run pytest tests/test_runner_benchmark.py::test_run_benchmark_appends_samples_when_prior_exist -v`

Expected: PASS.

- [ ] **Step 4: Run the full benchmark test file to verify no regressions**

Run: `uv run pytest tests/test_runner_benchmark.py -v`

Expected: All tests pass (including the new one).

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/benchmark.py
git commit -m "feat: append benchmark samples instead of overwriting from index 0"
```

---

## Task 3: RED — test summary aggregates across prior + new samples

**Files:**
- Modify: `tests/test_runner_benchmark.py` (append to end of file)

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_runner_benchmark.py`:

```python
def test_run_benchmark_summary_aggregates_across_prior_and_new_samples(
    tmp_path: Path,
):
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )
    spec = _spec("claude-haiku-4-5-20251001", client)

    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )
    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[spec], n_samples=2, use_cot=True,
    )

    import csv as _csv
    with (results_dir / "summary.csv").open() as f:
        rows = list(_csv.DictReader(f))
    matching = [
        r for r in rows
        if r["prompt_variant"] == "cot"
        and r["model"] == "claude-haiku-4-5-20251001"
        and r["problem_id"] == p.problem_id
    ]
    assert len(matching) == 1, matching
    row = matching[0]
    assert int(row["n_samples"]) == 4, row
    assert int(row["n_correct"]) == 4, row
    assert float(row["accuracy"]) == 1.0, row
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_runner_benchmark.py::test_run_benchmark_summary_aggregates_across_prior_and_new_samples -v`

Expected: FAIL. After Task 2, the second run writes `sample_2/3.json` to disk, but `_write_summary` still aggregates only the two new in-memory records, so `row["n_samples"]` is `2` (not `4`). Since the second run's key collides with the first, the preserved-rows filter drops the prior row, so the final row shows only the new run's counts.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_runner_benchmark.py
git commit -m "test: RED - summary must aggregate across prior + new samples"
```

---

## Task 4: GREEN — aggregate summary from on-disk samples for touched keys

**Files:**
- Modify: `src/llm_relations/runner/benchmark.py`

- [ ] **Step 1: Add a helper that loads on-disk `SampleRecord`s for a key**

Add this helper in `src/llm_relations/runner/benchmark.py`, placed directly below `_next_sample_index`:

```python
def _load_samples_on_disk(
    results_dir: Path,
    prompt_variant: str,
    display_name: str,
    problem_id: str,
) -> list[SampleRecord]:
    """Load every `sample_*.json` for a given (variant, model, problem) key.

    Used by `_write_summary` so that re-runs aggregate across all samples on
    disk (old + new), not just the ones produced by the current run.
    """
    safe_model = display_name.replace("/", "_")
    out_dir = results_dir / "raw" / prompt_variant / safe_model / problem_id
    if not out_dir.exists():
        return []
    samples: list[SampleRecord] = []
    for f in sorted(out_dir.glob("sample_*.json")):
        data = json.loads(f.read_text())
        samples.append(SampleRecord(**data))
    return samples
```

- [ ] **Step 2: Rewrite `_write_summary` to re-aggregate from disk for touched keys**

Find the current `_write_summary` function:

```python
def _write_summary(results_dir: Path, records: list[SampleRecord]) -> None:
    csv_path = results_dir / "summary.csv"
    new_rows = _aggregate(records)
    # Keep prior rows whose (prompt_variant, model, problem_id) key is not
    # being overwritten by this run, so re-running with new prompt variants
    # accumulates rather than clobbers.
    preserved = [
        row for row in _read_existing_summary(csv_path)
        if (row.get("prompt_variant", ""), row["model"], row["problem_id"]) not in new_rows
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in preserved:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDNAMES})
        for agg in new_rows.values():
            writer.writerow(_agg_to_csv_row(agg))
```

Replace it with:

```python
def _write_summary(results_dir: Path, records: list[SampleRecord]) -> None:
    csv_path = results_dir / "summary.csv"

    # Keys touched by this run. For each, aggregate over ALL sample files on
    # disk (prior + new), not just the in-memory records from this run, so
    # appending new samples produces a summary row covering the full set.
    touched_keys = {
        (r.prompt_variant, r.model, r.problem_id) for r in records
    }
    on_disk_records: list[SampleRecord] = []
    for prompt_variant, model, problem_id in touched_keys:
        on_disk_records.extend(
            _load_samples_on_disk(results_dir, prompt_variant, model, problem_id)
        )
    new_rows = _aggregate(on_disk_records)

    # Preserve prior rows whose key is not touched by this run, so re-running
    # with new prompt variants (or new models) accumulates rather than clobbers.
    preserved = [
        row for row in _read_existing_summary(csv_path)
        if (row.get("prompt_variant", ""), row["model"], row["problem_id"]) not in new_rows
    ]

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in preserved:
            writer.writerow({k: row.get(k, "") for k in SUMMARY_FIELDNAMES})
        for agg in new_rows.values():
            writer.writerow(_agg_to_csv_row(agg))
```

- [ ] **Step 3: Run the new test to verify it passes**

Run: `uv run pytest tests/test_runner_benchmark.py::test_run_benchmark_summary_aggregates_across_prior_and_new_samples -v`

Expected: PASS.

- [ ] **Step 4: Run the full benchmark test file to verify no regressions**

Run: `uv run pytest tests/test_runner_benchmark.py -v`

Expected: All tests pass. In particular, `test_run_benchmark_summary_merges_existing_rows` (which runs `cot` then `no_cot`) should still pass — the two runs touch disjoint keys, so each row's on-disk aggregation covers only its own samples.

- [ ] **Step 5: Commit**

```bash
git add src/llm_relations/runner/benchmark.py
git commit -m "feat: aggregate summary.csv across on-disk samples for touched keys"
```

---

## Task 5: RED — test gap handling uses `max+1`

**Files:**
- Modify: `tests/test_runner_benchmark.py` (append to end of file)

- [ ] **Step 1: Write the failing test**

Append this test to `tests/test_runner_benchmark.py`:

```python
def test_run_benchmark_sample_index_skips_gaps(tmp_path: Path):
    """Gaps in sample numbering are not filled — new samples land at max+1."""
    p = _problem()
    problems_dir = tmp_path / "problems"
    problems_dir.mkdir()
    save_problem(p, problems_dir / f"{p.problem_id}.json")
    results_dir = tmp_path / "results"

    # Pre-create sample_0.json and sample_2.json by hand.
    target_dir = (
        results_dir / "raw" / "cot" / "claude-haiku-4-5-20251001" / p.problem_id
    )
    target_dir.mkdir(parents=True)

    def _stub_record(idx: int) -> dict:
        return {
            "problem_id": p.problem_id,
            "model": "claude-haiku-4-5-20251001",
            "sample": idx,
            "variant": "baseline",
            "prompt_variant": "cot",
            "prompt": "prompt",
            "response_text": "stub",
            "parsed_answer": {"analog": "mek", "button_color": "blue"},
            "correct_answer": {"analog": "mek", "button_color": "blue"},
            "is_correct": True,
            "error_type": None,
            "parse_error": False,
            "input_tokens": 0,
            "output_tokens": 0,
            "latency_ms": 0,
            "timestamp": "1970-01-01T00:00:00+00:00",
        }

    (target_dir / "sample_0.json").write_text(
        json.dumps(_stub_record(0), indent=2, sort_keys=True)
    )
    (target_dir / "sample_2.json").write_text(
        json.dumps(_stub_record(2), indent=2, sort_keys=True)
    )

    client = _client_returning(
        '```json\n{"analog": "mek", "button_color": "blue"}\n```'
    )

    run_benchmark(
        problems_dir=problems_dir, results_dir=results_dir,
        model_specs=[_spec("claude-haiku-4-5-20251001", client)],
        n_samples=1, use_cot=True,
    )

    files = sorted(f.name for f in target_dir.glob("sample_*.json"))
    assert files == [
        "sample_0.json", "sample_2.json", "sample_3.json"
    ], files
    # The newly written record is index 3.
    rec = json.loads((target_dir / "sample_3.json").read_text())
    assert rec["sample"] == 3
```

- [ ] **Step 2: Run test to verify it passes**

Run: `uv run pytest tests/test_runner_benchmark.py::test_run_benchmark_sample_index_skips_gaps -v`

Expected: PASS. This test should already pass after Task 2 because `_next_sample_index` uses `max+1`, which is `3` for `{0, 2}`. This task exists as a regression guard — if someone later "fixes" gaps by counting files instead, this test catches it.

Note: this is an exception to strict RED-first because Task 2's implementation already satisfies this behavior. The test is still valuable as a documented invariant.

- [ ] **Step 3: Run the full benchmark test file to confirm**

Run: `uv run pytest tests/test_runner_benchmark.py -v`

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_runner_benchmark.py
git commit -m "test: gap in sample indices should not be filled (max+1 invariant)"
```

---

## Task 6: Full regression run

- [ ] **Step 1: Run the full test suite**

Run: `uv run pytest -v`

Expected: All tests pass. If anything unrelated fails, stop and investigate; do not mark the plan complete.

- [ ] **Step 2: Manual sanity check of `_next_sample_index` behavior**

This is optional but helpful: start a Python REPL and verify the helper directly.

```bash
uv run python -c "
from pathlib import Path
import tempfile, json
from llm_relations.runner.benchmark import _next_sample_index

with tempfile.TemporaryDirectory() as d:
    results = Path(d)
    # Empty directory
    assert _next_sample_index(results, 'cot', 'm', 'p') == 0
    # Populate a couple of files
    td = results / 'raw' / 'cot' / 'm' / 'p'
    td.mkdir(parents=True)
    (td / 'sample_0.json').write_text('{}')
    (td / 'sample_7.json').write_text('{}')
    (td / 'sample_bad.json').write_text('{}')
    assert _next_sample_index(results, 'cot', 'm', 'p') == 8
print('ok')
"
```

Expected output: `ok`.

- [ ] **Step 3: Done**

Plan is complete. No further commits required at this step.

---

## Self-Review

**Spec coverage:**

- Sample numbering (`0`, `K+1`, missing dir) — Task 2 (`_next_sample_index`, loop change), Task 1 (test), Task 5 (gap test).
- Summary reflects union of samples for touched keys — Task 4 (implementation), Task 3 (test).
- Rows for untouched keys preserved verbatim — covered by existing `test_run_benchmark_summary_merges_existing_rows`, re-verified in Task 4 Step 4.
- Gap policy (`max+1`, not fill) — Task 5.
- Malformed filenames ignored — baked into `_next_sample_index` via the `except ValueError: continue` branch; not a dedicated test (low value).
- On-disk layout / schema / CLI unchanged — nothing in the plan touches those.

**Placeholder scan:** None found. Every step has exact code, commands, or expected output.

**Type / name consistency:**

- `_next_sample_index(results_dir, prompt_variant, display_name, problem_id)` — called with matching arg names in Task 2 Step 2 and Task 6 Step 2.
- `_load_samples_on_disk(results_dir, prompt_variant, display_name, problem_id)` — called with `prompt_variant, model, problem_id` in `_write_summary` (Task 4 Step 2). The third positional is `display_name` in the signature but `model` at the call site; both are strings and `SampleRecord.model` is the display name, so this is consistent but the variable name drift is worth a mental note. Left as-is; renaming the call-site loop variable adds noise without clarity.
- `SampleRecord(**data)` in `_load_samples_on_disk` relies on JSON keys matching dataclass fields exactly. `_write_sample` writes via `asdict(record)` with sorted keys, so the round-trip is guaranteed. The stub record in Task 5 Step 1 enumerates every field explicitly — verified against the `SampleRecord` definition at `src/llm_relations/runner/benchmark.py:17`.
