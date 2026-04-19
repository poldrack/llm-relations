# Append Benchmark Runs Instead of Overwriting

## Problem

`scripts/run_benchmark.py` (via `src/llm_relations/runner/benchmark.py`) always
writes samples starting at index `0`, producing `sample_0.json` …
`sample_{n-1}.json` per (prompt_variant, model, problem_id). Re-running the
script silently overwrites prior samples and replaces the corresponding rows in
`summary.csv`, discarding earlier work.

We want re-runs to **append** new samples to what already exists on disk, and
have `summary.csv` reflect the combined set of samples.

## Desired Behavior

1. When running the benchmark, for each (prompt_variant, model, problem_id):
   - If no samples exist on disk, start indexing at `0` (unchanged).
   - If samples `sample_0.json` … `sample_K.json` already exist, the new run's
     samples begin at index `K+1` and count up from there.
2. `summary.csv` rows for any (prompt_variant, model, problem_id) touched by
   the current run reflect **all** samples on disk for that key — i.e.
   `n_samples`, `n_correct`, `accuracy`, and the aggregate counts/means are
   recomputed across the union of prior and new samples.
3. Rows for keys not touched by the current run are preserved verbatim (current
   behavior).

## Design

### Sample numbering

Add a helper in `src/llm_relations/runner/benchmark.py`:

```python
def _next_sample_index(
    results_dir: Path,
    prompt_variant: str,
    display_name: str,
    problem_id: str,
) -> int:
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

In `run_benchmark`, the inner loop becomes:

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

**Gap policy:** indices returned by the helper are always `max+1`. Gaps in
existing sample numbering (e.g. `sample_0.json`, `sample_2.json`) are not
filled — new samples land at `max+1` (here, `3`). This keeps indexing
append-only and predictable.

**Malformed filenames:** if a file named `sample_xyz.json` exists in the
directory, the `int(...)` parse fails and the file is ignored for index
computation.

### Summary aggregation

Change `_write_summary` so that, for each (prompt_variant, model, problem_id)
touched by this run's new records, aggregation is done over ALL
`sample_*.json` files on disk for that key — not just the records produced in
the current run.

Approach:

1. Build `touched_keys = {(r.prompt_variant, r.model, r.problem_id) for r in new_records}`.
2. For each touched key, glob the corresponding `raw/{prompt_variant}/{safe_model}/{problem_id}/sample_*.json`,
   load each as JSON, and instantiate a `SampleRecord` from it.
3. Aggregate these on-disk records with the existing `_aggregate(...)` helper.
4. Preserve prior summary rows whose key is not in `touched_keys` (unchanged).

This guarantees that when `n_samples=2` is run twice, the summary row shows
`n_samples=4` with accuracy computed over all four.

### What does NOT change

- On-disk layout: `results/raw/{prompt_variant}/{safe_model}/{problem_id}/sample_{N}.json`.
- `SampleRecord` schema.
- CSV schema (`SUMMARY_FIELDNAMES`).
- `_write_sample` internals.
- `run_benchmark` signature.
- `scripts/run_benchmark.py` CLI surface.

## Tests (RED-first)

Add to `tests/test_runner_benchmark.py`:

1. **`test_run_benchmark_appends_samples_when_prior_exist`**
   - Save one problem; run `run_benchmark` with `n_samples=2`; run again with
     `n_samples=2`.
   - Assert 4 files exist under the expected dir.
   - Assert filenames include `sample_0.json` … `sample_3.json`.
   - Assert each record's `sample` field matches its filename index.

2. **`test_run_benchmark_summary_aggregates_across_prior_and_new_samples`**
   - Same two back-to-back runs with a client returning a correct answer.
   - Assert the single row for that (prompt_variant, model, problem_id) has
     `n_samples == 4` and `n_correct == 4`.

3. **`test_run_benchmark_sample_index_skips_gaps`**
   - Pre-create `sample_0.json` and `sample_2.json` by hand in the target dir
     (minimal well-formed `SampleRecord` JSON).
   - Run with `n_samples=1`.
   - Assert the new file is `sample_3.json`.

No existing tests should need to change — first-run behavior is unchanged.

## Out of Scope

- Concurrency / multi-process coordination (no locking is added).
- Rewriting `summary.csv` from disk for untouched keys.
- Changing `sample_*.json` naming or directory layout.
- CLI flag to force the old overwrite behavior (not requested).
