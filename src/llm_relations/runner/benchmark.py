from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_relations.parser import parse_answer, ParseError
from llm_relations.problem import Problem, load_problem
from llm_relations.runner.client import build_system_prompt
from llm_relations.runner.specs import ModelSpec
from llm_relations.scorer import score_answer


@dataclass(frozen=True)
class SampleRecord:
    problem_id: str
    model: str
    sample: int
    variant: str
    prompt_variant: str
    prompt: str
    response_text: str
    parsed_answer: Optional[dict[str, str]]
    correct_answer: dict[str, str]
    is_correct: bool
    error_type: Optional[str]
    parse_error: bool
    input_tokens: int
    output_tokens: int
    latency_ms: int
    timestamp: str


def _run_one_sample(
    spec: ModelSpec,
    sample: int,
    problem: Problem,
    system_prompt: str,
    prompt_variant: str,
) -> SampleRecord:
    result = spec.client.call(
        model=spec.api_model_name,
        user_prompt=problem.prompt_text,
        system_prompt=system_prompt,
    )
    try:
        parsed = parse_answer(result.response_text)
    except ParseError:
        parsed = None
    score = score_answer(problem, parsed)
    return SampleRecord(
        problem_id=problem.problem_id,
        model=spec.display_name,
        sample=sample,
        variant=problem.variant,
        prompt_variant=prompt_variant,
        prompt=problem.prompt_text,
        response_text=result.response_text,
        parsed_answer=parsed,
        correct_answer=problem.correct_answer,
        is_correct=score.is_correct,
        error_type=score.error_type,
        parse_error=parsed is None,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        latency_ms=result.latency_ms,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _safe_model_name(display_name: str) -> str:
    """Sanitize a model display_name for filesystem use.

    Slashes in display names like 'lmstudio:google/gemma-3n-e4b' would
    otherwise create nested directories. The on-disk layout is flat, so we
    replace slashes with underscores. The JSON record and summary CSV still
    keep the unsanitized display_name.
    """
    return display_name.replace("/", "_")


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
    safe_model = _safe_model_name(display_name)
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
    safe_model = _safe_model_name(display_name)
    out_dir = results_dir / "raw" / prompt_variant / safe_model / problem_id
    if not out_dir.exists():
        return []
    indexed: list[tuple[int, Path]] = []
    for f in out_dir.glob("sample_*.json"):
        try:
            idx = int(f.stem.removeprefix("sample_"))
        except ValueError:
            continue
        indexed.append((idx, f))
    indexed.sort()
    samples: list[SampleRecord] = []
    for _idx, f in indexed:
        data = json.loads(f.read_text())
        samples.append(SampleRecord(**data))
    return samples


def _write_sample(results_dir: Path, record: SampleRecord) -> None:
    # Sanitize for filesystem use only — the JSON record and summary CSV keep
    # the unsanitized display_name. Slashes in model names (e.g.
    # 'lmstudio:google/gemma-3n-e4b') would otherwise create nested dirs.
    safe_model = _safe_model_name(record.model)
    out_dir = results_dir / "raw" / record.prompt_variant / safe_model / record.problem_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"sample_{record.sample}.json").write_text(
        json.dumps(asdict(record), indent=2, sort_keys=True)
    )


SUMMARY_FIELDNAMES = [
    "prompt_variant", "model", "variant", "problem_id",
    "n_samples", "n_correct", "accuracy",
    "n_feature_match", "n_positional_match", "n_other", "n_parse_error",
    "mean_output_tokens", "mean_latency_ms",
    "mean_latency_ms_correct", "mean_latency_ms_error",
]


def _aggregate(records: list[SampleRecord]) -> dict[tuple[str, str, str], dict]:
    rows: dict[tuple[str, str, str], dict] = {}
    for r in records:
        key = (r.prompt_variant, r.model, r.problem_id)
        if key not in rows:
            rows[key] = {
                "prompt_variant": r.prompt_variant,
                "model": r.model,
                "variant": r.variant,
                "problem_id": r.problem_id,
                "n_samples": 0,
                "n_correct": 0,
                "n_feature_match": 0,
                "n_positional_match": 0,
                "n_other": 0,
                "n_parse_error": 0,
                "total_output_tokens": 0,
                "total_latency_ms": 0,
                "total_latency_ms_correct": 0,
                "total_latency_ms_error": 0,
                "n_error": 0,
            }
        agg = rows[key]
        agg["n_samples"] += 1
        agg["n_correct"] += int(r.is_correct)
        if r.error_type == "feature_match":
            agg["n_feature_match"] += 1
        elif r.error_type == "positional_match":
            agg["n_positional_match"] += 1
        elif r.error_type == "parse_error":
            agg["n_parse_error"] += 1
        elif r.error_type == "other":
            agg["n_other"] += 1
        agg["total_output_tokens"] += r.output_tokens
        agg["total_latency_ms"] += r.latency_ms
        if r.is_correct:
            agg["total_latency_ms_correct"] += r.latency_ms
        else:
            agg["total_latency_ms_error"] += r.latency_ms
            agg["n_error"] += 1
    return rows


def _agg_to_csv_row(agg: dict) -> dict:
    n = agg["n_samples"]
    n_correct = agg["n_correct"]
    n_error = agg["n_error"]
    return {
        "prompt_variant": agg["prompt_variant"],
        "model": agg["model"],
        "variant": agg["variant"],
        "problem_id": agg["problem_id"],
        "n_samples": n,
        "n_correct": n_correct,
        "accuracy": n_correct / n if n else 0.0,
        "n_feature_match": agg["n_feature_match"],
        "n_positional_match": agg["n_positional_match"],
        "n_other": agg["n_other"],
        "n_parse_error": agg["n_parse_error"],
        "mean_output_tokens": agg["total_output_tokens"] / n if n else 0,
        "mean_latency_ms": agg["total_latency_ms"] / n if n else 0,
        "mean_latency_ms_correct": (
            agg["total_latency_ms_correct"] / n_correct if n_correct else ""
        ),
        "mean_latency_ms_error": (
            agg["total_latency_ms_error"] / n_error if n_error else ""
        ),
    }


def _read_existing_summary(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def rebuild_summary_from_disk(results_dir: Path) -> None:
    """Rebuild ``summary.csv`` from every ``sample_*.json`` under ``raw/``.

    Useful after changing ``SUMMARY_FIELDNAMES`` (e.g. adding new columns): a
    regular ``run_benchmark`` call only re-aggregates keys it touches, so
    untouched rows carry blanks for new columns. This walks the full on-disk
    sample tree and writes a fresh, fully-populated summary.
    """
    raw_dir = results_dir / "raw"
    if not raw_dir.exists():
        return
    all_records: list[SampleRecord] = []
    # raw/{prompt_variant}/{safe_model}/{problem_id}/sample_*.json
    for sample_file in raw_dir.rglob("sample_*.json"):
        try:
            data = json.loads(sample_file.read_text())
            all_records.append(SampleRecord(**data))
        except (json.JSONDecodeError, TypeError):
            # Skip malformed / schema-drifted records rather than crash.
            continue
    rows = _aggregate(all_records)
    csv_path = results_dir / "summary.csv"
    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for agg in rows.values():
            writer.writerow(_agg_to_csv_row(agg))


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
        requested = set(variants)
        available = {p.variant for p in problems}
        unknown = requested - available
        if unknown:
            raise ValueError(
                f"Unknown variant(s): {sorted(unknown)}. "
                f"Available variants in {problems_dir}: {sorted(available)}"
            )
        problems = [p for p in problems if p.variant in requested]
        if not problems:
            raise ValueError(
                f"No problems matched variants={sorted(requested)} in {problems_dir}"
            )
    system_prompt = build_system_prompt(use_cot=use_cot)
    prompt_variant = "cot" if use_cot else "no_cot"
    records: list[SampleRecord] = []
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
    _write_summary(results_dir, records)
