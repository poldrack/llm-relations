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


def _write_sample(results_dir: Path, record: SampleRecord) -> None:
    # Sanitize for filesystem use only — the JSON record and summary CSV keep
    # the unsanitized display_name. Slashes in model names (e.g.
    # 'lmstudio:google/gemma-3n-e4b') would otherwise create nested dirs.
    safe_model = record.model.replace("/", "_")
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
    return rows


def _agg_to_csv_row(agg: dict) -> dict:
    n = agg["n_samples"]
    return {
        "prompt_variant": agg["prompt_variant"],
        "model": agg["model"],
        "variant": agg["variant"],
        "problem_id": agg["problem_id"],
        "n_samples": n,
        "n_correct": agg["n_correct"],
        "accuracy": agg["n_correct"] / n if n else 0.0,
        "n_feature_match": agg["n_feature_match"],
        "n_positional_match": agg["n_positional_match"],
        "n_other": agg["n_other"],
        "n_parse_error": agg["n_parse_error"],
        "mean_output_tokens": agg["total_output_tokens"] / n if n else 0,
        "mean_latency_ms": agg["total_latency_ms"] / n if n else 0,
    }


def _read_existing_summary(csv_path: Path) -> list[dict]:
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


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


def run_benchmark(
    problems_dir: Path,
    results_dir: Path,
    model_specs: list[ModelSpec],
    n_samples: int,
    use_cot: bool = True,
) -> None:
    problems = [load_problem(f) for f in sorted(problems_dir.glob("*.json"))]
    system_prompt = build_system_prompt(use_cot=use_cot)
    prompt_variant = "cot" if use_cot else "no_cot"
    records: list[SampleRecord] = []
    for spec in model_specs:
        for problem in problems:
            for s in range(n_samples):
                record = _run_one_sample(
                    spec, s, problem, system_prompt, prompt_variant
                )
                _write_sample(results_dir, record)
                records.append(record)
    _write_summary(results_dir, records)
