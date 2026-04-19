"""CLI: regenerate summary.csv from every sample_*.json on disk.

Useful after adding new summary columns (e.g. mean_latency_ms_correct,
mean_latency_ms_error) so existing rows are backfilled.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_relations.runner.benchmark import rebuild_summary_from_disk


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    rebuild_summary_from_disk(args.results_dir)
    print(f"Rebuilt {args.results_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
