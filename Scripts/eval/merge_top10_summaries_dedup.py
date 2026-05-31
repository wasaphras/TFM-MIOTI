"""Merge eval1_baseline + eval2_neighbors summaries under eval_top10_dedup/."""

from __future__ import annotations

from .corpus_layout import DEDUP
from .merge_summaries import main_from_evals

EVALS: tuple[tuple[str, str, str], ...] = (
    ("eval1_baseline", "eval1_baseline", "baseline_k20_dedup"),
    ("eval2_neighbors", "eval2_neighbors", "neighbors_dedup"),
)


def main() -> None:
    main_from_evals(
        EVALS,
        default_root=DEDUP.eval_top10_root,
        default_out_name="results_summary_baseline_neighbors.csv",
        description="Merge dedup eval1 + eval2 results_summary.csv",
    )


if __name__ == "__main__":
    main()
