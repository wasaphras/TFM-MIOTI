"""Merge results_summary.csv from the four standard top-10 evals."""

from __future__ import annotations

from .. import config
from .corpus_layout import STANDARD
from .merge_summaries import main_from_evals

EVALS: tuple[tuple[str, str, str], ...] = (
    ("eval1_baseline", "eval1_baseline", "baseline_k20"),
    ("eval2_neighbors", "eval2_neighbors", "neighbors"),
    ("eval3_enhanced", "eval3_enhanced", "llm_enhanced_query"),
    ("eval4_multiquery", "eval4_multiquery", "multi_query_fusion"),
)


def main() -> None:
    main_from_evals(
        EVALS,
        default_root=STANDARD.eval_top10_root,
        default_out_name="results_summary_all_evals.csv",
        description="Merge eval1-4 results_summary.csv under eval_top10/",
    )


if __name__ == "__main__":
    main()
