"""Merge per-eval results_summary.csv from the four top-10 evals into one CSV.

The top-10 pipeline (``Scripts.eval.top10``) has four methods:

1. **eval1_baseline** — baseline retrieval at k=20 (pair 10: no CE rerank on interleave).
2. **eval2_neighbors** — same with graph neighbor expansion.
3. **eval3_enhanced** — LLM-rewritten question per chunk strategy + CE at k=20.
4. **eval4_multiquery** — enhanced + variant queries, multi retrieval, union, dedupe, rerank.

Each run writes ``<out>/results_summary.csv`` (chunk_strategy, retriever, hit_rate, mrr, n).
This script concatenates the four files with ``eval_id`` and ``eval_label`` columns.

Default input layout::

    Data/eval_top10/eval1_baseline/results_summary.csv
    Data/eval_top10/eval2_neighbors/results_summary.csv
    …

Default output::

    Data/eval_top10/results_summary_all_evals.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .. import config

# eval_id -> (subdir under --root, short label for the merged CSV)
EVALS: tuple[tuple[str, str, str], ...] = (
    ("eval1_baseline", "eval1_baseline", "baseline_k20"),
    ("eval2_neighbors", "eval2_neighbors", "neighbors"),
    ("eval3_enhanced", "eval3_enhanced", "llm_enhanced_query"),
    ("eval4_multiquery", "eval4_multiquery", "multi_query_fusion"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=config.DATA_DIR / "eval_top10",
        help="Directory containing eval1_baseline/, eval2_neighbors/, …",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <root>/results_summary_all_evals.csv)",
    )
    args = p.parse_args()
    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / "results_summary_all_evals.csv"

    fieldnames = [
        "eval_id",
        "eval_label",
        "chunk_strategy",
        "retriever",
        "hit_rate",
        "mrr",
        "n",
    ]
    rows_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fieldnames)
        w.writeheader()
        for eval_id, subdir, label in EVALS:
            src = root / subdir / "results_summary.csv"
            if not src.is_file():
                print(f"Skip (missing): {src}")
                continue
            with open(src, newline="", encoding="utf-8") as fin:
                r = csv.DictReader(fin)
                for row in r:
                    w.writerow(
                        {
                            "eval_id": eval_id,
                            "eval_label": label,
                            "chunk_strategy": row.get("chunk_strategy", ""),
                            "retriever": row.get("retriever", ""),
                            "hit_rate": row.get("hit_rate", ""),
                            "mrr": row.get("mrr", ""),
                            "n": row.get("n", ""),
                        }
                    )
                    rows_written += 1
            print(f"Merged {src} ({eval_id})")

    print(f"Wrote {out_path} ({rows_written} rows)")


if __name__ == "__main__":
    main()
