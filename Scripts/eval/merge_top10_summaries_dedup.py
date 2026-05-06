"""Merge eval1_baseline + eval2_neighbors results_summary.csv under eval_top10_dedup/."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .top10.dedup_paths import DEDUP_EVAL_ROOT

EVALS: tuple[tuple[str, str, str], ...] = (
    ("eval1_baseline", "eval1_baseline", "baseline_k20_dedup"),
    ("eval2_neighbors", "eval2_neighbors", "neighbors_dedup"),
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--root",
        type=Path,
        default=DEDUP_EVAL_ROOT,
        help="Directory containing eval1_baseline/, eval2_neighbors/",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV (default: <root>/results_summary_baseline_neighbors.csv)",
    )
    args = p.parse_args()
    root = Path(args.root)
    out_path = (
        Path(args.out)
        if args.out
        else root / "results_summary_baseline_neighbors.csv"
    )

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
