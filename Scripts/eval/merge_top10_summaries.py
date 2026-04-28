"""Merge per-eval results_summary.csv from the four top-10 evals into one CSV.

Each run writes ``<out>/results_summary.csv`` (chunk_strategy, retriever, hit_rate, mrr,
optional hit_at_* columns, n). This script concatenates the four files with ``eval_id``
and ``eval_label`` columns.

Default output::

    Data/eval_top10/results_summary_all_evals.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .. import config

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

    sources: list[tuple[str, str, Path]] = []
    for eval_id, subdir, label in EVALS:
        src = root / subdir / "results_summary.csv"
        if src.is_file():
            sources.append((eval_id, label, src))
        else:
            print(f"Skip (missing): {src}")

    header = ["eval_id", "eval_label"]
    for _, _, src in sources:
        with open(src, newline="", encoding="utf-8") as fin:
            dr = csv.DictReader(fin)
            for c in dr.fieldnames or []:
                if c not in header:
                    header.append(c)

    all_rows: list[dict[str, str]] = []
    for eval_id, label, src in sources:
        with open(src, newline="", encoding="utf-8") as fin:
            dr = csv.DictReader(fin)
            for row in dr:
                rec: dict[str, str] = {"eval_id": eval_id, "eval_label": label}
                rec.update({k: row.get(k, "") for k in dr.fieldnames or []})
                all_rows.append(rec)
        print(f"Merged {src} ({eval_id})")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=header, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)

    print(f"Wrote {out_path} ({len(all_rows)} rows)")


if __name__ == "__main__":
    main()
