"""Merge per-eval results_summary.csv files under a top-10 eval root."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def merge_eval_summaries(
    *,
    root: Path,
    evals: tuple[tuple[str, str, str], ...],
    out_path: Path,
) -> int:
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
        for eval_id, subdir, label in evals:
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
    return rows_written


def main_from_evals(
    evals: tuple[tuple[str, str, str], ...],
    *,
    default_root: Path,
    default_out_name: str,
    description: str,
) -> None:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--root", type=Path, default=default_root)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output CSV (default: <root>/{default_out_name})",
    )
    args = p.parse_args()
    root = Path(args.root)
    out_path = Path(args.out) if args.out else root / default_out_name
    merge_eval_summaries(root=root, evals=evals, out_path=out_path)
