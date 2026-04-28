"""Write results_summary.csv and rank_breakdown_long.csv for top10 evals."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ..metrics import aggregate_ranks_topk
from .pairs import cell_key


def write_csvs(
    out_dir: Path,
    pairs: tuple[tuple[str, str], ...],
    completed: dict[str, dict[str, Any]],
    *,
    max_rank: int = 20,
    aggregate_fn: Callable[..., dict[str, Any]] | None = None,
) -> None:
    out_dir = Path(out_dir)
    agg = aggregate_fn or (lambda ranks: aggregate_ranks_topk(ranks, max_rank=max_rank))
    summary_rows: list[dict] = []
    breakdown_rows: list[dict] = []

    for cid, rid in pairs:
        key = cell_key(cid, rid)
        if key not in completed:
            continue
        ranks = completed[key].get("ranks") or []
        a = agg(ranks)
        summary_rows.append(
            {
                "chunk_strategy": cid,
                "retriever": rid,
                "hit_rate": round(a["hit_rate"], 4),
                "mrr": round(a["mrr"], 4),
                "n": a["n"],
            }
        )
        for bucket, cnt in a["buckets"].items():
            breakdown_rows.append(
                {
                    "chunk_strategy": cid,
                    "retriever": rid,
                    "rank_bucket": bucket,
                    "count": cnt,
                }
            )

    sum_path = out_dir / "results_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["chunk_strategy", "retriever", "hit_rate", "mrr", "n"],
        )
        w.writeheader()
        for row in tqdm(
            summary_rows,
            desc="Write results_summary.csv",
            unit="row",
            leave=True,
        ):
            w.writerow(row)

    br_path = out_dir / "rank_breakdown_long.csv"
    with open(br_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["chunk_strategy", "retriever", "rank_bucket", "count"],
        )
        w.writeheader()
        for row in tqdm(
            breakdown_rows,
            desc="Write rank_breakdown_long.csv",
            unit="row",
            leave=True,
        ):
            w.writerow(row)

    pivot_path = out_dir / "hit_rate_pivot.csv"
    pivot: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        pivot.setdefault(row["chunk_strategy"], {})[row["retriever"]] = row["hit_rate"]
    ordered_chunk: list[str] = []
    seen_c: set[str] = set()
    for cid, _ in pairs:
        if cid not in seen_c:
            ordered_chunk.append(cid)
            seen_c.add(cid)
    all_rids = list(dict.fromkeys(r for _, r in pairs))
    with open(pivot_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_strategy"] + all_rids)
        for cid in ordered_chunk:
            w.writerow([cid] + [pivot.get(cid, {}).get(r, "") for r in all_rids])

    print(f"Wrote {sum_path}, {br_path}, {pivot_path}")
