"""Write results_summary.csv, rank_breakdown_long.csv, hit_rate_pivot, optional per-query ranks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from ..metrics import aggregate_ranks_topk
from ._shared import load_ground_truth
from .pairs import cell_key


def _summary_fieldnames(agg_sample: dict[str, Any], max_rank: int) -> list[str]:
    base = ["chunk_strategy", "retriever", "hit_rate", "mrr"]
    for k in ("hit_at_3", "hit_at_5", "hit_at_10", "hit_at_20"):
        if k in agg_sample:
            base.append(k)
    if max_rank < 20 and f"hit_at_{max_rank}" in agg_sample:
        if f"hit_at_{max_rank}" not in base:
            base.append(f"hit_at_{max_rank}")
    base.append("n")
    return base


def write_csvs(
    out_dir: Path,
    pairs: tuple[tuple[str, str], ...],
    completed: dict[str, dict[str, Any]],
    *,
    max_rank: int = 20,
    aggregate_fn: Callable[..., dict[str, Any]] | None = None,
    ground_truth_path: Path | None = None,
    limit_queries: int | None = None,
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
        row: dict[str, Any] = {
            "chunk_strategy": cid,
            "retriever": rid,
            "hit_rate": round(a["hit_rate"], 4),
            "mrr": round(a["mrr"], 4),
            "n": a["n"],
        }
        for k in ("hit_at_3", "hit_at_5", "hit_at_10", "hit_at_20"):
            if k in a:
                row[k] = round(float(a[k]), 4)
        mk = f"hit_at_{max_rank}"
        if max_rank < 20 and mk in a and mk not in row:
            row[mk] = round(float(a[mk]), 4)
        summary_rows.append(row)

    sample_ranks: list[int | None] = []
    if pairs:
        for cid, rid in pairs:
            key = cell_key(cid, rid)
            if key in completed:
                sample_ranks = completed[key].get("ranks") or []
                break
    sample_agg = agg(sample_ranks) if sample_ranks else aggregate_ranks_topk([], max_rank=max_rank)
    sum_fields = _summary_fieldnames(sample_agg, max_rank)

    sum_path = out_dir / "results_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        for row in tqdm(
            summary_rows,
            desc="Write results_summary.csv",
            unit="row",
            leave=True,
        ):
            w.writerow(row)

    for cid, rid in pairs:
        key = cell_key(cid, rid)
        if key not in completed:
            continue
        ranks = completed[key].get("ranks") or []
        a = agg(ranks)
        for bucket, cnt in a["buckets"].items():
            breakdown_rows.append(
                {
                    "chunk_strategy": cid,
                    "retriever": rid,
                    "rank_bucket": bucket,
                    "count": cnt,
                }
            )

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

    if ground_truth_path is not None and Path(ground_truth_path).is_file():
        gt = load_ground_truth(Path(ground_truth_path))
        if limit_queries is not None:
            gt = gt[: int(limit_queries)]
        pq_path = out_dir / "per_query_ranks.csv"
        with open(pq_path, "w", newline="", encoding="utf-8") as f:
            pw = csv.DictWriter(
                f,
                fieldnames=[
                    "query_index",
                    "query_id",
                    "chunk_strategy",
                    "retriever",
                    "rank",
                ],
            )
            pw.writeheader()
            for cid, rid in pairs:
                key = cell_key(cid, rid)
                if key not in completed:
                    continue
                ranks = completed[key].get("ranks") or []
                for qi, r in enumerate(ranks):
                    if qi >= len(gt):
                        break
                    pw.writerow(
                        {
                            "query_index": qi,
                            "query_id": str(gt[qi].get("id", "")),
                            "chunk_strategy": cid,
                            "retriever": rid,
                            "rank": "" if r is None else r,
                        }
                    )
        print(f"Wrote {pq_path}")

    print(f"Wrote {sum_path}, {br_path}, {pivot_path}")
