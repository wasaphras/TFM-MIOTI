"""Compare per_query_ranks.csv from two eval output dirs (e.g. baseline vs neighbors)."""

from __future__ import annotations

import argparse
from pathlib import Path


def _rank_cell(s: str) -> int | None:
    s = (s or "").strip()
    if s == "":
        return None
    return int(s)


def load_per_query(path: Path) -> dict[tuple[str, str, str], int | None]:
    """Map (query_id, chunk_strategy, retriever) -> rank."""
    out: dict[tuple[str, str, str], int | None] = {}
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            key = (
                str(row.get("query_id", "")),
                str(row.get("chunk_strategy", "")),
                str(row.get("retriever", "")),
            )
            out[key] = _rank_cell(str(row.get("rank", "")))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dir_a", type=Path, help="First eval out dir (e.g. eval1_baseline)")
    p.add_argument("dir_b", type=Path, help="Second eval out dir (e.g. eval2_neighbors)")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write movement CSV (default: print summary only)",
    )
    p.add_argument(
        "--label-a",
        type=str,
        default="A",
        help="Label for first directory",
    )
    p.add_argument(
        "--label-b",
        type=str,
        default="B",
        help="Label for second directory",
    )
    args = p.parse_args()
    pa = args.dir_a / "per_query_ranks.csv"
    pb = args.dir_b / "per_query_ranks.csv"
    if not pa.is_file():
        raise SystemExit(f"Missing {pa}")
    if not pb.is_file():
        raise SystemExit(f"Missing {pb}")

    ma = load_per_query(pa)
    mb = load_per_query(pb)
    keys = sorted(set(ma) & set(mb))

    rows_out: list[dict[str, object]] = []
    counts: dict[str, int] = {}
    for k in keys:
        ra, rb = ma[k], mb[k]
        qid, cid, rid = k
        if ra is None and rb is None:
            mov = "miss_both"
            delta = None
        elif ra is None and rb is not None:
            mov = "miss_to_hit"
            delta = None
        elif ra is not None and rb is None:
            mov = "hit_to_miss"
            delta = None
        else:
            assert ra is not None and rb is not None
            d = rb - ra
            delta = d
            if d < 0:
                mov = "rank_improved"
            elif d > 0:
                mov = "rank_worse"
            else:
                mov = "same_rank"
        counts[mov] = counts.get(mov, 0) + 1
        rows_out.append(
            {
                "query_id": qid,
                "chunk_strategy": cid,
                "retriever": rid,
                f"rank_{args.label_a}": "" if ra is None else ra,
                f"rank_{args.label_b}": "" if rb is None else rb,
                "movement": mov,
                "delta_b_minus_a": "" if delta is None else delta,
            }
        )

    print(f"Compared {len(keys)} cells ({args.label_a} vs {args.label_b})")
    for m, c in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {m}: {c}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()) if rows_out else [])
            if rows_out:
                w.writeheader()
                w.writerows(rows_out)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
