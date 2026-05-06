"""Bash-friendly state for dedup top-10 prefetch pipeline (run_dedup_top10_evals.sh)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dedup_paths import DEDUP_EVAL_ROOT, DEDUP_GROUND_TRUTH
from .pairs import SELECTED_PAIRS_EVAL1, retriever_for_eval234
from .prefetch_io import count_prefetched_queries

EVAL_IDS = ("eval1_baseline", "eval2_neighbors")


def _nq_from_ground_truth(gt_path: Path) -> int:
    if not gt_path.is_file():
        return 0
    return sum(1 for line in gt_path.open(encoding="utf-8") if line.strip())


def _pairs_for_eval(eval_id: str) -> tuple[tuple[str, str], ...]:
    if eval_id == "eval1_baseline":
        return SELECTED_PAIRS_EVAL1
    if eval_id == "eval2_neighbors":
        return tuple((c, retriever_for_eval234(c, r)) for c, r in SELECTED_PAIRS_EVAL1)
    raise ValueError(eval_id)


def _prefetch_write_done(prefetch_root: Path, eval_id: str, pairs: tuple[tuple[str, str], ...], nq: int) -> bool:
    if nq <= 0:
        return False
    need = len(pairs) * nq
    got = count_prefetched_queries(prefetch_root, eval_id, pairs, nq)
    return got >= need


def _prefetch_read_done(out_dir: Path, n_pairs: int) -> bool:
    p = out_dir / "results_summary.csv"
    if not p.is_file():
        return False
    with p.open(encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return n_lines == n_pairs + 1


def _merge_done(merged_path: Path) -> bool:
    if not merged_path.is_file():
        return False
    with merged_path.open(encoding="utf-8") as f:
        n_lines = sum(1 for _ in f)
    return n_lines > 1


def compute_state(
    *,
    gt_path: Path,
    prefetch_root: Path,
    eval_root: Path,
    merged_path: Path,
) -> dict[str, int | bool]:
    nq = _nq_from_ground_truth(gt_path)
    n_pairs = len(SELECTED_PAIRS_EVAL1)
    out: dict[str, int | bool] = {
        "nq": nq,
        "n_pairs": n_pairs,
        "prefetch_total_per_eval": len(SELECTED_PAIRS_EVAL1) * nq if nq else 0,
    }
    for eid in EVAL_IDS:
        pairs = _pairs_for_eval(eid)
        out[f"{eid}_prefetch_write_done"] = _prefetch_write_done(
            prefetch_root, eid, pairs, nq
        )
        out[f"{eid}_prefetch_read_done"] = _prefetch_read_done(
            eval_root / eid, n_pairs
        )
    out["merge_done"] = _merge_done(merged_path)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json", action="store_true", help="Print one JSON object to stdout")
    p.add_argument(
        "--export-sh",
        action="store_true",
        help="Print KEY=value lines for bash: eval $(python -m ... --export-sh)",
    )
    args = p.parse_args()

    prefetch_root = DEDUP_EVAL_ROOT / "prefetch"
    merged = DEDUP_EVAL_ROOT / "results_summary_baseline_neighbors.csv"
    st = compute_state(
        gt_path=DEDUP_GROUND_TRUTH,
        prefetch_root=prefetch_root,
        eval_root=DEDUP_EVAL_ROOT,
        merged_path=merged,
    )
    if args.json:
        print(json.dumps(st, indent=2))
        return
    if args.export_sh:
        # Bash-friendly names (no eval_id underscores mid-name confusion)
        print(f"NQ={st['nq']}")
        print(f"N_PAIRS={st['n_pairs']}")
        print(f"E1_PREFETCH_WRITE_DONE={'1' if st['eval1_baseline_prefetch_write_done'] else '0'}")
        print(f"E1_PREFETCH_READ_DONE={'1' if st['eval1_baseline_prefetch_read_done'] else '0'}")
        print(f"E2_PREFETCH_WRITE_DONE={'1' if st['eval2_neighbors_prefetch_write_done'] else '0'}")
        print(f"E2_PREFETCH_READ_DONE={'1' if st['eval2_neighbors_prefetch_read_done'] else '0'}")
        print(f"MERGE_DONE={'1' if st['merge_done'] else '0'}")
        return
    p.print_help()


if __name__ == "__main__":
    main()
