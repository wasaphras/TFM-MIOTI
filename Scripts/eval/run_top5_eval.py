"""
Run eval for the top-5 (chunk_strategy × retriever) cells from the grid results.

Chunk strategy: len_1000_o100
Retrievers: hyb_*_ce_r50 (RRF k60/k30/fetch40, weighted_norm, interleave).

Uses ``run_grid_eval.run_grid`` with fixed strategy lists; outputs go to a separate directory
from the full grid (default: Data/eval_top5_1000/).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .. import config
from .run_grid_eval import run_grid

TOP5_CHUNK_STRATEGY = "len_1000_o100"

TOP5_RETRIEVERS: tuple[str, ...] = (
    "hyb_rrf_k60_ce_r50",
    "hyb_rrf_k30_ce_r50",
    "hyb_rrf_fetch40_ce_r50",
    "hyb_weighted_norm_ce_r50",
    "hyb_interleave_ce_r50",
)

DEFAULT_GROUND_TRUTH = config.DATA_DIR / "ground_truth_top5_1000.jsonl"
DEFAULT_OUT = config.DATA_DIR / "eval_top5_1000"


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Run top-5 hybrid+CE reranked retrievers on len_1000_o100 only. "
            "Same checkpoint/resume behavior as run_grid_eval."
        )
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=DEFAULT_GROUND_TRUTH,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=config.EVAL_CORPUS_MANIFEST,
        help="Must list all CELEX ids referenced in ground truth (default: full corpus manifest)",
    )
    p.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows from ground truth (default: all)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete checkpoint and run from scratch",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help="Checkpoint JSON path (default: <out>/eval_grid_checkpoint.json)",
    )
    args = p.parse_args()

    if not args.ground_truth.exists():
        raise SystemExit(
            f"Missing {args.ground_truth}. Generate with:\n"
            "  python -m Scripts.eval.ground_truth_generate_top5 --n 1000"
        )

    run_grid(
        args.ground_truth,
        args.out,
        chunk_strategies=(TOP5_CHUNK_STRATEGY,),
        retrievers=TOP5_RETRIEVERS,
        manifest_path=args.manifest,
        limit_queries=args.limit_queries,
        only_ready_chunk_strategies=False,
        resume=not args.no_resume,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
