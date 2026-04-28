"""Bulk LLM phase for eval 3/4: write enhanced_questions + multi_query_questions JSONL (no Chroma / BM25)."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from tqdm import tqdm

from ... import config
from ._shared import load_ground_truth, validate_gt_against_manifest
from .chunk_stats import ensure_chunk_stats
from .pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies, retriever_for_eval234
from .question_enhance import (
    QuestionEnhancer,
    clear_question_disk_caches,
    ensure_enhanced_row,
    ensure_variants_row,
)


def _pairs_from_args(specs: list[str] | None) -> tuple[tuple[str, str], ...]:
    if specs:
        out: list[tuple[str, str]] = []
        for s in specs:
            if ":" not in s:
                raise SystemExit(f"Invalid --pairs entry {s!r}")
            c, r = s.split(":", 1)
            out.append((c.strip(), r.strip()))
        return tuple(out)
    return tuple((c, retriever_for_eval234(c, r)) for c, r in SELECTED_PAIRS_EVAL1)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pre-generate LLM rewritten questions (eval3) and variants (eval4) before embedding prefetch."
    )
    p.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH_JSONL)
    p.add_argument("--manifest", type=Path, default=config.EVAL_CORPUS_MANIFEST)
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument("--pairs", nargs="+", default=None, metavar="chunk:retriever")
    p.add_argument(
        "--eval",
        choices=("eval3", "eval4", "both"),
        default="both",
        help="Which JSONL trees to fill (eval4 also ensures enhanced rows first)",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--llm-max-attempts",
        type=int,
        default=12,
        metavar="N",
        help="Per-question retries when JSON is invalid or Ollama glitches (default: 12)",
    )
    p.add_argument(
        "--retry-sleep-seconds",
        type=float,
        default=2.0,
        metavar="SEC",
        help="Pause before each retry after the first attempt (default: 2; use 0 to disable)",
    )
    p.add_argument(
        "--heuristic-fallback",
        action="store_true",
        help="If variants still fail after all retries, write rule-based paraphrases (loud warning)",
    )
    args = p.parse_args()

    gt = load_ground_truth(Path(args.ground_truth))
    if not gt:
        raise ValueError("Empty ground truth")
    if args.limit_queries is not None:
        gt = gt[: args.limit_queries]
    validate_gt_against_manifest(gt, Path(args.manifest))

    pairs = _pairs_from_args(args.pairs)
    strategies = distinct_chunk_strategies(pairs)
    stats = ensure_chunk_stats(strategies)

    enhancer = QuestionEnhancer()
    try:
        for sid in strategies:
            tw = int(stats[sid]["target_words"])
            if args.eval in ("eval3", "both"):
                for row in tqdm(
                    gt,
                    desc=f"LLM enhance {sid}",
                    unit="q",
                ):
                    ensure_enhanced_row(
                        sid,
                        row,
                        tw,
                        enhancer,
                        verbose=args.verbose,
                        enhance_max_attempts=args.llm_max_attempts,
                        enhance_retry_sleep_s=args.retry_sleep_seconds,
                    )
                clear_question_disk_caches()
                gc.collect()

            if args.eval in ("eval4", "both"):
                for row in tqdm(
                    gt,
                    desc=f"LLM variants {sid}",
                    unit="q",
                ):
                    ensure_variants_row(
                        sid,
                        row,
                        tw,
                        enhancer,
                        verbose=args.verbose,
                        variant_max_attempts=args.llm_max_attempts,
                        variant_retry_sleep_s=args.retry_sleep_seconds,
                        heuristic_fallback=args.heuristic_fallback,
                    )
                clear_question_disk_caches()
                gc.collect()
    finally:
        enhancer.close()
        clear_question_disk_caches()
        gc.collect()

    print("LLM materialization finished. Next: run eval scripts with --prefetch-write (no LLM in that phase).")


if __name__ == "__main__":
    main()
