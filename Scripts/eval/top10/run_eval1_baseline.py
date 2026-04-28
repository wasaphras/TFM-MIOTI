"""Eval 1: baseline retrieval at k=20 (pruned pair grid; see pairs.SELECTED_PAIRS_EVAL1)."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ... import config
from ..retrieval_strategies import base_retriever_id
from ._engine import run_checkpointed_eval
from .pairs import SELECTED_PAIRS_EVAL1
from .prefetch_io import document_to_record, record_to_document
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k, run_retriever_k

EVAL_ID = "eval1_baseline"
DEFAULT_OUT = config.DATA_DIR / "eval_top10" / "eval1_baseline"
DEFAULT_PREFETCH_DIR = config.DATA_DIR / "eval_top10" / "prefetch"


def _parse_pairs(specs: list[str] | None) -> tuple[tuple[str, str], ...]:
    if not specs:
        return SELECTED_PAIRS_EVAL1
    out: list[tuple[str, str]] = []
    for s in specs:
        if ":" not in s:
            raise SystemExit(f"Invalid --pairs entry {s!r}; use chunk:retriever")
        c, r = s.split(":", 1)
        out.append((c.strip(), r.strip()))
    return tuple(out)


def main() -> None:
    p = argparse.ArgumentParser(description=EVAL_ID)
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=config.GROUND_TRUTH_JSONL,
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--manifest",
        type=Path,
        default=config.EVAL_CORPUS_MANIFEST,
    )
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument("--final-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="chunk:retriever",
        help="Subset of pairs (default: SELECTED_PAIRS_EVAL1)",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--prefetch-dir",
        type=Path,
        default=DEFAULT_PREFETCH_DIR,
        help="Directory for two-phase prefetch JSON (default: Data/eval_top10/prefetch)",
    )
    p.add_argument(
        "--prefetch-write",
        action="store_true",
        help="Only run dense/BM25 retrieval and save candidate lists (no cross-encoder)",
    )
    p.add_argument(
        "--prefetch-read",
        action="store_true",
        help="Score from --prefetch-write artifacts (rerank only; no Chroma)",
    )
    p.add_argument(
        "--no-resume-prefetch",
        action="store_true",
        help="With --prefetch-write, delete existing prefetch tree for this eval before writing",
    )
    args = p.parse_args()

    pairs = _parse_pairs(args.pairs)
    final_k = args.final_k
    candidate_k = args.candidate_k

    def retrieve_docs(
        row: dict, ctx, cid: str, rid: str, *, qi: int = 0
    ) -> list[Document]:
        return run_retriever_k(
            rid,
            ctx,
            str(row["question"]),
            final_k=final_k,
            candidate_k=candidate_k,
        )

    def prefetch_build_payload(row: dict, ctx, cid: str, rid: str, qi: int) -> dict:
        q = str(row["question"])
        base = base_retriever_id(rid)
        unique = run_base_retriever_k(base, ctx, q, candidate_k=candidate_k)
        return {
            "query": q,
            "retriever": rid,
            "candidates": [document_to_record(d) for d in unique],
        }

    def prefetch_finalize(row: dict, payload: dict, cid: str, rid: str) -> list[Document]:
        prid = str(payload.get("retriever") or rid)
        q = str(payload["query"])
        unique = [record_to_document(r) for r in payload["candidates"]]
        return finalize_from_candidates(prid, q, unique, final_k=final_k)

    run_checkpointed_eval(
        eval_id=EVAL_ID,
        pairs=pairs,
        ground_truth_path=args.ground_truth,
        out_dir=args.out,
        manifest_path=args.manifest,
        limit_queries=args.limit_queries,
        resume=not args.no_resume,
        no_resume=args.no_resume,
        checkpoint_path=args.checkpoint,
        final_k=final_k,
        candidate_k=candidate_k,
        retrieve_docs=retrieve_docs,
        prefetch_root=args.prefetch_dir,
        prefetch_write=args.prefetch_write,
        prefetch_read=args.prefetch_read,
        prefetch_build_payload=prefetch_build_payload if args.prefetch_write else None,
        prefetch_finalize=prefetch_finalize if args.prefetch_read else None,
        no_resume_prefetch=args.no_resume_prefetch,
    )


if __name__ == "__main__":
    main()
