"""Eval 3: LLM-enhanced question per chunk strategy; retrieve + CE rerank at k=20."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ... import config
from ..retrieval_strategies import base_retriever_id
from .chunk_stats import ensure_chunk_stats
from ._engine import run_checkpointed_eval
from .pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies, retriever_for_eval234
from .prefetch_io import document_to_record, record_to_document
from .question_enhance import QuestionEnhancer, ensure_enhanced_row, require_enhanced_row
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k, run_retriever_k

EVAL_ID = "eval3_enhanced"
DEFAULT_OUT = config.DATA_DIR / "eval_top10" / "eval3_enhanced"
DEFAULT_PREFETCH_DIR = config.DATA_DIR / "eval_top10" / "prefetch"


def _pairs(specs: list[str] | None) -> tuple[tuple[str, str], ...]:
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
    p = argparse.ArgumentParser(description=EVAL_ID)
    p.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH_JSONL)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=config.EVAL_CORPUS_MANIFEST)
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument("--final-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--pairs", nargs="+", default=None, metavar="chunk:retriever")
    p.add_argument("--verbose", action="store_true", help="Log each LLM enhance call")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--prefetch-dir", type=Path, default=DEFAULT_PREFETCH_DIR)
    p.add_argument("--prefetch-write", action="store_true")
    p.add_argument("--prefetch-read", action="store_true")
    p.add_argument("--no-resume-prefetch", action="store_true")
    args = p.parse_args()

    pairs = _pairs(args.pairs)
    strategies = distinct_chunk_strategies(pairs)
    stats = ensure_chunk_stats(strategies)
    need_llm = not args.prefetch_write and not args.prefetch_read
    enhancer = QuestionEnhancer() if need_llm else None
    final_k = args.final_k
    candidate_k = args.candidate_k

    def retrieve_docs(
        row: dict, ctx, cid: str, rid: str, *, qi: int = 0
    ) -> list[Document]:
        tw = int(stats[cid]["target_words"])
        assert enhancer is not None
        er = ensure_enhanced_row(
            cid, row, tw, enhancer, verbose=args.verbose
        )
        return run_retriever_k(
            rid,
            ctx,
            er["enhanced_question"],
            final_k=final_k,
            candidate_k=candidate_k,
        )

    def prefetch_build_payload(row: dict, ctx, cid: str, rid: str, qi: int) -> dict:
        tw = int(stats[cid]["target_words"])
        er = require_enhanced_row(cid, row, tw)
        q = str(er["enhanced_question"])
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
