"""Eval 4: enhanced + 2 variants, 3 independent retrievals, union, dedupe, rerank with original Q."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ... import config
from ..retrieval_strategies import _pad_to_k, base_retriever_id, dedupe_preserve_order
from .chunk_stats import ensure_chunk_stats
from ._engine import run_checkpointed_eval
from .pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies, retriever_for_eval234
from .prefetch_io import document_to_record, record_to_document
from .question_enhance import QuestionEnhancer, ensure_variants_row, require_variants_row
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k

EVAL_ID = "eval4_multiquery"
DEFAULT_OUT = config.DATA_DIR / "eval_top10" / "eval4_multiquery"
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
    p.add_argument(
        "--per-query-candidate-k",
        type=int,
        default=80,
        help="Candidate breadth per sub-query before union",
    )
    p.add_argument("--pairs", nargs="+", default=None, metavar="chunk:retriever")
    p.add_argument("--verbose", action="store_true")
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
    pqk = args.per_query_candidate_k

    def retrieve_docs(
        row: dict, ctx, cid: str, rid: str, *, qi: int = 0
    ) -> list[Document]:
        tw = int(stats[cid]["target_words"])
        assert enhancer is not None
        mq = ensure_variants_row(cid, row, tw, enhancer, verbose=args.verbose)
        q0 = str(row["question"])
        base = base_retriever_id(rid)
        pool: list[Document] = []
        for q in (mq["enhanced_question"], mq["variants"][0], mq["variants"][1]):
            batch = run_base_retriever_k(base, ctx, q, candidate_k=pqk)
            pool.extend(batch)
            del batch
        merged = dedupe_preserve_order(pool)
        del pool
        from ..rerank_cross_encoder import rerank_documents

        if not merged:
            docs = []
        else:
            docs = rerank_documents(q0, merged, top_n=final_k)
        del merged
        out = _pad_to_k(docs, ctx, q0, final_k)[:final_k]
        del docs
        return out

    def prefetch_build_payload(row: dict, ctx, cid: str, rid: str, qi: int) -> dict:
        tw = int(stats[cid]["target_words"])
        mq = require_variants_row(cid, row, tw)
        q0 = str(row["question"])
        base = base_retriever_id(rid)
        pool: list[Document] = []
        for q in (mq["enhanced_question"], mq["variants"][0], mq["variants"][1]):
            batch = run_base_retriever_k(base, ctx, q, candidate_k=pqk)
            pool.extend(batch)
            del batch
        merged = dedupe_preserve_order(pool)
        del pool
        return {
            "query_rerank": q0,
            "retriever": rid,
            "candidates": [document_to_record(d) for d in merged],
        }

    def prefetch_finalize(row: dict, payload: dict, cid: str, rid: str) -> list[Document]:
        prid = str(payload.get("retriever") or rid)
        q0 = str(payload["query_rerank"])
        unique = [record_to_document(r) for r in payload["candidates"]]
        return finalize_from_candidates(prid, q0, unique, final_k=final_k)

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
        candidate_k=pqk,
        retrieve_docs=retrieve_docs,
        extra_meta={"multiquery_candidate_k": pqk},
        prefetch_root=args.prefetch_dir,
        prefetch_write=args.prefetch_write,
        prefetch_read=args.prefetch_read,
        prefetch_build_payload=prefetch_build_payload if args.prefetch_write else None,
        prefetch_finalize=prefetch_finalize if args.prefetch_read else None,
        no_resume_prefetch=args.no_resume_prefetch,
    )


if __name__ == "__main__":
    main()
