"""Eval 2: base retrieve -> neighbor expansion (configurable offsets / seed top) -> dedupe -> CE rerank."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ... import config
from ..retrieval_strategies import _pad_to_k, base_retriever_id
from ._engine import run_checkpointed_eval
from .neighbor_index import NeighborIndex, expand_with_neighbors, load_neighbor_index
from .pairs import SELECTED_PAIRS_EVAL1, retriever_for_eval234
from .prefetch_io import document_to_record, record_to_document
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k

EVAL_ID = "eval2_neighbors"
DEFAULT_OUT = config.DATA_DIR / "eval_top10" / "eval2_neighbors"
DEFAULT_PREFETCH_DIR = config.DATA_DIR / "eval_top10" / "prefetch"


def _parse_neighbor_offsets(spec: str) -> tuple[int, ...]:
    spec = spec.replace(" ", "")
    parts = [p for p in spec.split(",") if p != ""]
    if not parts:
        raise SystemExit("neighbor-offsets must be comma-separated ints, e.g. -2,-1,1,2")
    return tuple(int(p) for p in parts)


def _pairs_eval234(specs: list[str] | None) -> tuple[tuple[str, str], ...]:
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
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--prefetch-dir", type=Path, default=DEFAULT_PREFETCH_DIR)
    p.add_argument("--prefetch-write", action="store_true")
    p.add_argument("--prefetch-read", action="store_true")
    p.add_argument("--no-resume-prefetch", action="store_true")
    p.add_argument(
        "--neighbor-offsets",
        type=str,
        default="-2,-1,1,2",
        help="Comma-separated neighbor offsets within same CELEX (default -2,-1,1,2)",
    )
    p.add_argument(
        "--neighbor-seed-top",
        type=int,
        default=None,
        metavar="N",
        help="Only expand neighbors for the top-N retrieved docs (None = all seeds, legacy).",
    )
    args = p.parse_args()

    pairs = _pairs_eval234(args.pairs)
    final_k = args.final_k
    candidate_k = args.candidate_k
    neighbor_offsets = _parse_neighbor_offsets(args.neighbor_offsets)
    neighbor_seed_top = args.neighbor_seed_top

    idx_cache: dict[str, NeighborIndex] = {}

    def retrieve_docs(
        row: dict, ctx, cid: str, rid: str, *, qi: int = 0
    ) -> list[Document]:
        if cid not in idx_cache:
            idx_cache[cid] = load_neighbor_index(cid)
        nidx = idx_cache[cid]
        base = base_retriever_id(rid)
        q = str(row["question"])
        raw = run_base_retriever_k(base, ctx, q, candidate_k=candidate_k)
        if ctx.documents is None:
            raise RuntimeError("Eval2 requires BM25-capable context (chunks loaded)")
        by_uid = ctx.chunk_uid_map()
        expanded = expand_with_neighbors(
            raw,
            by_uid,
            nidx,
            neighbor_offsets,
            neighbor_seed_top=neighbor_seed_top,
        )
        del raw
        from ..rerank_cross_encoder import rerank_documents

        if not expanded:
            docs = []
        else:
            docs = rerank_documents(q, expanded, top_n=final_k)
        del expanded
        out = _pad_to_k(docs, ctx, q, final_k)[:final_k]
        del docs
        return out

    def prefetch_build_payload(row: dict, ctx, cid: str, rid: str, qi: int) -> dict:
        if cid not in idx_cache:
            idx_cache[cid] = load_neighbor_index(cid)
        nidx = idx_cache[cid]
        base = base_retriever_id(rid)
        q = str(row["question"])
        raw = run_base_retriever_k(base, ctx, q, candidate_k=candidate_k)
        if ctx.documents is None:
            raise RuntimeError("Eval2 requires BM25-capable context (chunks loaded)")
        by_uid = ctx.chunk_uid_map()
        expanded = expand_with_neighbors(
            raw,
            by_uid,
            nidx,
            neighbor_offsets,
            neighbor_seed_top=neighbor_seed_top,
        )
        del raw
        return {
            "query": q,
            "retriever": rid,
            "candidates": [document_to_record(d) for d in expanded],
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
        extra_meta={
            "neighbor_offsets": list(neighbor_offsets),
            "neighbor_seed_top": neighbor_seed_top,
        },
        prefetch_root=args.prefetch_dir,
        prefetch_write=args.prefetch_write,
        prefetch_read=args.prefetch_read,
        prefetch_build_payload=prefetch_build_payload if args.prefetch_write else None,
        prefetch_finalize=prefetch_finalize if args.prefetch_read else None,
        no_resume_prefetch=args.no_resume_prefetch,
    )


if __name__ == "__main__":
    main()
