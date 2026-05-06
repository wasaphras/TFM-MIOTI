"""Eval 2 (dedup corpus): neighbors + CE rerank at k=20; uses dedup chunks + neighbor_index_dedup."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ..retrieval_strategies import _pad_to_k, base_retriever_id
from ._engine_dedup import run_checkpointed_eval
from .dedup_paths import DEDUP_EVAL_ROOT, DEDUP_GROUND_TRUTH, DEDUP_MANIFEST
from .neighbor_index_dedup import NeighborIndex, expand_with_neighbors, load_neighbor_index
from .pairs import SELECTED_PAIRS_EVAL1, retriever_for_eval234
from .prefetch_io import document_to_record, record_to_document
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k

EVAL_ID = "eval2_neighbors"
DEFAULT_OUT = DEDUP_EVAL_ROOT / "eval2_neighbors"
DEFAULT_PREFETCH_DIR = DEDUP_EVAL_ROOT / "prefetch"

_DEDUP_PREFETCH_EPILOG = """
Two-phase run (neighbors expansion in phase 1; rerank only in phase 2):

  Phase 1 — Chroma + BM25 + neighbor expansion; save candidates (no cross-encoder):
    EVAL_CUDA_EMPTY_CACHE=1 python -m Scripts.eval.top10.run_eval2_neighbors_dedup --prefetch-write

  Phase 2 — cross-encoder rerank from disk (no Chroma):
    EVAL_CUDA_EMPTY_CACHE=1 python -m Scripts.eval.top10.run_eval2_neighbors_dedup --prefetch-read

  Re-run the same command to resume. Use --no-resume-prefetch to wipe prefetch/eval2_neighbors/.

If you already started a *live* eval, delete Data/eval_top10_dedup/eval2_neighbors/checkpoint.json
before phase 1.

If phase 2 still OOM on GPU: export RERANK_DEVICE=cpu or lower RERANK_PREDICT_BATCH_SIZE.
If OOM between prefetch-write and prefetch-read on a shared GPU with Ollama embeddings:
  export DEDUP_EVAL_OLLAMA_STOP=1 (see run_dedup_top10_evals.sh header) or stop the embed model manually.
"""


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
    p = argparse.ArgumentParser(
        description=f"{EVAL_ID} (dedup train corpus)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_DEDUP_PREFETCH_EPILOG,
    )
    p.add_argument("--ground-truth", type=Path, default=DEDUP_GROUND_TRUTH)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--manifest", type=Path, default=DEDUP_MANIFEST)
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
    args = p.parse_args()

    pairs = _pairs_eval234(args.pairs)
    final_k = args.final_k
    candidate_k = args.candidate_k

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
        expanded = expand_with_neighbors(raw, by_uid, nidx)
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
        expanded = expand_with_neighbors(raw, by_uid, nidx)
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
        extra_meta={"neighbor_offsets": [-2, -1, 1, 2]},
        prefetch_root=args.prefetch_dir,
        prefetch_write=args.prefetch_write,
        prefetch_read=args.prefetch_read,
        prefetch_build_payload=prefetch_build_payload if args.prefetch_write else None,
        prefetch_finalize=prefetch_finalize if args.prefetch_read else None,
        no_resume_prefetch=args.no_resume_prefetch,
    )


if __name__ == "__main__":
    main()
