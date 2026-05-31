"""Eval 1 (dedup corpus): baseline retrieval at k=20; uses chunks_dedup_* + chroma_chunk_dedup_*."""

from __future__ import annotations

import argparse
from pathlib import Path

from langchain_core.documents import Document

from ..retrieval_strategies import base_retriever_id
from ._engine_dedup import run_checkpointed_eval
from .dedup_paths import DEDUP_EVAL_ROOT, DEDUP_GROUND_TRUTH, DEDUP_MANIFEST
from .pairs import SELECTED_PAIRS_EVAL1
from .prefetch_io import document_to_record, record_to_document
from .retrieval_k20 import finalize_from_candidates, run_base_retriever_k, run_retriever_k

EVAL_ID = "eval1_baseline"
DEFAULT_OUT = DEDUP_EVAL_ROOT / "eval1_baseline"
DEFAULT_PREFETCH_DIR = DEDUP_EVAL_ROOT / "prefetch"

_DEDUP_PREFETCH_EPILOG = """
Two-phase run (same pattern as non-dedup top-10; avoids holding Chroma + cross-encoder on GPU):

  Phase 1 - embeddings + Chroma + BM25 only (no cross-encoder):
    EVAL_CUDA_EMPTY_CACHE=1 python -m Scripts.eval.top10.run_eval1_baseline_dedup --prefetch-write

  Phase 2 - load saved candidates; cross-encoder rerank only (no Chroma):
    EVAL_CUDA_EMPTY_CACHE=1 python -m Scripts.eval.top10.run_eval1_baseline_dedup --prefetch-read

  Re-run the same command to resume. Use --no-resume-prefetch to wipe prefetch/eval1_baseline/.

If you already started a *live* eval, delete Data/eval_top10_dedup/eval1_baseline/checkpoint.json
before phase 1.

If phase 2 still OOM on GPU: export RERANK_DEVICE=cpu or lower RERANK_PREDICT_BATCH_SIZE.
If OOM between prefetch-write and prefetch-read on a shared GPU with Ollama embeddings:
  export DEDUP_EVAL_OLLAMA_STOP=1 (see run_dedup_top10_evals.sh header) or stop the embed model manually.
"""


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
    p.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        metavar="chunk:retriever",
        help="Subset of pairs (default: 10 from image)",
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument(
        "--prefetch-dir",
        type=Path,
        default=DEFAULT_PREFETCH_DIR,
        help="Directory for two-phase prefetch JSON (default: under eval_top10_dedup/)",
    )
    p.add_argument("--prefetch-write", action="store_true")
    p.add_argument("--prefetch-read", action="store_true")
    p.add_argument("--no-resume-prefetch", action="store_true")
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
