"""
Oracle-style stages: is the gold chunk in the base candidate pool, only after neighbors,
only after CE rerank, or absent?

Requires Chroma + chunks JSONL for the strategy; run with small --limit-queries first.
"""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

from ... import config
from ...chunking_strategies import chroma_persist_dir
from ...embeddings_chromadb import load_vectorstore, release_chroma_process_cache
from ..metrics import first_hit_rank
from ..rerank_cross_encoder import rerank_documents, unload_cross_encoder
from ..retrieval_strategies import RetrievalContext, base_retriever_id, load_documents_from_chunks_jsonl
from ._shared import load_ground_truth, validate_gt_against_manifest
from .neighbor_index import expand_with_neighbors, load_neighbor_index
from .retrieval_k20 import run_base_retriever_k


def _parse_offsets(spec: str) -> tuple[int, ...]:
    parts = [p for p in spec.replace(" ", "").split(",") if p]
    return tuple(int(x) for x in parts)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chunk-strategy", type=str, default="len_1000_o100")
    p.add_argument("--retriever", type=str, default="hyb_rrf_k60_ce_r50")
    p.add_argument("--ground-truth", type=Path, default=config.GROUND_TRUTH_JSONL)
    p.add_argument("--manifest", type=Path, default=config.EVAL_CORPUS_MANIFEST)
    p.add_argument("--limit-queries", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument("--final-k", type=int, default=20)
    p.add_argument("--neighbor-offsets", type=str, default="-2,-1,1,2")
    p.add_argument("--neighbor-seed-top", type=int, default=None)
    p.add_argument("--skip-neighbors", action="store_true")
    p.add_argument("--skip-rerank", action="store_true")
    args = p.parse_args()

    cid = args.chunk_strategy
    rid = args.retriever
    base = base_retriever_id(rid)
    offsets = _parse_offsets(args.neighbor_offsets)

    gt = load_ground_truth(Path(args.ground_truth))
    gt = gt[: max(1, int(args.limit_queries))]
    validate_gt_against_manifest(gt, Path(args.manifest))

    chunks_path = config.DATA_DIR / f"chunks_{cid}.jsonl"
    if not chunks_path.is_file():
        raise SystemExit(f"Missing {chunks_path}")
    persist = chroma_persist_dir(cid)
    if not (Path(persist) / "chroma.sqlite3").is_file():
        raise SystemExit(f"Missing Chroma at {persist}")

    vs = load_vectorstore(persist)
    documents = load_documents_from_chunks_jsonl(chunks_path)
    ctx = RetrievalContext(vs, documents)
    nidx = None if args.skip_neighbors else load_neighbor_index(cid)
    by_uid = ctx.chunk_uid_map()

    hits_from_base = 0
    hits_from_neighbors_only = 0
    lost_by_rerank = 0
    absent = 0

    for row in gt:
        q = str(row["question"])
        ref = str(row["reference"])
        gold = str(row.get("gold_snippet") or "")
        raw = run_base_retriever_k(base, ctx, q, candidate_k=args.candidate_k)
        r_base = first_hit_rank(raw, ref, gold)

        expanded = raw
        if nidx is not None:
            expanded = expand_with_neighbors(
                raw, by_uid, nidx, offsets, neighbor_seed_top=args.neighbor_seed_top
            )
        r_exp = first_hit_rank(expanded, ref, gold)

        if args.skip_rerank:
            r_ce = r_exp
        else:
            docs_ce = rerank_documents(q, expanded, top_n=args.final_k) if expanded else []
            r_ce = first_hit_rank(docs_ce, ref, gold)
            del docs_ce

        if r_ce is not None:
            if r_base is not None:
                hits_from_base += 1
            else:
                hits_from_neighbors_only += 1
        elif r_exp is not None:
            lost_by_rerank += 1
        else:
            absent += 1
        del raw, expanded

    unload_cross_encoder()
    del vs, documents, ctx
    release_chroma_process_cache()
    gc.collect()

    print(f"Oracle stages (n={len(gt)}, chunk={cid}, retriever={rid}, candidate_k={args.candidate_k}):")
    print(f"  hit_after_ce_topk (gold in base pool): {hits_from_base}")
    print(f"  hit_after_ce_topk (gold only via neighbors): {hits_from_neighbors_only}")
    print(f"  in_expanded_but_not_ce_topk: {lost_by_rerank}")
    print(f"  absent_from_expanded: {absent}")


if __name__ == "__main__":
    main()
