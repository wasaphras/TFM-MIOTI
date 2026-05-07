"""
Step 1: Replay top-k retrieval (prefetch preferred), run constrained RAG generation, write JSONL.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from ... import config
from ...embeddings_chromadb import load_vectorstore, release_chroma_process_cache
from ...retriever import _build_rag_prompt
from ..retrieval_strategies import (
    DENSE_BASE_IDS,
    RetrievalContext,
    base_retriever_id,
    load_documents_from_chunks_jsonl,
)
from ..rerank_cross_encoder import unload_cross_encoder
from ..top10._shared import (
    atomic_write_json,
    load_ground_truth,
    sha256_file,
    validate_gt_against_manifest,
)
from ..top10.dedup_paths import (
    DEDUP_EVAL_ROOT,
    DEDUP_GROUND_TRUTH,
    DEDUP_MANIFEST,
    chunks_jsonl_path_dedup,
    chroma_persist_dir_dedup,
)
from ..top10.pairs import pairs_fingerprint
from ..top10.prefetch_io import (
    document_to_record,
    load_prefetch_payload,
    prefetch_query_path,
    record_to_document,
)
from ..top10.retrieval_k20 import finalize_from_candidates, run_retriever_k
from ..top10.dedup_gpu_teardown import maybe_stop_ollama_embedding_model

DEFAULT_CHUNK = "len_500_o50"
DEFAULT_RETRIEVER = "hyb_fill_dense_then_bm25_ce_r50"
EVAL_ID_PREFETCH = "eval1_baseline"
DEFAULT_OUT = DEDUP_EVAL_ROOT / "llm_triad_len500_hyb_fill" / "rag_responses.jsonl"


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec.get("id"), str):
                done.add(rec["id"])
    return done


def _try_prefetch_docs(
    prefetch_root: Path,
    *,
    eval_id: str,
    cid: str,
    rid: str,
    qi: int,
    q: str,
    final_k: int,
) -> tuple[list[Document] | None, Path | None]:
    p = prefetch_query_path(prefetch_root, eval_id, cid, rid, qi)
    if not p.is_file():
        return None, None
    payload = load_prefetch_payload(p)
    prid = str(payload.get("retriever") or rid)
    pq = str(payload.get("query") or q)
    raw = payload.get("candidates") or []
    unique = [record_to_document(r) for r in raw]
    docs = finalize_from_candidates(prid, pq, unique, final_k=final_k)
    return docs[:final_k], p


def _live_docs(
    *,
    cid: str,
    rid: str,
    q: str,
    final_k: int,
    candidate_k: int,
    vs: object,
    documents: list[Document] | None,
    ctx_dense: RetrievalContext | None,
    ctx_full: RetrievalContext | None,
) -> tuple[list[Document], RetrievalContext | None, RetrievalContext | None]:
    base = base_retriever_id(rid)
    if base not in DENSE_BASE_IDS:
        assert documents is not None
        if ctx_full is None:
            ctx_full = RetrievalContext(vs, documents)
        ctx = ctx_full
    else:
        if ctx_dense is None:
            ctx_dense = RetrievalContext(vs, None)
        ctx = ctx_dense
    docs = run_retriever_k(
        rid,
        ctx,
        q,
        final_k=final_k,
        candidate_k=candidate_k,
    )
    return docs, ctx_dense, ctx_full


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ground-truth", type=Path, default=DEDUP_GROUND_TRUTH)
    p.add_argument("--manifest", type=Path, default=DEDUP_MANIFEST)
    p.add_argument("--chunk-strategy", default=DEFAULT_CHUNK)
    p.add_argument("--retriever", default=DEFAULT_RETRIEVER)
    p.add_argument("--final-k", type=int, default=20)
    p.add_argument("--candidate-k", type=int, default=100)
    p.add_argument(
        "--prefetch-root",
        type=Path,
        default=DEDUP_EVAL_ROOT / "prefetch",
    )
    p.add_argument(
        "--eval-id-prefetch",
        default=EVAL_ID_PREFETCH,
        help="Subdirectory under prefetch (default matches eval1_baseline dedup)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument("--llm-model", default=config.LLM_MODEL)
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip rows whose id already appears in the output JSONL",
    )
    args = p.parse_args()

    gt_path = Path(args.ground_truth)
    gt = load_ground_truth(gt_path)
    if not gt:
        raise SystemExit("Empty ground truth")
    if args.limit_queries is not None:
        gt = gt[: args.limit_queries]
    validate_gt_against_manifest(gt, Path(args.manifest))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gt_sha = sha256_file(gt_path)
    man_sha = sha256_file(Path(args.manifest))
    pair_tp = ((args.chunk_strategy, args.retriever),)
    fp = pairs_fingerprint(pair_tp)

    meta = {
        "schema": "rag_responses_v2",
        "ground_truth": str(gt_path.resolve()),
        "ground_truth_sha256": gt_sha,
        "manifest": str(Path(args.manifest).resolve()),
        "manifest_sha256": man_sha,
        "chunk_strategy": args.chunk_strategy,
        "retriever": args.retriever,
        "pairs_fingerprint": fp,
        "final_k": args.final_k,
        "candidate_k": args.candidate_k,
        "prefetch_root": str(Path(args.prefetch_root).resolve()),
        "eval_id_prefetch": args.eval_id_prefetch,
        "generator_model": args.llm_model,
        "generator_temperature": 0.0,
        "system_prompt_sha256": hashlib.sha256(
            config.SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "notes": (
            "Rows include ground_truth_answer from the GT file (`answer` field); it is LLM-authored, "
            "not independent human adjudication. reference_contexts is [gold_snippet] when present."
        ),
    }

    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")
    if not (args.resume and out_path.is_file()):
        meta["n_queries_total"] = len(gt)
        atomic_write_json(meta_path, meta)

    done_ids = _load_done_ids(out_path) if args.resume else set()

    chat = ChatOllama(model=args.llm_model, temperature=0.0)
    sys_msg = SystemMessage(content=config.SYSTEM_PROMPT)

    prefetch_root = Path(args.prefetch_root)
    cid, rid = args.chunk_strategy, args.retriever
    nq = len(gt)

    vs = None
    documents: list[Document] | None = None
    ctx_dense: RetrievalContext | None = None
    ctx_full: RetrievalContext | None = None
    chunks_path = chunks_jsonl_path_dedup(cid)
    persist_dir = chroma_persist_dir_dedup(cid)

    prefetch_used = 0
    live_used = 0

    mode = "a" if args.resume and out_path.is_file() else "w"

    def ensure_live_stack() -> None:
        nonlocal vs, documents, ctx_dense, ctx_full
        if vs is not None:
            return
        if not chunks_path.is_file():
            raise FileNotFoundError(f"Missing {chunks_path}")
        if not (Path(persist_dir) / "chroma.sqlite3").is_file():
            raise FileNotFoundError(f"Missing Chroma at {persist_dir}")
        vs = load_vectorstore(str(persist_dir))
        base = base_retriever_id(rid)
        if base not in DENSE_BASE_IDS:
            documents = load_documents_from_chunks_jsonl(chunks_path)
        ctx_dense = None
        ctx_full = None

    with open(out_path, mode, encoding="utf-8") as fout:
        for qi, row in enumerate(
            tqdm(gt, desc="rag_generate", unit="q"),
        ):
            qid = str(row.get("id") or "")
            if qid in done_ids:
                continue

            q = str(row["question"])
            docs: list[Document] | None = None
            pf_path: Path | None = None
            docs, pf_path = _try_prefetch_docs(
                prefetch_root,
                eval_id=args.eval_id_prefetch,
                cid=cid,
                rid=rid,
                qi=qi,
                q=q,
                final_k=args.final_k,
            )
            source = "prefetch"
            if docs is None:
                ensure_live_stack()
                assert vs is not None
                docs, ctx_dense, ctx_full = _live_docs(
                    cid=cid,
                    rid=rid,
                    q=q,
                    final_k=args.final_k,
                    candidate_k=args.candidate_k,
                    vs=vs,
                    documents=documents,
                    ctx_dense=ctx_dense,
                    ctx_full=ctx_full,
                )
                source = "live"
                live_used += 1
            else:
                prefetch_used += 1

            assert docs is not None and len(docs) == args.final_k
            user_content = _build_rag_prompt(q, docs)
            msg = chat.invoke([sys_msg, HumanMessage(content=user_content)])
            answer = str(msg.content or "").strip()

            gs = str(row.get("gold_snippet") or "").strip()
            gta = str(row.get("answer") or "").strip()
            record = {
                "id": qid,
                "question": q,
                "reference": str(row.get("reference") or ""),
                "gold_snippet": str(row.get("gold_snippet") or ""),
                "ground_truth_answer": gta,
                "reference_contexts": [gs] if gs else [],
                "retrieved_documents": [document_to_record(d) for d in docs],
                "rag_answer": answer,
                "generator_model": args.llm_model,
                "final_k": args.final_k,
                "candidate_k": args.candidate_k,
                "retrieval_source": source,
                "prefetch_path": str(pf_path) if pf_path else None,
                "query_index": qi,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            os.fsync(fout.fileno())

            if (qi + 1) % 10 == 0:
                gc.collect(2)
                if os.environ.get("EVAL_CUDA_EMPTY_CACHE", "").lower() in (
                    "1",
                    "true",
                    "yes",
                ):
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

    # Teardown live stack if created
    if vs is not None:
        del vs
        vs = None
    if documents is not None:
        del documents
    release_chroma_process_cache()
    unload_cross_encoder()
    gc.collect()
    maybe_stop_ollama_embedding_model()

    meta_update = dict(meta)
    meta_update["prefetch_queries"] = prefetch_used
    meta_update["live_queries"] = live_used
    meta_update["n_queries_total"] = nq
    atomic_write_json(meta_path, meta_update)
    print(f"Wrote {out_path} (prefetch={prefetch_used}, live={live_used})")


if __name__ == "__main__":
    main()
