"""Shared checkpointed eval loop over (chunk_strategy, retriever) pairs."""

from __future__ import annotations

import gc
import json
import os
import shutil
import signal
from pathlib import Path
from typing import Any, Callable

from tqdm import tqdm

from langchain_core.documents import Document

from ... import config
from ...chunking_strategies import chroma_persist_dir
from ...embeddings_chromadb import load_vectorstore, release_chroma_process_cache
from ..metrics import first_hit_rank
from ..rerank_cross_encoder import unload_cross_encoder
from ..retrieval_strategies import (
    DENSE_BASE_IDS,
    RetrievalContext,
    base_retriever_id,
    load_documents_from_chunks_jsonl,
)
from ._shared import atomic_write_json, load_ground_truth, sha256_file, validate_gt_against_manifest
from .pairs import cell_key, pairs_fingerprint
from .prefetch_io import (
    PREFETCH_META_NAME,
    count_prefetched_queries,
    load_prefetch_payload,
    prefetch_meta_matches_disk,
    prefetch_query_path,
    save_prefetch_payload,
    write_prefetch_bundle_meta,
)

CHECKPOINT_VERSION = 1
DEFAULT_CHECKPOINT_NAME = "checkpoint.json"


def _maybe_empty_cuda_cache() -> None:
    if os.environ.get("EVAL_CUDA_EMPTY_CACHE", "").lower() in ("1", "true", "yes"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _meta_dict(
    *,
    eval_id: str,
    ground_truth_path: Path,
    manifest_path: Path,
    pairs: tuple[tuple[str, str], ...],
    limit_queries: int | None,
    final_k: int,
    candidate_k: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = Path(ground_truth_path).resolve()
    mp = Path(manifest_path).resolve()
    m: dict[str, Any] = {
        "version": CHECKPOINT_VERSION,
        "eval_id": eval_id,
        "ground_truth": str(gt),
        "ground_truth_sha256": sha256_file(gt),
        "manifest": str(mp),
        "manifest_sha256": sha256_file(mp),
        "pairs_fingerprint": pairs_fingerprint(pairs),
        "pairs": [[c, r] for c, r in pairs],
        "limit_queries": limit_queries,
        "final_k": final_k,
        "candidate_k": candidate_k,
    }
    if extra:
        m.update(extra)
    return m


def _meta_matches(loaded: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = (
        "version",
        "eval_id",
        "ground_truth",
        "ground_truth_sha256",
        "manifest",
        "manifest_sha256",
        "pairs_fingerprint",
        "limit_queries",
        "final_k",
        "candidate_k",
    )
    for k in keys:
        if loaded.get(k) != current.get(k):
            return False
    # optional extra keys in current must match if present in loaded
    for k in (
        "neighbor_offsets",
        "neighbor_seed_top",
        "multiquery_candidate_k",
        "include_original_query",
    ):
        if k in current and loaded.get(k) != current.get(k):
            return False
    return True


def _run_prefetch_read_only(
    *,
    eval_id: str,
    pairs: tuple[tuple[str, str], ...],
    gt: list[dict],
    nq: int,
    out_dir: Path,
    ck_path: Path,
    meta: dict[str, Any],
    resume: bool,
    no_resume: bool,
    final_k: int,
    prefetch_root: Path,
    prefetch_finalize: Callable[[dict, dict[str, Any], str, str], list[Document]],
) -> None:
    """Phase 2: ranks from disk prefetch only (no Chroma / embedding model)."""
    prefetch_root = Path(prefetch_root)
    bundle_path = prefetch_root / eval_id / PREFETCH_META_NAME
    prefetch_meta_matches_disk(meta, bundle_path)

    completed: dict[str, dict[str, Any]] = {}
    partial: dict[str, Any] | None = None
    loaded = None
    if resume and ck_path.is_file():
        try:
            with open(ck_path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            loaded = None
    if loaded and resume:
        prev_meta = loaded.get("meta") or {}
        if not _meta_matches(prev_meta, meta):
            raise SystemExit(
                f"Checkpoint at {ck_path} does not match this run.\n"
                "Use --no-resume to discard it."
            )
        completed = {k: dict(v) for k, v in (loaded.get("completed") or {}).items()}
        for k, v in list(completed.items()):
            ranks = v.get("ranks")
            if not isinstance(ranks, list) or len(ranks) != nq:
                del completed[k]
        partial = loaded.get("partial")
        if partial and (
            len(partial.get("ranks") or ()) > nq
            or not partial.get("chunk_strategy")
            or not partial.get("retriever")
        ):
            partial = None
        print(f"Resuming {eval_id} (prefetch-read): {len(completed)} cells from {ck_path}")

    stop_requested = False

    def _handle_stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _handle_stop)

    def _persist(partial_out: dict[str, Any] | None) -> None:
        payload = {"meta": meta, "completed": completed, "partial": partial_out}
        atomic_write_json(ck_path, payload)

    total_cells = len(pairs) * nq

    def _ranks_done_count() -> int:
        s = sum(len(v.get("ranks") or ()) for v in completed.values())
        if partial:
            s += len(partial.get("ranks") or ())
        return s

    grid_pbar = tqdm(
        total=total_cells,
        initial=_ranks_done_count(),
        desc=f"{eval_id}_prefetch_read",
        unit="q",
    )
    interrupted = False
    try:
        for cid, rid in pairs:
            if stop_requested:
                interrupted = True
                break
            key = cell_key(cid, rid)
            ranks: list[int | None] = list(completed.get(key, {}).get("ranks") or [])
            if len(ranks) == nq:
                continue
            if partial and partial.get("chunk_strategy") == cid and partial.get("retriever") == rid:
                pr = partial.get("ranks") or []
                if len(pr) < nq:
                    ranks = [int(x) if x is not None else None for x in pr]

            for qi in range(len(ranks), nq):
                if stop_requested:
                    _persist(
                        {
                            "chunk_strategy": cid,
                            "retriever": rid,
                            "ranks": ranks,
                        }
                    )
                    interrupted = True
                    break
                row = gt[qi]
                grid_pbar.set_postfix(
                    chunk=cid,
                    retriever=rid[:32] + ("…" if len(rid) > 32 else ""),
                    qi=f"{qi + 1}/{nq}",
                    refresh=True,
                )
                ppath = prefetch_query_path(prefetch_root, eval_id, cid, rid, qi)
                if not ppath.is_file():
                    raise FileNotFoundError(
                        f"Missing prefetch file {ppath}. Finish --prefetch-write for this query."
                    )
                payload = load_prefetch_payload(ppath)
                docs = prefetch_finalize(row, payload, cid, rid)
                try:
                    r = first_hit_rank(
                        docs,
                        str(row["reference"]),
                        str(row.get("gold_snippet") or ""),
                    )
                finally:
                    del docs
                ranks.append(r)
                if (qi + 1) % 25 == 0:
                    gc.collect()
                    _maybe_empty_cuda_cache()
                grid_pbar.update(1)
                _persist(
                    {
                        "chunk_strategy": cid,
                        "retriever": rid,
                        "ranks": list(ranks),
                    }
                )

            if interrupted:
                break

            completed[key] = {"ranks": list(ranks)}
            _persist(None)
            gc.collect()

        gc.collect()
        unload_cross_encoder()
        _maybe_empty_cuda_cache()

    except KeyboardInterrupt:
        interrupted = True
        try:
            _persist(partial)
        except Exception:
            pass
    finally:
        grid_pbar.close()
        signal.signal(signal.SIGINT, prev_sigint)
        try:
            unload_cross_encoder()
            gc.collect()
            _maybe_empty_cuda_cache()
        except Exception:
            pass

    if interrupted or stop_requested:
        print(f"Stopped. Checkpoint: {ck_path}. Re-run the same command to resume.")
        return

    from ._results import write_csvs

    write_csvs(
        out_dir,
        pairs,
        completed,
        max_rank=final_k,
        ground_truth_path=Path(meta["ground_truth"]),
        limit_queries=meta.get("limit_queries"),
    )
    if ck_path.is_file():
        ck_path.unlink()
        print(f"Removed checkpoint (complete): {ck_path}")


def run_checkpointed_eval(
    *,
    eval_id: str,
    pairs: tuple[tuple[str, str], ...],
    ground_truth_path: Path,
    out_dir: Path,
    manifest_path: Path,
    limit_queries: int | None,
    resume: bool,
    no_resume: bool,
    checkpoint_path: Path | None,
    final_k: int,
    candidate_k: int,
    retrieve_docs: Callable[
        [
            dict,  # gt row
            RetrievalContext,
            str,  # chunk_strategy
            str,  # retriever id for this eval
        ],
        list[Document],
    ],
    extra_meta: dict[str, Any] | None = None,
    prefetch_root: Path | None = None,
    prefetch_write: bool = False,
    prefetch_read: bool = False,
    prefetch_build_payload: Callable[[dict, RetrievalContext, str, str, int], dict[str, Any]]
    | None = None,
    prefetch_finalize: Callable[[dict, dict[str, Any], str, str], list[Document]] | None = None,
    no_resume_prefetch: bool = False,
) -> None:
    """
    For each pair, compute first_hit_rank lists; checkpoint after each query.

    Two-phase mode (GPU memory): ``--prefetch-write`` stores dense/BM25 candidate lists per query
    (embedding pass only); ``--prefetch-read`` loads them and runs cross-encoder rerank without
    loading Chroma, so the reranker can use the GPU alone.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = Path(checkpoint_path) if checkpoint_path else out_dir / DEFAULT_CHECKPOINT_NAME

    gt = load_ground_truth(Path(ground_truth_path))
    if not gt:
        raise ValueError(f"No rows in {ground_truth_path}")
    if limit_queries is not None:
        if limit_queries < 1:
            raise ValueError("limit_queries must be >= 1")
        gt = gt[:limit_queries]
    nq = len(gt)
    validate_gt_against_manifest(gt, Path(manifest_path))

    meta = _meta_dict(
        eval_id=eval_id,
        ground_truth_path=Path(ground_truth_path),
        manifest_path=Path(manifest_path),
        pairs=pairs,
        limit_queries=limit_queries,
        final_k=final_k,
        candidate_k=candidate_k,
        extra=extra_meta,
    )

    if no_resume:
        resume = False
        if ck_path.is_file():
            ck_path.unlink()
        for name in (
            "results_summary.csv",
            "rank_breakdown_long.csv",
            "hit_rate_pivot.csv",
            "per_query_ranks.csv",
        ):
            p = out_dir / name
            if p.is_file():
                p.unlink()

    if prefetch_write and prefetch_read:
        raise ValueError("prefetch_write and prefetch_read are mutually exclusive")
    if prefetch_write:
        if prefetch_root is None or prefetch_build_payload is None:
            raise ValueError("prefetch_write requires prefetch_root and prefetch_build_payload")
    if prefetch_read:
        if prefetch_root is None or prefetch_finalize is None:
            raise ValueError("prefetch_read requires prefetch_root and prefetch_finalize")
        _run_prefetch_read_only(
            eval_id=eval_id,
            pairs=pairs,
            gt=gt,
            nq=nq,
            out_dir=out_dir,
            ck_path=ck_path,
            meta=meta,
            resume=resume,
            no_resume=no_resume,
            final_k=final_k,
            prefetch_root=Path(prefetch_root),
            prefetch_finalize=prefetch_finalize,
        )
        return

    pf_root = Path(prefetch_root) if prefetch_root else None
    if prefetch_write and pf_root is not None:
        eval_pf = pf_root / eval_id
        if no_resume_prefetch and eval_pf.is_dir():
            shutil.rmtree(eval_pf)
        write_prefetch_bundle_meta(eval_pf / PREFETCH_META_NAME, meta)

    completed: dict[str, dict[str, Any]] = {}
    partial: dict[str, Any] | None = None
    loaded = None
    if resume and ck_path.is_file():
        try:
            with open(ck_path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            loaded = None
    if loaded and resume:
        prev_meta = loaded.get("meta") or {}
        if not _meta_matches(prev_meta, meta):
            raise SystemExit(
                f"Checkpoint at {ck_path} does not match this run.\n"
                "Use --no-resume to discard it."
            )
        completed = {k: dict(v) for k, v in (loaded.get("completed") or {}).items()}
        for k, v in list(completed.items()):
            ranks = v.get("ranks")
            if not isinstance(ranks, list) or len(ranks) != nq:
                del completed[k]
        partial = loaded.get("partial")
        if partial and (
            len(partial.get("ranks") or ()) > nq
            or not partial.get("chunk_strategy")
            or not partial.get("retriever")
        ):
            partial = None
        print(f"Resuming {eval_id}: {len(completed)} cells done from {ck_path}")

    stop_requested = False

    def _handle_stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _handle_stop)

    def _persist(partial_out: dict[str, Any] | None) -> None:
        payload = {"meta": meta, "completed": completed, "partial": partial_out}
        atomic_write_json(ck_path, payload)

    total_cells = len(pairs) * nq

    def _ranks_done_count() -> int:
        s = sum(len(v.get("ranks") or ()) for v in completed.values())
        if partial:
            s += len(partial.get("ranks") or ())
        return s

    pf_initial = (
        count_prefetched_queries(pf_root, eval_id, pairs, nq)
        if prefetch_write and pf_root is not None
        else _ranks_done_count()
    )
    grid_pbar = tqdm(
        total=total_cells,
        initial=pf_initial,
        desc=f"{eval_id}_prefetch_write" if prefetch_write else eval_id,
        unit="q",
    )

    interrupted = False
    cur_cid: str | None = None
    vs = None
    documents = None
    docs_loaded = False
    # One context per mode per chunk strategy — rebuilding BM25 + token lists for every
    # (chunk, retriever) cell duplicated ~1M token arrays and blew RSS on large corpora.
    ctx_full: RetrievalContext | None = None
    ctx_dense: RetrievalContext | None = None

    try:
        if prefetch_write:
            unload_cross_encoder()
            gc.collect()
            _maybe_empty_cuda_cache()

        for cid, rid in pairs:
            chunks_path = config.DATA_DIR / f"chunks_{cid}.jsonl"
            if stop_requested:
                interrupted = True
                break
            key = cell_key(cid, rid)
            ranks: list[int | None] = list(completed.get(key, {}).get("ranks") or [])
            if not prefetch_write and len(ranks) == nq:
                continue

            if prefetch_write and pf_root is not None:
                all_pf = True
                for qix in range(nq):
                    if not prefetch_query_path(pf_root, eval_id, cid, rid, qix).is_file():
                        all_pf = False
                        break
                if all_pf:
                    continue

            if partial and partial.get("chunk_strategy") == cid and partial.get("retriever") == rid:
                pr = partial.get("ranks") or []
                if len(pr) < nq:
                    ranks = [int(x) if x is not None else None for x in pr]

            if cid != cur_cid:
                ctx_full = None
                ctx_dense = None
                unload_cross_encoder()
                if vs is not None:
                    del vs
                    vs = None
                if documents is not None:
                    del documents
                    documents = None
                docs_loaded = False
                release_chroma_process_cache()
                gc.collect()
                _maybe_empty_cuda_cache()

                if not chunks_path.is_file():
                    raise FileNotFoundError(f"Missing {chunks_path}")
                persist_dir = chroma_persist_dir(cid)
                if not (Path(persist_dir) / "chroma.sqlite3").is_file():
                    raise FileNotFoundError(f"Missing Chroma at {persist_dir}")
                vs = load_vectorstore(persist_dir)
                cur_cid = cid

            base = base_retriever_id(rid)
            if base not in DENSE_BASE_IDS:
                if not docs_loaded:
                    documents = load_documents_from_chunks_jsonl(chunks_path)
                    docs_loaded = True
                if ctx_full is None:
                    ctx_full = RetrievalContext(vs, documents)
                ctx = ctx_full
            else:
                if ctx_dense is None:
                    ctx_dense = RetrievalContext(vs, None)
                ctx = ctx_dense

            qi_lo = 0 if prefetch_write else len(ranks)
            for qi in range(qi_lo, nq):
                if stop_requested:
                    if not prefetch_write:
                        _persist(
                            {
                                "chunk_strategy": cid,
                                "retriever": rid,
                                "ranks": ranks,
                            }
                        )
                    interrupted = True
                    break
                row = gt[qi]
                grid_pbar.set_postfix(
                    chunk=cid,
                    retriever=rid[:32] + ("…" if len(rid) > 32 else ""),
                    qi=f"{qi + 1}/{nq}",
                    refresh=True,
                )
                if prefetch_write and pf_root is not None:
                    ppath = prefetch_query_path(pf_root, eval_id, cid, rid, qi)
                    if ppath.is_file():
                        grid_pbar.update(1)
                        continue
                    assert prefetch_build_payload is not None
                    payload = prefetch_build_payload(row, ctx, cid, rid, qi)
                    save_prefetch_payload(ppath, payload)
                    del payload
                    if (qi + 1) % 10 == 0:
                        gc.collect(2)
                        _maybe_empty_cuda_cache()
                else:
                    docs = retrieve_docs(row, ctx, cid, rid, qi=qi)
                    try:
                        r = first_hit_rank(
                            docs,
                            str(row["reference"]),
                            str(row.get("gold_snippet") or ""),
                        )
                    finally:
                        del docs
                    ranks.append(r)
                    if (qi + 1) % 25 == 0:
                        gc.collect()
                        _maybe_empty_cuda_cache()
                    _persist(
                        {
                            "chunk_strategy": cid,
                            "retriever": rid,
                            "ranks": list(ranks),
                        }
                    )
                if (qi + 1) % 25 == 0:
                    gc.collect()
                    _maybe_empty_cuda_cache()
                grid_pbar.update(1)

            if interrupted:
                break

            if not prefetch_write:
                completed[key] = {"ranks": list(ranks)}
                _persist(None)
            gc.collect(0)
            _maybe_empty_cuda_cache()

        ctx_full = None
        ctx_dense = None
        if vs is not None:
            del vs
            vs = None
        if documents is not None:
            del documents
            documents = None
        release_chroma_process_cache()
        gc.collect()
        unload_cross_encoder()
        _maybe_empty_cuda_cache()

    except KeyboardInterrupt:
        interrupted = True
        try:
            if not prefetch_write:
                _persist(partial)
        except Exception:
            pass
    finally:
        grid_pbar.close()
        signal.signal(signal.SIGINT, prev_sigint)
        try:
            ctx_full = None
            ctx_dense = None
            if vs is not None:
                del vs
            if documents is not None:
                del documents
            release_chroma_process_cache()
            unload_cross_encoder()
            gc.collect()
            _maybe_empty_cuda_cache()
        except Exception:
            pass

    if interrupted or stop_requested:
        if prefetch_write:
            print(
                "Prefetch write stopped. Re-run the same command to resume "
                "(existing per-query JSON files are kept)."
            )
        else:
            print(f"Stopped. Checkpoint: {ck_path}. Re-run the same command to resume.")
        return

    if prefetch_write:
        print(
            f"Prefetch write complete under {pf_root / eval_id}. "
            "Run the same module with --prefetch-read (same GT/pairs/k) to score with the reranker."
        )
        return

    from ._results import write_csvs

    write_csvs(
        out_dir,
        pairs,
        completed,
        max_rank=final_k,
        ground_truth_path=Path(ground_truth_path),
        limit_queries=limit_queries,
    )
    if ck_path.is_file():
        ck_path.unlink()
        print(f"Removed checkpoint (complete): {ck_path}")
