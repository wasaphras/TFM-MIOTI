"""
Run chunk_strategy × retriever eval grid (default 10 chunk strategies × 20 retrievers).

Loads manifest and refuses to run if GT references CELEX ids outside the indexed corpus.

Checkpointing: progress is written to ``eval_grid_checkpoint.json`` in ``--out`` after each
query (atomic replace). Re-run the same command to resume after Ctrl+C or crash. Use
``--no-resume`` to ignore/delete the checkpoint and start from scratch.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import signal
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .. import config
from ..chunking_strategies import CHUNK_STRATEGY_IDS, chroma_persist_dir
from ..embeddings_chromadb import load_vectorstore, release_chroma_process_cache
from .metrics import aggregate_ranks, first_hit_rank
from .rerank_cross_encoder import unload_cross_encoder
from .retrieval_strategies import (
    RERANK_RETRIEVER_IDS,
    RETRIEVER_IDS,
    RetrievalContext,
    base_retriever_id,
    load_documents_from_chunks_jsonl,
    needs_dense_only_context_for_base,
    run_retriever,
)

CHECKPOINT_VERSION = 2
DEFAULT_CHECKPOINT_NAME = "eval_grid_checkpoint.json"


def _cell_key(chunk_strategy: str, retriever: str) -> str:
    return f"{chunk_strategy}::{retriever}"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checkpoint_meta_dict(
    ground_truth_path: Path,
    limit_queries: int | None,
    c_ids: tuple[str, ...],
    r_ids: tuple[str, ...],
) -> dict[str, Any]:
    p = Path(ground_truth_path).resolve()
    return {
        "version": CHECKPOINT_VERSION,
        "ground_truth": str(p),
        "ground_truth_sha256": _sha256_file(p),
        "limit_queries": limit_queries,
        "chunk_strategies": list(c_ids),
        "retrievers": list(r_ids),
    }


def _meta_matches_disk(loaded: dict[str, Any], current: dict[str, Any]) -> bool:
    keys = (
        "version",
        "ground_truth",
        "ground_truth_sha256",
        "limit_queries",
        "chunk_strategies",
        "retrievers",
    )
    for k in keys:
        if loaded.get(k) != current.get(k):
            return False
    return True


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _maybe_empty_cuda_cache() -> None:
    if os.environ.get("EVAL_CUDA_EMPTY_CACHE", "").lower() in ("1", "true", "yes"):
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def load_ground_truth(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Load {path.name}", unit=" lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _validate_gt_against_manifest(gt: list[dict], manifest_path: Path) -> None:
    """Abort if any GT reference CELEX is not in the corpus used to build indices."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run:\n"
            "  python -m Scripts.eval.build_chunk_indices --all [--limit N]\n"
            "Eval refuses to run without a manifest so GT/index scope cannot silently diverge."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    allowed = {str(c) for c in manifest.get("celex_ids", []) if c}
    refs = {str(r["reference"]) for r in gt if r.get("reference")}
    missing = sorted(refs - allowed)
    if missing:
        raise SystemExit(
            f"Ground truth references CELEX ids not in {manifest_path}:\n  {missing}\n"
            "Regenerate ground truth after building indices:\n"
            "  python -m Scripts.eval.ground_truth_generate --n <N>\n"
            "Or rebuild indices with the same train.jsonl scope as when GT was created."
        )


def _chunk_strategy_is_ready(strategy_id: str) -> bool:
    """True if chunks jsonl and Chroma persist exist (same checks as the eval loop)."""
    chunks_path = config.DATA_DIR / f"chunks_{strategy_id}.jsonl"
    persist = Path(chroma_persist_dir(strategy_id))
    return chunks_path.is_file() and (persist / "chroma.sqlite3").is_file()


def run_grid(
    ground_truth_path: Path,
    out_dir: Path,
    chunk_strategies: tuple[str, ...] | None = None,
    retrievers: tuple[str, ...] | None = None,
    manifest_path: Path | None = None,
    limit_queries: int | None = None,
    only_ready_chunk_strategies: bool = False,
    resume: bool = True,
    checkpoint_path: Path | None = None,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_path = Path(checkpoint_path) if checkpoint_path else out_dir / DEFAULT_CHECKPOINT_NAME

    gt = load_ground_truth(ground_truth_path)
    if not gt:
        raise ValueError(f"No rows in {ground_truth_path}")
    if limit_queries is not None:
        if limit_queries < 1:
            raise ValueError("limit_queries must be >= 1")
        gt = gt[:limit_queries]

    mp = Path(manifest_path or config.EVAL_CORPUS_MANIFEST)
    _validate_gt_against_manifest(gt, mp)

    c_ids = chunk_strategies or CHUNK_STRATEGY_IDS
    r_ids = retrievers or RETRIEVER_IDS
    unknown_c = [c for c in c_ids if c not in CHUNK_STRATEGY_IDS]
    if unknown_c:
        raise SystemExit(
            f"Unknown chunk strategy id(s): {unknown_c}. Valid: {list(CHUNK_STRATEGY_IDS)}"
        )
    unknown_r = [r for r in r_ids if r not in RETRIEVER_IDS]
    if unknown_r:
        raise SystemExit(
            f"Unknown retriever id(s): {unknown_r}. Valid: {list(RETRIEVER_IDS)}"
        )

    if only_ready_chunk_strategies:
        requested = list(c_ids)
        c_ids = tuple(c for c in c_ids if _chunk_strategy_is_ready(c))
        skipped = [c for c in requested if c not in c_ids]
        if skipped:
            print(
                "Skipping chunk strategies without chunks_*.jsonl + chroma DB: "
                + ", ".join(skipped)
            )
        if not c_ids:
            raise SystemExit(
                "No chunk strategies are ready (need Data/chunks_<id>.jsonl and "
                "Data/chroma_chunk_<id>/chroma.sqlite3 for each). "
                "Finish build_chunk_indices + embeddings for at least one strategy, "
                "or drop --only-ready-strategies and pass --chunk-strategies explicitly."
            )

    nq = len(gt)
    meta = _checkpoint_meta_dict(Path(ground_truth_path), limit_queries, c_ids, r_ids)

    if not resume and ck_path.exists():
        ck_path.unlink()
        print(f"Removed checkpoint (--no-resume): {ck_path}")

    completed: dict[str, dict[str, Any]] = {}
    partial: dict[str, Any] | None = None
    last_partial: dict[str, Any] | None = None

    loaded = _load_checkpoint(ck_path) if resume else None
    if loaded and resume:
        prev_meta = loaded.get("meta") or {}
        if not _meta_matches_disk(prev_meta, meta):
            raise SystemExit(
                f"Checkpoint at {ck_path} does not match this run "
                f"(ground truth, --limit-queries, or strategy lists changed).\n"
                f"  Use --no-resume to discard it and start over, or restore matching inputs."
            )
        completed = dict(loaded.get("completed") or {})
        partial = loaded.get("partial")
        # Drop corrupted / short entries
        bad = [k for k, v in completed.items() if len(v.get("ranks", ())) != nq]
        for k in bad:
            del completed[k]
        if partial and (
            len(partial.get("ranks") or ()) > nq
            or not partial.get("chunk_strategy")
            or not partial.get("retriever")
        ):
            partial = None
        last_partial = partial
        print(
            f"Resuming eval from checkpoint ({len(completed)} cells done, "
            f"{'1 in progress' if partial else 'none in progress'}): {ck_path}"
        )
    elif resume and ck_path.exists():
        print(f"Ignoring unreadable checkpoint, starting fresh: {ck_path}")

    def _persist(partial_out: dict[str, Any] | None) -> None:
        nonlocal partial, last_partial
        last_partial = partial_out
        partial = partial_out
        payload = {"meta": meta, "completed": completed, "partial": partial_out}
        _atomic_write_json(ck_path, payload)

    def _ranks_done_count() -> int:
        s = sum(len(v["ranks"]) for v in completed.values())
        if partial:
            s += len(partial.get("ranks") or ())
        return s

    def _cell_complete(key: str) -> bool:
        v = completed.get(key)
        return v is not None and len(v.get("ranks", ())) == nq

    def _finalize_from_ranks(
        ranks: list[int | None], cid: str, rid: str
    ) -> tuple[dict, list[dict]]:
        agg = aggregate_ranks(ranks)
        summary = {
            "chunk_strategy": cid,
            "retriever": rid,
            "hit_rate": round(agg["hit_rate"], 4),
            "mrr": round(agg["mrr"], 4),
            "n": agg["n"],
        }
        br: list[dict] = []
        for bucket, cnt in agg["buckets"].items():
            br.append(
                {
                    "chunk_strategy": cid,
                    "retriever": rid,
                    "rank_bucket": bucket,
                    "count": cnt,
                }
            )
        return summary, br

    stop_requested = False

    def _handle_stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _handle_stop)
    prev_sigterm = None
    if hasattr(signal, "SIGTERM"):
        prev_sigterm = signal.signal(signal.SIGTERM, _handle_stop)

    summary_rows: list[dict] = []
    breakdown_rows: list[dict] = []
    interrupted = False

    total_cells = len(c_ids) * len(r_ids) * nq
    initial_done = _ranks_done_count()
    grid_pbar = tqdm(
        total=total_cells,
        initial=initial_done,
        desc="Eval grid (queries)",
        unit="q",
    )

    try:
        for cid in c_ids:
            if stop_requested:
                interrupted = True
                break

            chunks_path = config.DATA_DIR / f"chunks_{cid}.jsonl"
            if not Path(chunks_path).exists():
                raise FileNotFoundError(
                    f"Missing {chunks_path}. Run: python -m Scripts.eval.build_chunk_indices --all"
                )

            baseline_rids = [r for r in r_ids if r not in RERANK_RETRIEVER_IDS]
            rerank_rids = [r for r in r_ids if r in RERANK_RETRIEVER_IDS]
            baseline_dense = [
                r for r in baseline_rids if needs_dense_only_context_for_base(r)
            ]
            baseline_other = [r for r in baseline_rids if r not in baseline_dense]
            rerank_dense = [
                r
                for r in rerank_rids
                if needs_dense_only_context_for_base(base_retriever_id(r))
            ]
            rerank_other = [r for r in rerank_rids if r not in rerank_dense]

            if all(_cell_complete(_cell_key(cid, r)) for r in r_ids):
                per_rid_summary: dict[str, dict] = {}
                per_rid_breakdown: dict[str, list[dict]] = {}
                for rid in r_ids:
                    key = _cell_key(cid, rid)
                    ranks = completed[key]["ranks"]
                    s, br = _finalize_from_ranks(ranks, cid, rid)
                    per_rid_summary[rid] = s
                    per_rid_breakdown[rid] = br
                for rid in r_ids:
                    summary_rows.append(per_rid_summary[rid])
                    breakdown_rows.extend(per_rid_breakdown[rid])
                tqdm.write(f"{cid}: all retrievers loaded from checkpoint (skipped Chroma)")
                continue

            persist_dir = chroma_persist_dir(cid)
            vs = load_vectorstore(persist_dir)

            per_rid_summary = {}
            per_rid_breakdown = {}

            def _normalize_ranks_list(
                raw: list[Any],
            ) -> list[int | None]:
                out: list[int | None] = []
                for x in raw:
                    if x is None:
                        out.append(None)
                    else:
                        out.append(int(x))
                return out

            def _run_cell_queries(
                rid: str,
                ctx: RetrievalContext,
                existing: list[int | None] | None,
            ) -> list[int | None]:
                key = _cell_key(cid, rid)
                ranks: list[int | None] = list(existing) if existing else []
                for qi in range(len(ranks), nq):
                    if stop_requested:
                        _persist(
                            {
                                "chunk_strategy": cid,
                                "retriever": rid,
                                "ranks": ranks,
                            }
                        )
                        return ranks
                    row = gt[qi]
                    docs = run_retriever(rid, ctx, row["question"])
                    r = first_hit_rank(
                        docs, row["reference"], row["gold_snippet"]
                    )
                    ranks.append(r)
                    grid_pbar.update(1)
                    _persist(
                        {
                            "chunk_strategy": cid,
                            "retriever": rid,
                            "ranks": list(ranks),
                        }
                    )
                completed[key] = {"ranks": list(ranks)}
                _persist(None)
                return ranks

            def _process_rid(rid: str, ctx: RetrievalContext) -> None:
                key = _cell_key(cid, rid)
                if _cell_complete(key):
                    ranks = completed[key]["ranks"]
                    s, br = _finalize_from_ranks(ranks, cid, rid)
                    per_rid_summary[rid] = s
                    per_rid_breakdown[rid] = br
                    tqdm.write(
                        f"{cid} / {rid}: hit_rate={s['hit_rate']:.3f} mrr={s['mrr']:.3f} (checkpoint)"
                    )
                    return

                existing: list[int | None] | None = None
                if (
                    partial
                    and partial.get("chunk_strategy") == cid
                    and partial.get("retriever") == rid
                ):
                    pr = partial.get("ranks") or []
                    if len(pr) < nq:
                        existing = _normalize_ranks_list(list(pr))

                ranks_result = _run_cell_queries(rid, ctx, existing)
                if len(ranks_result) < nq:
                    return
                summary, br = _finalize_from_ranks(ranks_result, cid, rid)
                per_rid_summary[rid] = summary
                per_rid_breakdown[rid] = br
                tqdm.write(
                    f"{cid} / {rid}: hit_rate={summary['hit_rate']:.3f} "
                    f"mrr={summary['mrr']:.3f}"
                )

            documents: list | None = None

            if baseline_dense:
                ctx_dense = RetrievalContext(vs, None)
                try:
                    for rid in baseline_dense:
                        if stop_requested:
                            interrupted = True
                            break
                        _process_rid(rid, ctx_dense)
                        gc.collect()
                finally:
                    del ctx_dense

            if not stop_requested and baseline_other:
                documents = load_documents_from_chunks_jsonl(chunks_path)
                ctx_full = RetrievalContext(vs, documents)
                try:
                    for rid in baseline_other:
                        if stop_requested:
                            interrupted = True
                            break
                        _process_rid(rid, ctx_full)
                        gc.collect()
                finally:
                    del ctx_full

            if not stop_requested and rerank_rids:
                unload_cross_encoder()
                _maybe_empty_cuda_cache()
                if documents is not None and not rerank_other:
                    del documents
                    documents = None

            if not stop_requested and rerank_dense:
                ctx_dense = RetrievalContext(vs, None)
                try:
                    for rid in rerank_dense:
                        if stop_requested:
                            interrupted = True
                            break
                        _process_rid(rid, ctx_dense)
                        gc.collect()
                finally:
                    del ctx_dense

            if not stop_requested and rerank_other:
                if documents is None:
                    documents = load_documents_from_chunks_jsonl(chunks_path)
                ctx_full = RetrievalContext(vs, documents)
                try:
                    for rid in rerank_other:
                        if stop_requested:
                            interrupted = True
                            break
                        _process_rid(rid, ctx_full)
                        gc.collect()
                finally:
                    del ctx_full

            if documents is not None:
                del documents

            if not stop_requested and rerank_rids:
                unload_cross_encoder()

            if not stop_requested and not interrupted:
                for rid in r_ids:
                    summary_rows.append(per_rid_summary[rid])
                    breakdown_rows.extend(per_rid_breakdown[rid])

            del vs, per_rid_summary, per_rid_breakdown
            release_chroma_process_cache()
            gc.collect()
            _maybe_empty_cuda_cache()

    except KeyboardInterrupt:
        interrupted = True
        _persist(last_partial)
        tqdm.write(
            "KeyboardInterrupt: checkpoint saved. Re-run the same command to resume."
        )
    finally:
        grid_pbar.close()
        signal.signal(signal.SIGINT, prev_sigint)
        if prev_sigterm is not None:
            signal.signal(signal.SIGTERM, prev_sigterm)

    if interrupted or stop_requested:
        print(
            f"\nEval stopped before completion. Checkpoint: {ck_path}\n"
            "Re-run the **same** command (same ground truth, limits, strategies, retrievers) to resume.\n"
            "Use --no-resume only if you want to discard progress and start over."
        )
        return

    sum_path = out_dir / "results_summary.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_strategy",
                "retriever",
                "hit_rate",
                "mrr",
                "n",
            ],
        )
        w.writeheader()
        for row in tqdm(summary_rows, desc="Write results_summary.csv", unit="row"):
            w.writerow(row)

    br_path = out_dir / "rank_breakdown_long.csv"
    with open(br_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "chunk_strategy",
                "retriever",
                "rank_bucket",
                "count",
            ],
        )
        w.writeheader()
        for row in tqdm(
            breakdown_rows, desc="Write rank_breakdown_long.csv", unit="row"
        ):
            w.writerow(row)

    # Wide pivot: rows = chunk_strategy, columns = retriever, values = hit_rate
    pivot_path = out_dir / "hit_rate_pivot.csv"
    pivot: dict[str, dict[str, float]] = {}
    for row in summary_rows:
        c = row["chunk_strategy"]
        pivot.setdefault(c, {})[row["retriever"]] = row["hit_rate"]
    ret_cols = list(r_ids)
    with open(pivot_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["chunk_strategy"] + list(ret_cols))
        for cid in tqdm(c_ids, desc="Write hit_rate pivot", unit="row"):
            w.writerow(
                [cid] + [pivot.get(cid, {}).get(r, "") for r in ret_cols]
            )

    print(f"Wrote {sum_path}, {br_path}, {pivot_path}")


def main():
    p = argparse.ArgumentParser(description="Run chunk × retriever eval grid.")
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=config.GROUND_TRUTH_JSONL,
    )
    p.add_argument(
        "--out",
        type=Path,
        default=config.EVAL_OUTPUT_DIR,
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=config.EVAL_CORPUS_MANIFEST,
        help="Must match the manifest used when generating ground truth (default: Data/eval_corpus_manifest.json)",
    )
    p.add_argument(
        "--limit-queries",
        type=int,
        default=None,
        metavar="N",
        help="Use only the first N rows from ground truth (default: all). "
        "Total progress = len(chunk_strategies) × len(retrievers) × N.",
    )
    p.add_argument(
        "--chunk-strategies",
        nargs="+",
        default=None,
        metavar="ID",
        help="Subset of chunk strategy ids (default: all 10). Example: --chunk-strategies char_nn_only",
    )
    p.add_argument(
        "--retrievers",
        nargs="+",
        default=None,
        metavar="ID",
        help="Subset of retriever ids (default: all 20, including *_ce_r50 rerank variants). "
        "Example: --retrievers dense_sim_k10 bm25_k10_ce_r50",
    )
    p.add_argument(
        "--only-ready-strategies",
        action="store_true",
        help="Use only chunk strategies that already have chunks_*.jsonl and "
        "Data/chroma_chunk_<id>/chroma.sqlite3 (skip the rest).",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore/delete checkpoint and run from scratch (default: resume from "
        f"{DEFAULT_CHECKPOINT_NAME} in --out if config matches).",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        metavar="PATH",
        help=f"Checkpoint JSON path (default: <out-dir>/{DEFAULT_CHECKPOINT_NAME}).",
    )
    args = p.parse_args()
    if not args.ground_truth.exists():
        raise SystemExit(
            f"Missing {args.ground_truth}. Generate with "
            "python -m Scripts.eval.ground_truth_generate --n <N>"
        )
    c_tup = tuple(args.chunk_strategies) if args.chunk_strategies else None
    r_tup = tuple(args.retrievers) if args.retrievers else None
    run_grid(
        args.ground_truth,
        args.out,
        chunk_strategies=c_tup,
        retrievers=r_tup,
        manifest_path=args.manifest,
        limit_queries=args.limit_queries,
        only_ready_chunk_strategies=args.only_ready_strategies,
        resume=not args.no_resume,
        checkpoint_path=args.checkpoint,
    )


if __name__ == "__main__":
    main()
