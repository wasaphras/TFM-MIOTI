"""
Build Chroma persist dirs + chunks_<strategy>.jsonl per chunking strategy.

Also writes Data/eval_corpus_manifest.json (CELEX ids in scope) so ground truth
and run_grid_eval stay aligned with the same train.jsonl rows / --limit.

Resume-safe: skip when Chroma row count matches JSONL line count; partial
embeds continue from the next chunk (no re-chunk, no full Chroma wipe).
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from .. import config
from ..chunking_strategies import CHUNK_STRATEGY_IDS, chroma_persist_dir, write_chunks_jsonl_from_df
from ..data_extraction_load import load_data
from ..embeddings_chromadb import (
    chroma_collection_count,
    persist_chroma_from_chunks_jsonl,
    release_chroma_process_cache,
)
from ..preprocess import preprocess_for_rag

import pandas as pd


def _count_jsonl_records(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _write_eval_corpus_manifest(df: pd.DataFrame, limit: int | None) -> None:
    """Record which documents are in scope for chunk/embed; GT + eval must match this set."""
    path = config.EVAL_CORPUS_MANIFEST
    celex_ids = [str(c or "") for c in df["celex_id"].tolist()]
    payload = {
        "celex_ids": celex_ids,
        "n_docs": len(df),
        "limit": limit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote eval corpus manifest ({len(celex_ids)} docs): {path}")


def _train_jsonl_nonempty_doc_count(limit: int | None) -> int:
    """Number of non-empty lines in train.jsonl, capped at ``limit`` when set."""
    n = 0
    with open(config.TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            n += 1
            if limit is not None and n >= limit:
                return n
    return n


def _eval_manifest_is_current(limit: int | None) -> bool:
    """True if eval_corpus_manifest matches train scope and is not older than train.jsonl."""
    mp = config.EVAL_CORPUS_MANIFEST
    train = config.TRAIN_JSONL
    if not mp.is_file() or not train.is_file():
        return False
    try:
        if mp.stat().st_mtime < train.stat().st_mtime:
            return False
        with open(mp, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("limit") != limit:
            return False
        if data.get("n_docs") != _train_jsonl_nonempty_doc_count(limit):
            return False
    except Exception:
        return False
    return True


def _write_eval_corpus_manifest_from_train(limit: int | None) -> None:
    """Same manifest as dataframe path, without loading categories or full preprocess."""
    path = config.EVAL_CORPUS_MANIFEST
    celex_ids: list[str] = []
    with open(config.TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            celex_ids.append(str(rec.get("celex_id") or ""))
            if limit is not None and len(celex_ids) >= limit:
                break
    payload = {
        "celex_ids": celex_ids,
        "n_docs": len(celex_ids),
        "limit": limit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Wrote eval corpus manifest ({len(celex_ids)} docs, from train.jsonl): {path}")


def _strategy_fully_built(strategy_id: str, force: bool) -> bool:
    if force:
        return False
    chunks_path = config.DATA_DIR / f"chunks_{strategy_id}.jsonl"
    if not chunks_path.exists() or chunks_path.stat().st_size == 0:
        return False
    jsonl_n = _count_jsonl_records(chunks_path)
    if jsonl_n <= 0:
        return False
    persist = Path(chroma_persist_dir(strategy_id))
    chroma_n = chroma_collection_count(persist, config.CHROMA_COLLECTION)
    return chroma_n is not None and chroma_n == jsonl_n


def _all_strategies_fully_built(ids: tuple[str, ...], force: bool) -> bool:
    return all(_strategy_fully_built(sid, force) for sid in ids)


def build_one(
    strategy_id: str,
    df: pd.DataFrame | None,
    force: bool,
) -> None:
    try:
        persist = Path(chroma_persist_dir(strategy_id))
        chunks_path = config.DATA_DIR / f"chunks_{strategy_id}.jsonl"

        if force and persist.exists():
            shutil.rmtree(persist)

        jsonl_ok = chunks_path.exists() and chunks_path.stat().st_size > 0
        jsonl_n = _count_jsonl_records(chunks_path) if jsonl_ok else 0
        chroma_n = chroma_collection_count(persist, config.CHROMA_COLLECTION)

        if (
            not force
            and jsonl_ok
            and jsonl_n > 0
            and chroma_n is not None
            and chroma_n == jsonl_n
        ):
            print(
                f"Skip {strategy_id}: already complete "
                f"({chroma_n} vectors == {jsonl_n} JSONL records)"
            )
            return

        need_new_jsonl = force or not jsonl_ok or jsonl_n == 0
        if need_new_jsonl:
            if df is None:
                raise RuntimeError(
                    f"Cannot chunk {strategy_id}: train data was not loaded "
                    f"(try running without a stale skip, or rebuild after updating train.jsonl)."
                )
            tmp_path = chunks_path.with_suffix(chunks_path.suffix + ".tmp")
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_n = write_chunks_jsonl_from_df(df, strategy_id, tmp_path)
            tmp_path.replace(chunks_path)
            print(f"Wrote {jsonl_n} chunks to {chunks_path}")
        else:
            print(
                f"Resume {strategy_id}: reusing JSONL ({jsonl_n} chunks); "
                f"Chroma had {chroma_n!r}, will re-embed if needed"
            )

        # New chunking invalidates any existing vectors (chunk_uid / order differ).
        if need_new_jsonl and persist.exists():
            print(
                f"Regenerated chunks JSONL for {strategy_id}; "
                f"removing old Chroma (vectors would not match new chunks)"
            )
            shutil.rmtree(persist)

        expected = jsonl_n
        chroma_now = chroma_collection_count(persist, config.CHROMA_COLLECTION)

        if chroma_now == expected:
            print(
                f"Skip {strategy_id}: Chroma already matches JSONL "
                f"({expected} vectors)"
            )
            return

        skip_first = 0
        if chroma_now is None:
            if persist.exists():
                print(f"Removing unusable Chroma dir {persist} (cannot read collection)")
                shutil.rmtree(persist)
            persist.mkdir(parents=True, exist_ok=True)
        elif chroma_now > expected:
            print(
                f"Chroma has more vectors than JSONL ({chroma_now} > {expected}); "
                f"rebuilding {persist}"
            )
            shutil.rmtree(persist)
            persist.mkdir(parents=True, exist_ok=True)
        else:
            skip_first = chroma_now
            print(
                f"Resume {strategy_id}: {skip_first}/{expected} vectors in Chroma; "
                f"embedding from chunk index {skip_first}"
            )
            persist.mkdir(parents=True, exist_ok=True)

        print(f"Embedding Chroma for {strategy_id} ({expected} chunks)…")
        persist_chroma_from_chunks_jsonl(
            chunks_path,
            persist_directory=persist,
            collection_name=config.CHROMA_COLLECTION,
            expected_n=expected,
            skip_first=skip_first,
        )
        print(f"Chroma ready: {persist}")
    finally:
        release_chroma_process_cache()


def main():
    p = argparse.ArgumentParser(
        description=(
            "Build Chroma indices per chunk strategy.\n\n"
            "Default (no --force): resume-friendly. Each strategy is skipped when "
            "chunks_*.jsonl record count matches Chroma vector count. If embedding "
            "stopped mid-way, the same strategy resumes from the next vector "
            "(partial Chroma is kept).\n\n"
            "With --force: every selected strategy is wiped and rebuilt (JSONL + Chroma). "
            "Do not use --force if you only want to continue or verify a finished build."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Build every strategy in CHUNK_STRATEGY_IDS",
    )
    p.add_argument(
        "--only-strategy",
        type=str,
        default=None,
        help="Single strategy id (e.g. len_2000_o0)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "FULL REBUILD: delete each strategy's Chroma dir and regenerate JSONL + "
            "embeddings. Skipping is disabled. Omit this flag to resume or skip "
            "already-complete strategies."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use first N documents only (for quick tests)",
    )
    args = p.parse_args()

    if args.force:
        print(
            "\n  --force: rebuilding every selected strategy (JSONL + Chroma). "
            "For skip/resume on already-complete indices, run the same command WITHOUT --force.\n"
        )

    if args.all:
        ids = CHUNK_STRATEGY_IDS
    elif args.only_strategy:
        ids = (args.only_strategy,)
    else:
        p.error("Pass --all or --only-strategy <id>")

    for sid in ids:
        if sid not in CHUNK_STRATEGY_IDS:
            raise SystemExit(f"Unknown strategy: {sid}. Valid: {CHUNK_STRATEGY_IDS}")

    ids_t = tuple(ids)
    if not args.force and _all_strategies_fully_built(ids_t, args.force):
        if not _eval_manifest_is_current(args.limit):
            _write_eval_corpus_manifest_from_train(args.limit)
        print(
            "All selected strategies are already complete; skipping train.jsonl load "
            "and preprocess (saves RAM)."
        )
        df = None
    else:
        print("Loading data once for all strategies…")
        df = load_data(limit=args.limit)
        df = preprocess_for_rag(df)
        print(f"Loaded {len(df)} documents.")
        _write_eval_corpus_manifest(df, args.limit)

    for sid in tqdm(ids, desc="Chunk strategies (build indices)", unit="strategy"):
        build_one(sid, df=df, force=args.force)


if __name__ == "__main__":
    main()
