"""
Filter existing chunks_*.jsonl + Chroma into dedup paths without re-embedding.

Reads CELEX ids from Data/train_dedup.jsonl, streams Data/chunks_<strategy>.jsonl
to Data/chunks_dedup_<strategy>.jsonl (same records, subset by metadata.celex_id),
copies Data/chroma_chunk_<strategy>/ to Data/chroma_chunk_dedup_<strategy>/, then
deletes Chroma rows whose celex_id is not in the dedup set.

Does NOT call Ollama or persist_chroma_from_chunks_jsonl.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from tqdm import tqdm

from .. import config
from ..chunking_strategies import CHUNK_STRATEGY_IDS, chroma_persist_dir
from ..embeddings_chromadb import chroma_collection_count, release_chroma_process_cache
from .top10.dedup_paths import (
    DEDUP_MANIFEST,
    DEDUP_TRAIN_JSONL,
    chroma_persist_dir_dedup,
    chunks_jsonl_path_dedup,
)
from .top10.pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_jsonl_records(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _load_dedup_celex_ordered(train_path: Path) -> list[str]:
    celex_ids: list[str] = []
    seen: set[str] = set()
    dupes: list[str] = []
    line_no = 0
    with open(train_path, encoding="utf-8") as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            celex = str(rec.get("celex_id") or "").strip()
            text = rec.get("text")
            if not celex:
                raise SystemExit(f"{train_path}: line {line_no}: missing celex_id")
            if text is None or str(text).strip() == "":
                raise SystemExit(f"{train_path}: line {line_no} ({celex}): empty text")
            if celex in seen:
                dupes.append(celex)
            seen.add(celex)
            celex_ids.append(celex)
    if dupes:
        raise SystemExit(
            f"{train_path}: duplicate celex_id values (first few): {sorted(set(dupes))[:20]}"
        )
    return celex_ids


def _filter_chunks_jsonl(
    *,
    strategy_id: str,
    keep_celex: set[str],
    out_path: Path,
) -> tuple[int, int]:
    """Return (kept_count, dropped_count)."""
    src = config.DATA_DIR / f"chunks_{strategy_id}.jsonl"
    if not src.is_file():
        raise FileNotFoundError(f"Missing source chunks file: {src}")
    kept = 0
    dropped = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"filter chunks {strategy_id}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            c = str(meta.get("celex_id") or "").strip()
            if c in keep_celex:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    tmp.replace(out_path)
    return kept, dropped


def _delete_chroma_not_in_celex(
    persist_dedup: Path,
    keep_celex: set[str],
    *,
    batch_size: int = 1000,
    page_size: int = 5000,
) -> int:
    """Delete embeddings whose metadata celex_id is not in keep_celex. Returns deleted count."""
    client = chromadb.PersistentClient(path=str(persist_dedup))
    coll = client.get_collection(name=config.CHROMA_COLLECTION)
    to_delete: list[str] = []
    offset = 0
    while True:
        data = coll.get(
            include=["metadatas"],
            limit=page_size,
            offset=offset,
        )
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        if not ids:
            break
        if len(ids) != len(metas):
            raise RuntimeError(
                f"Chroma get() id/metadata length mismatch: ids={len(ids)} metas={len(metas)}"
            )
        for doc_id, meta in zip(ids, metas):
            m = meta or {}
            c = str(m.get("celex_id") or "").strip()
            if c not in keep_celex:
                to_delete.append(str(doc_id))
        offset += len(ids)
        if len(ids) < page_size:
            break

    for i in tqdm(range(0, len(to_delete), batch_size), desc="Chroma delete batches", unit="batch"):
        batch = to_delete[i : i + batch_size]
        if batch:
            coll.delete(ids=batch)
    return len(to_delete)


def _verify_uids_in_chroma(persist_dedup: Path, uids: list[str], *, batch: int = 500) -> None:
    client = chromadb.PersistentClient(path=str(persist_dedup))
    coll = client.get_collection(name=config.CHROMA_COLLECTION)
    missing: list[str] = []
    for i in tqdm(range(0, len(uids), batch), desc="verify chunk_uid in Chroma", unit="batch"):
        sl = uids[i : i + batch]
        got = coll.get(ids=sl, include=[])
        back = set(got.get("ids") or [])
        for u in sl:
            if u not in back:
                missing.append(u)
    if missing:
        raise SystemExit(
            f"Chroma missing {len(missing)} chunk_uids (first 10): {missing[:10]}"
        )


def _dedup_jsonl_celex_subset(chunks_path: Path, keep_celex: set[str]) -> bool:
    """True if every record's metadata.celex_id is in keep_celex."""
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            c = str(meta.get("celex_id") or "").strip()
            if c not in keep_celex:
                return False
    return True


def build_one_dedup(
    strategy_id: str,
    keep_celex: set[str],
    *,
    force: bool,
) -> dict[str, int | str]:
    """Build filtered chunks JSONL + filtered Chroma copy. Returns stats dict."""
    if strategy_id not in CHUNK_STRATEGY_IDS:
        raise SystemExit(f"Unknown strategy: {strategy_id}. Valid: {CHUNK_STRATEGY_IDS}")

    src_chunks = config.DATA_DIR / f"chunks_{strategy_id}.jsonl"
    src_persist = Path(chroma_persist_dir(strategy_id))
    dst_chunks = chunks_jsonl_path_dedup(strategy_id)
    dst_persist = chroma_persist_dir_dedup(strategy_id)

    if not src_chunks.is_file():
        raise FileNotFoundError(f"Missing {src_chunks}")
    if not (src_persist / "chroma.sqlite3").is_file():
        raise FileNotFoundError(f"Missing Chroma at {src_persist}")

    old_jsonl_n = _count_jsonl_records(src_chunks)
    old_chroma_n = chroma_collection_count(src_persist, config.CHROMA_COLLECTION)
    if old_chroma_n is None or old_chroma_n != old_jsonl_n:
        raise SystemExit(
            f"{strategy_id}: source Chroma count {old_chroma_n!r} != source JSONL {old_jsonl_n}. "
            "Repair/rebuild the original index first (python -m Scripts.eval.build_chunk_indices …)."
        )

    if not force and dst_chunks.is_file() and (dst_persist / "chroma.sqlite3").is_file():
        cur_jsonl_n = _count_jsonl_records(dst_chunks)
        cur_chroma_n = chroma_collection_count(dst_persist, config.CHROMA_COLLECTION)
        if (
            cur_jsonl_n > 0
            and cur_chroma_n is not None
            and cur_chroma_n == cur_jsonl_n
            and _dedup_jsonl_celex_subset(dst_chunks, keep_celex)
        ):
            print(
                f"Skip {strategy_id}: dedup chunks+Chroma already match "
                f"({cur_jsonl_n} vectors); use --force to rebuild"
            )
            return {
                "strategy_id": strategy_id,
                "skipped": 1,
                "dedup_jsonl_n": cur_jsonl_n,
                "source_jsonl_n": old_jsonl_n,
                "dropped_chunks": old_jsonl_n - cur_jsonl_n,
                "chroma_deleted_rows": 0,
            }

    kept, dropped = _filter_chunks_jsonl(
        strategy_id=strategy_id, keep_celex=keep_celex, out_path=dst_chunks
    )
    print(
        f"{strategy_id}: wrote {dst_chunks} ({kept} kept, {dropped} dropped from source {old_jsonl_n})"
    )

    chroma_now = (
        chroma_collection_count(dst_persist, config.CHROMA_COLLECTION)
        if (dst_persist / "chroma.sqlite3").is_file()
        else None
    )
    need_chroma_rebuild = force or chroma_now is None or chroma_now != kept

    if need_chroma_rebuild:
        if dst_persist.exists():
            shutil.rmtree(dst_persist)
        print(f"{strategy_id}: copying Chroma {src_persist} -> {dst_persist} (no re-embedding)")
        shutil.copytree(src_persist, dst_persist)
        deleted = _delete_chroma_not_in_celex(dst_persist, keep_celex)
        print(f"{strategy_id}: deleted {deleted} Chroma rows not in dedup CELEX set")
    else:
        deleted = 0
        print(
            f"{strategy_id}: dedup Chroma row count already matches JSONL ({kept}); "
            "skipping copy/delete"
        )
    release_chroma_process_cache()

    final_chroma = chroma_collection_count(dst_persist, config.CHROMA_COLLECTION)
    if final_chroma != kept:
        raise SystemExit(
            f"{strategy_id}: Chroma count {final_chroma!r} != dedup JSONL {kept}. "
            "Aborting."
        )

    uids: list[str] = []
    with open(dst_chunks, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            uid = str(meta.get("chunk_uid") or "").strip()
            if uid:
                uids.append(uid)
    if len(uids) != kept:
        raise RuntimeError(f"{strategy_id}: uid list length mismatch")
    _verify_uids_in_chroma(dst_persist, uids)

    return {
        "strategy_id": strategy_id,
        "skipped": 0,
        "dedup_jsonl_n": kept,
        "source_jsonl_n": old_jsonl_n,
        "dropped_chunks": dropped,
        "chroma_deleted_rows": deleted,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Filter full chunk JSONL + Chroma into dedup_* paths using train_dedup.jsonl. "
            "No embeddings."
        )
    )
    p.add_argument(
        "--top10",
        action="store_true",
        help="Build distinct chunk strategies from SELECTED_PAIRS_EVAL1 (top-10 eval)",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        metavar="ID",
        default=None,
        help="Chunk strategy ids (e.g. len_1000_o100)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Rebuild dedup chunks JSONL and re-copy Chroma from source before filtering",
    )
    args = p.parse_args()

    if args.top10 and args.strategies:
        raise SystemExit("Pass only one of --top10 or --strategies")
    if args.top10:
        ids = list(distinct_chunk_strategies(SELECTED_PAIRS_EVAL1))
    elif args.strategies:
        ids = list(args.strategies)
    else:
        raise SystemExit("Pass --top10 or --strategies ID [ID ...]")

    for sid in ids:
        if sid not in CHUNK_STRATEGY_IDS:
            raise SystemExit(f"Unknown strategy: {sid}. Valid: {CHUNK_STRATEGY_IDS}")

    train_path = Path(DEDUP_TRAIN_JSONL).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(f"Missing {train_path}")

    celex_order = _load_dedup_celex_ordered(train_path)
    keep_celex = set(celex_order)
    train_sha256 = _sha256_file(train_path)

    per_strategy: list[dict[str, int | str]] = []
    for sid in tqdm(ids, desc="Dedup chunk strategies", unit="strategy"):
        try:
            stats = build_one_dedup(sid, keep_celex, force=args.force)
            per_strategy.append(stats)
        finally:
            release_chroma_process_cache()

    manifest = {
        "celex_ids": celex_order,
        "n_docs": len(celex_order),
        "source_train_jsonl": str(train_path),
        "source_train_sha256": train_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "strategies": ids,
        "per_strategy": per_strategy,
        "method": "filtered_existing_chunks_and_chroma_no_reembedding",
    }
    DEDUP_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(DEDUP_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Wrote {DEDUP_MANIFEST} ({len(celex_order)} docs, {len(ids)} strategies)")


if __name__ == "__main__":
    main()
