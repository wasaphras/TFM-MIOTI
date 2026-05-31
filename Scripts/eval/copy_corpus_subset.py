"""Filter chunks JSONL + Chroma from a source Data dir into the active config.DATA_DIR."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import chromadb
from tqdm import tqdm

from .. import config
from ..chunking_strategies import CHUNK_STRATEGY_IDS, chroma_persist_dir
from ..embeddings_chromadb import chroma_collection_count, release_chroma_process_cache
from .corpus_layout import CorpusLayout, STANDARD


def _count_jsonl(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def filter_chunks_jsonl(
    *,
    strategy_id: str,
    keep_celex: set[str],
    src_chunks: Path,
    dst_chunks: Path,
) -> tuple[int, int]:
    kept = dropped = 0
    dst_chunks.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst_chunks.with_suffix(dst_chunks.suffix + ".tmp")
    with open(src_chunks, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in tqdm(fin, desc=f"filter {dst_chunks.name}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            c = str((rec.get("metadata") or {}).get("celex_id") or "").strip()
            if c in keep_celex:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                dropped += 1
    tmp.replace(dst_chunks)
    return kept, dropped


def _delete_chroma_not_in_celex(persist: Path, keep_celex: set[str]) -> int:
    client = chromadb.PersistentClient(path=str(persist))
    coll = client.get_collection(name=config.CHROMA_COLLECTION)
    to_delete: list[str] = []
    offset = 0
    page_size = 5000
    while True:
        data = coll.get(include=["metadatas"], limit=page_size, offset=offset)
        ids = data.get("ids") or []
        metas = data.get("metadatas") or []
        if not ids:
            break
        for doc_id, meta in zip(ids, metas):
            c = str((meta or {}).get("celex_id") or "").strip()
            if c not in keep_celex:
                to_delete.append(str(doc_id))
        offset += len(ids)
        if len(ids) < page_size:
            break
    batch_size = 1000
    for i in range(0, len(to_delete), batch_size):
        batch = to_delete[i : i + batch_size]
        if batch:
            coll.delete(ids=batch)
    return len(to_delete)


def copy_strategy_subset(
    strategy_id: str,
    keep_celex: set[str],
    *,
    src_root: Path,
    layout: CorpusLayout = STANDARD,
    force: bool = False,
) -> None:
    src_chunks = src_root / f"{layout.chunks_prefix}{strategy_id}.jsonl"
    src_persist = src_root / f"{layout.chroma_prefix}{strategy_id}"
    dst_chunks = layout.chunks_jsonl_path(strategy_id)
    dst_persist = layout.chroma_persist_dir(strategy_id)

    if not src_chunks.is_file():
        raise FileNotFoundError(src_chunks)
    if not (src_persist / "chroma.sqlite3").is_file():
        raise FileNotFoundError(src_persist)

    src_n = _count_jsonl(src_chunks)
    src_chroma = chroma_collection_count(src_persist, config.CHROMA_COLLECTION)
    if src_chroma is None or src_chroma != src_n:
        raise SystemExit(
            f"{strategy_id}: source index incomplete ({src_chroma} vs {src_n} jsonl lines)"
        )

    if (
        not force
        and dst_chunks.is_file()
        and (dst_persist / "chroma.sqlite3").is_file()
    ):
        dst_n = _count_jsonl(dst_chunks)
        dst_chroma = chroma_collection_count(dst_persist, config.CHROMA_COLLECTION)
        if dst_n > 0 and dst_chroma == dst_n:
            print(f"Skip {strategy_id}: subset already present ({dst_n} chunks)")
            return

    kept, dropped = filter_chunks_jsonl(
        strategy_id=strategy_id,
        keep_celex=keep_celex,
        src_chunks=src_chunks,
        dst_chunks=dst_chunks,
    )
    print(f"{strategy_id}: {kept} kept, {dropped} dropped")

    if dst_persist.exists():
        shutil.rmtree(dst_persist)
    shutil.copytree(src_persist, dst_persist)
    deleted = _delete_chroma_not_in_celex(dst_persist, keep_celex)
    print(f"{strategy_id}: removed {deleted} Chroma rows outside subset")
    release_chroma_process_cache()

    final = chroma_collection_count(dst_persist, config.CHROMA_COLLECTION)
    if final != kept:
        raise SystemExit(f"{strategy_id}: chroma {final} != jsonl {kept}")


def main() -> None:
    p = argparse.ArgumentParser(description="Copy a CELEX subset from --src-root into DATA_DIR.")
    p.add_argument("--src-root", type=Path, required=True)
    p.add_argument("--celex-file", type=Path, required=True, help="JSON file with celex_ids list")
    p.add_argument("--strategies", nargs="+", default=list(CHUNK_STRATEGY_IDS))
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    with open(args.celex_file, encoding="utf-8") as f:
        keep = {str(c) for c in json.load(f)["celex_ids"]}
    for sid in args.strategies:
        if sid not in CHUNK_STRATEGY_IDS:
            raise SystemExit(f"Unknown strategy {sid}")
        copy_strategy_subset(sid, keep, src_root=args.src_root, force=args.force)


if __name__ == "__main__":
    main()
