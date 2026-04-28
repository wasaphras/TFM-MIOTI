"""Per-strategy neighbor index: CELEX-ordered chunk_uid lists for +/-k expansion."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from langchain_core.documents import Document
from tqdm import tqdm

from ... import config
from ...chunking_strategies import CHUNK_STRATEGY_IDS
from ..retrieval_strategies import dedupe_preserve_order
from ._shared import chunks_jsonl_path, neighbor_index_path


@dataclass
class NeighborIndex:
    celex_to_uids: dict[str, list[str]]
    uid_to_pos: dict[str, tuple[str, int]]

    def locate(self, chunk_uid: str) -> tuple[str, int] | None:
        if not chunk_uid:
            return None
        return self.uid_to_pos.get(str(chunk_uid))

    def uid_at(self, celex: str, pos: int) -> str | None:
        lst = self.celex_to_uids.get(celex)
        if not lst or pos < 0 or pos >= len(lst):
            return None
        return lst[pos]


def build_neighbor_index(strategy_id: str) -> NeighborIndex:
    path = chunks_jsonl_path(strategy_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    celex_to_uids: dict[str, list[str]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"neighbor_index {strategy_id}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            celex = str(meta.get("celex_id", "") or "")
            uid = str(meta.get("chunk_uid", "") or "")
            if not celex or not uid:
                continue
            lst = celex_to_uids[celex]
            if not lst or lst[-1] != uid:
                lst.append(uid)
    uid_to_pos: dict[str, tuple[str, int]] = {}
    for celex, uids in celex_to_uids.items():
        for i, u in enumerate(uids):
            uid_to_pos[u] = (celex, i)
    return NeighborIndex(
        celex_to_uids=dict(celex_to_uids),
        uid_to_pos=uid_to_pos,
    )


def save_neighbor_index(strategy_id: str, index: NeighborIndex) -> Path:
    out = neighbor_index_path(strategy_id)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(
            {
                "version": 1,
                "strategy_id": strategy_id,
                "celex_to_uids": index.celex_to_uids,
                "uid_to_pos": index.uid_to_pos,
            },
            f,
            protocol=4,
        )
    return out


def load_neighbor_index(strategy_id: str) -> NeighborIndex:
    out = neighbor_index_path(strategy_id)
    if not out.is_file():
        raise FileNotFoundError(
            f"Missing {out}. Run: python -m Scripts.eval.top10.neighbor_index --strategies {strategy_id}"
        )
    with open(out, "rb") as f:
        data = pickle.load(f)
    uid_to_pos: dict[str, tuple[str, int]] = {}
    for k, v in data["uid_to_pos"].items():
        if isinstance(v, (list, tuple)) and len(v) == 2:
            uid_to_pos[str(k)] = (str(v[0]), int(v[1]))
    return NeighborIndex(
        celex_to_uids={str(k): list(v) for k, v in data["celex_to_uids"].items()},
        uid_to_pos=uid_to_pos,
    )


def documents_by_uid(documents: list[Document]) -> dict[str, Document]:
    out: dict[str, Document] = {}
    for d in documents:
        uid = str((d.metadata or {}).get("chunk_uid") or "")
        if uid:
            out[uid] = d
    return out


def expand_with_neighbors(
    docs: list[Document],
    by_uid: dict[str, Document],
    index: NeighborIndex,
    offsets: tuple[int, ...] = (-2, -1, 1, 2),
    *,
    neighbor_seed_top: int | None = None,
) -> list[Document]:
    """
    Append neighbor chunks by CELEX order for each retrieved doc.

    If ``neighbor_seed_top`` is set, only the first N documents in ``docs`` get
    neighbor expansion (all ``docs`` are still kept in order first).
    If ``None``, every document gets inline neighbor expansion (legacy behavior).
    """
    out: list[Document] = []

    def _neighbors_for_doc(d: Document) -> list[Document]:
        extra: list[Document] = []
        uid = str((d.metadata or {}).get("chunk_uid") or "")
        loc = index.locate(uid)
        if not loc:
            return extra
        celex, pos = loc
        for off in offsets:
            nuid = index.uid_at(celex, pos + off)
            if nuid and nuid in by_uid:
                extra.append(by_uid[nuid])
        return extra

    if neighbor_seed_top is None:
        for d in docs:
            out.append(d)
            out.extend(_neighbors_for_doc(d))
    else:
        n = max(0, int(neighbor_seed_top))
        for d in docs:
            out.append(d)
        for d in docs[:n]:
            out.extend(_neighbors_for_doc(d))
    return dedupe_preserve_order(out)


def main() -> None:
    p = argparse.ArgumentParser(description="Build neighbor_index/*.pkl per chunk strategy.")
    p.add_argument(
        "--strategies",
        nargs="+",
        metavar="ID",
        help="Chunk strategy ids (default: all 10)",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Build for all CHUNK_STRATEGY_IDS",
    )
    args = p.parse_args()
    if args.all:
        ids = list(CHUNK_STRATEGY_IDS)
    elif args.strategies:
        ids = args.strategies
    else:
        raise SystemExit("Pass --strategies ID [ID ...] or --all")
    unknown = [i for i in ids if i not in CHUNK_STRATEGY_IDS]
    if unknown:
        raise SystemExit(f"Unknown strategy id(s): {unknown}")
    for sid in tqdm(ids, desc="neighbor_index", unit="strategy"):
        idx = build_neighbor_index(sid)
        path = save_neighbor_index(sid, idx)
        tqdm.write(f"Wrote {path} ({len(idx.uid_to_pos)} chunk uids)")
        del idx
        gc.collect()


if __name__ == "__main__":
    main()
