"""Build/load neighbor indices for the dedup chunk corpus."""

from __future__ import annotations

import argparse
import gc

from tqdm import tqdm

from ...chunking_strategies import CHUNK_STRATEGY_IDS
from ..corpus_layout import DEDUP
from .neighbor_index import (
    NeighborIndex,
    build_neighbor_index as _build,
    expand_with_neighbors,
    load_neighbor_index as _load,
    save_neighbor_index as _save,
)
from .pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies


def build_neighbor_index(strategy_id: str) -> NeighborIndex:
    return _build(strategy_id, layout=DEDUP)


def save_neighbor_index(strategy_id: str, index: NeighborIndex):
    return _save(strategy_id, index, layout=DEDUP)


def load_neighbor_index(strategy_id: str) -> NeighborIndex:
    return _load(strategy_id, layout=DEDUP)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build neighbor_index_dedup/*.pkl from chunks_dedup_*.jsonl."
    )
    p.add_argument(
        "--top10",
        action="store_true",
        help="Build for distinct chunk strategies in SELECTED_PAIRS_EVAL1",
    )
    p.add_argument("--strategies", nargs="+", metavar="ID", default=None)
    p.add_argument("--all", action="store_true", help="Build for all CHUNK_STRATEGY_IDS")
    args = p.parse_args()
    if args.all:
        ids = list(CHUNK_STRATEGY_IDS)
    elif args.top10:
        ids = list(distinct_chunk_strategies(SELECTED_PAIRS_EVAL1))
    elif args.strategies:
        ids = args.strategies
    else:
        raise SystemExit("Pass --top10, --strategies ID [ID ...], or --all")
    unknown = [i for i in ids if i not in CHUNK_STRATEGY_IDS]
    if unknown:
        raise SystemExit(f"Unknown strategy id(s): {unknown}")
    for sid in tqdm(ids, desc="neighbor_index_dedup", unit="strategy"):
        idx = build_neighbor_index(sid)
        path = save_neighbor_index(sid, idx)
        tqdm.write(f"Wrote {path} ({len(idx.uid_to_pos)} chunk uids)")
        del idx
        gc.collect()


if __name__ == "__main__":
    main()
