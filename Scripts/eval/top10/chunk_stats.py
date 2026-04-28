"""Median chunk character length per strategy (for LLM target word counts)."""

from __future__ import annotations

import gc
import json
import statistics
from pathlib import Path

from tqdm import tqdm

from ...chunking_strategies import CHUNK_STRATEGY_IDS
from ._shared import atomic_write_json, chunk_stats_path, chunks_jsonl_path


def compute_median_chars(strategy_id: str) -> float:
    path = chunks_jsonl_path(strategy_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}")
    lengths: list[int] = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"chunk_stats {strategy_id}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            text = rec.get("page_content") or ""
            lengths.append(len(str(text)))
    if not lengths:
        raise ValueError(f"No chunks in {path}")
    return float(statistics.median(lengths))


def ensure_chunk_stats(
    strategy_ids: tuple[str, ...],
    *,
    force_rebuild: bool = False,
) -> dict[str, dict]:
    """Load or build Data/chunk_stats.json for the given strategies."""
    out_path = chunk_stats_path()
    data: dict[str, dict] = {}
    if out_path.is_file():
        with open(out_path, encoding="utf-8") as f:
            data = json.load(f)
    to_compute = [sid for sid in strategy_ids if sid not in data or force_rebuild]
    strat_iter = (
        tqdm(to_compute, desc="chunk_stats", unit="strategy")
        if to_compute
        else to_compute
    )
    for sid in strat_iter:
        med = compute_median_chars(sid)
        target_words = int(max(60, min(250, med / 6.0)))
        data[sid] = {
            "median_page_content_chars": med,
            "target_words": target_words,
        }
        gc.collect()
    updated = bool(to_compute)
    if updated:
        atomic_write_json(out_path, data)
        print(f"Wrote {out_path}")
    return data


def target_words_for_strategy(strategy_id: str) -> int:
    stats = ensure_chunk_stats((strategy_id,))
    return int(stats[strategy_id]["target_words"])


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Build Data/chunk_stats.json")
    p.add_argument("--strategies", nargs="+", required=True)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    unknown = [s for s in args.strategies if s not in CHUNK_STRATEGY_IDS]
    if unknown:
        raise SystemExit(f"Unknown: {unknown}")
    ensure_chunk_stats(tuple(args.strategies), force_rebuild=args.force)


if __name__ == "__main__":
    main()
