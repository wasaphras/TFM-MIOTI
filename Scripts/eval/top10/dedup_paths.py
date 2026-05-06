"""Paths for the dedup top-10 eval pipeline (train_dedup.jsonl + filtered indices)."""

from __future__ import annotations

from pathlib import Path

from ... import config

DEDUP_TRAIN_JSONL = config.DATA_DIR / "train_dedup.jsonl"
DEDUP_MANIFEST = config.DATA_DIR / "eval_corpus_manifest_dedup.json"
DEDUP_GROUND_TRUTH = config.DATA_DIR / "ground_truth_dedup_top10_100.jsonl"
DEDUP_EVAL_ROOT = config.DATA_DIR / "eval_top10_dedup"
DEDUP_NEIGHBOR_INDEX_DIR = config.DATA_DIR / "neighbor_index_dedup"


def chunks_jsonl_path_dedup(strategy_id: str) -> Path:
    return config.DATA_DIR / f"chunks_dedup_{strategy_id}.jsonl"


def chroma_persist_dir_dedup(strategy_id: str) -> Path:
    return config.DATA_DIR / f"chroma_chunk_dedup_{strategy_id}"


def neighbor_index_path_dedup(strategy_id: str) -> Path:
    return DEDUP_NEIGHBOR_INDEX_DIR / f"{strategy_id}.pkl"
