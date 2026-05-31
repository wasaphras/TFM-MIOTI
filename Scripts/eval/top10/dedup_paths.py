"""Dedup corpus paths (re-exported from corpus_layout.DEDUP)."""

from __future__ import annotations

from pathlib import Path

from ..corpus_layout import DEDUP

DEDUP_TRAIN_JSONL = DEDUP.train_jsonl
DEDUP_MANIFEST = DEDUP.manifest
DEDUP_GROUND_TRUTH = DEDUP.default_ground_truth
DEDUP_EVAL_ROOT = DEDUP.eval_top10_root
DEDUP_NEIGHBOR_INDEX_DIR = DEDUP.neighbor_index_dir


def chunks_jsonl_path_dedup(strategy_id: str) -> Path:
    return DEDUP.chunks_jsonl_path(strategy_id)


def chroma_persist_dir_dedup(strategy_id: str) -> Path:
    return DEDUP.chroma_persist_dir(strategy_id)


def neighbor_index_path_dedup(strategy_id: str) -> Path:
    return DEDUP.neighbor_index_path(strategy_id)
