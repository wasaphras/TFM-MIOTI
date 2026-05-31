"""Standard vs dedup filesystem layout under config.DATA_DIR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import config


@dataclass(frozen=True)
class CorpusLayout:
    name: str
    chunks_prefix: str
    chroma_prefix: str
    manifest: Path
    default_ground_truth: Path
    eval_top10_root: Path
    neighbor_index_dir: Path
    train_jsonl: Path | None = None

    def chunks_jsonl_path(self, strategy_id: str) -> Path:
        return config.DATA_DIR / f"{self.chunks_prefix}{strategy_id}.jsonl"

    def chroma_persist_dir(self, strategy_id: str) -> Path:
        return config.DATA_DIR / f"{self.chroma_prefix}{strategy_id}"

    def neighbor_index_path(self, strategy_id: str) -> Path:
        return self.neighbor_index_dir / f"{strategy_id}.pkl"

    def chroma_persist_dir_str(self, strategy_id: str) -> str:
        return str(self.chroma_persist_dir(strategy_id))


STANDARD = CorpusLayout(
    name="standard",
    chunks_prefix="chunks_",
    chroma_prefix="chroma_chunk_",
    manifest=config.EVAL_CORPUS_MANIFEST,
    default_ground_truth=config.GROUND_TRUTH_JSONL,
    eval_top10_root=config.DATA_DIR / "eval_top10",
    neighbor_index_dir=config.DATA_DIR / "neighbor_index",
    train_jsonl=config.TRAIN_JSONL,
)

DEDUP = CorpusLayout(
    name="dedup",
    chunks_prefix="chunks_dedup_",
    chroma_prefix="chroma_chunk_dedup_",
    manifest=config.DATA_DIR / "eval_corpus_manifest_dedup.json",
    default_ground_truth=config.DATA_DIR / "ground_truth_dedup_top10_100.jsonl",
    eval_top10_root=config.DATA_DIR / "eval_top10_dedup",
    neighbor_index_dir=config.DATA_DIR / "neighbor_index_dedup",
    train_jsonl=config.DATA_DIR / "train_dedup.jsonl",
)
