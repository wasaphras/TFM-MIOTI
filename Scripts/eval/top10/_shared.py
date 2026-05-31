"""Shared helpers: fingerprints, atomic JSON, ground-truth load, manifest validation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ... import config


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def load_ground_truth(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Load {path.name}", unit=" lines"):
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def validate_gt_against_manifest(gt: list[dict], manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run build_chunk_indices first."
        )
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    allowed = {str(c) for c in manifest.get("celex_ids", []) if c}
    refs = {str(r["reference"]) for r in gt if r.get("reference")}
    missing = sorted(refs - allowed)
    if missing:
        raise SystemExit(
            f"Ground truth references CELEX ids not in {manifest_path}:\n  {missing}\n"
        )


def chunks_jsonl_path(strategy_id: str) -> Path:
    from ..corpus_layout import STANDARD

    return STANDARD.chunks_jsonl_path(strategy_id)


def neighbor_index_path(strategy_id: str) -> Path:
    from ..corpus_layout import STANDARD

    return STANDARD.neighbor_index_path(strategy_id)


def chunk_stats_path() -> Path:
    return config.DATA_DIR / "chunk_stats.json"
