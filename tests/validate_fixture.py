#!/usr/bin/env python3
"""Verify tests/fixture/Data has a complete 10-doc smoke corpus (small Chroma, not full-scale)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixture" / "Data"
STRATEGIES = (
    "len_500_o50",
    "len_1000_o100",
    "len_1500_o150",
    "len_2000_o0",
    "len_2000_o200",
    "para_nn_merge",
    "line_n_merge",
    "char_nn_only",
    "rec_nn_priority",
    "rec_legal_markers",
)
DEDUP_STRATEGIES = (
    "len_500_o50",
    "len_1000_o100",
    "len_1500_o150",
    "len_2000_o200",
    "rec_nn_priority",
)
MAX_CHROMA_MB = 15
N_TRAIN = 10
N_DEDUP = 8
N_GT = 2


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _chroma_mb(p: Path) -> float:
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return total / (1024 * 1024)


def main() -> None:
    if not FIXTURE.is_dir():
        _fail(f"missing {FIXTURE}; run: python tests/build_fixture.py")

    train = FIXTURE / "train.jsonl"
    if _lines(train) != N_TRAIN:
        _fail(f"{train}: expected {N_TRAIN} lines, got {_lines(train)}")

    dedup = FIXTURE / "train_dedup.jsonl"
    if _lines(dedup) != N_DEDUP:
        _fail(f"{dedup}: expected {N_DEDUP} lines, got {_lines(dedup)}")

    for name, n in (
        ("ground_truth.jsonl", N_GT),
        ("ground_truth_dedup_top10_100.jsonl", N_GT),
    ):
        p = FIXTURE / name
        if not p.is_file() or _lines(p) < n:
            _fail(f"{p}: need at least {n} rows")

    for sid in STRATEGIES:
        chunks = FIXTURE / f"chunks_{sid}.jsonl"
        chroma = FIXTURE / f"chroma_chunk_{sid}"
        if not chunks.is_file():
            _fail(f"missing {chunks}")
        if not (chroma / "chroma.sqlite3").is_file():
            _fail(f"missing {chroma}/chroma.sqlite3")
        mb = _chroma_mb(chroma)
        if mb > MAX_CHROMA_MB:
            _fail(
                f"{chroma} is {mb:.1f} MB (>{MAX_CHROMA_MB} MB). "
                "Looks like a full-corpus copy; rebuild with tests/build_fixture.py"
            )

    for sid in DEDUP_STRATEGIES:
        dc = FIXTURE / f"chunks_dedup_{sid}.jsonl"
        dch = FIXTURE / f"chroma_chunk_dedup_{sid}"
        if not dc.is_file() or not (dch / "chroma.sqlite3").is_file():
            _fail(f"missing dedup index for {sid}")

    with open(FIXTURE / "eval_corpus_manifest.json", encoding="utf-8") as f:
        m = json.load(f)
    if m.get("n_docs") != N_TRAIN:
        _fail(f"manifest n_docs={m.get('n_docs')}, expected {N_TRAIN}")

    checkpoints = list(FIXTURE.rglob("*.checkpoint.json"))
    if checkpoints:
        _fail(
            f"found {len(checkpoints)} .checkpoint.json files under fixture; "
            "remove them before commit (build_fixture.py cleans these)"
        )

    total_mb = sum(f.stat().st_size for f in FIXTURE.rglob("*") if f.is_file()) / (
        1024 * 1024
    )
    print(f"OK: fixture valid ({total_mb:.1f} MB under {FIXTURE})")
    print(f"  train={N_TRAIN} dedup={N_DEDUP} strategies={len(STRATEGIES)} chroma ~1-3 MB each")


if __name__ == "__main__":
    main()
