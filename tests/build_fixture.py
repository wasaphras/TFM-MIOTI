#!/usr/bin/env python3
"""Build tests/fixture/Data: 10-doc corpus, indices, 2 GT rows per track."""

from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixture" / "Data"
SRC_DATA = REPO / "Data"
SEED = 42
N_DOCS = 10
N_DEDUP = 8
N_GT = 2


def _run(cmd: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO, env=env, check=True)


def _count_jsonl(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _sample_train() -> tuple[list[str], list[str]]:
    rows: list[dict] = []
    with open(SRC_DATA / "train.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = random.Random(SEED)
    picked = rng.sample(rows, N_DOCS)
    FIXTURE.mkdir(parents=True, exist_ok=True)
    celex_all: list[str] = []
    with open(FIXTURE / "train.jsonl", "w", encoding="utf-8") as f:
        for rec in picked:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            celex_all.append(str(rec["celex_id"]))
    with open(FIXTURE / "train_dedup.jsonl", "w", encoding="utf-8") as f:
        for rec in picked[:N_DEDUP]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return celex_all, celex_all[:N_DEDUP]


def _try_ground_truth(env: dict[str, str], module: str, out: Path) -> bool:
    py = sys.executable
    try:
        _run([py, "-m", module, "--n", str(N_GT), "--no-resume"], env)
        return out.is_file() and _count_jsonl(out) >= N_GT
    except subprocess.CalledProcessError:
        return False


def _synthetic_ground_truth(
    *,
    celex_ids: list[str],
    chunks_path: Path,
    out: Path,
    dedup: bool,
) -> None:
    """Write n=len(celex_ids) GT rows from chunk text (no LLM)."""
    by_celex: dict[str, list[dict]] = {c: [] for c in celex_ids}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            meta = rec.get("metadata") or {}
            c = str(meta.get("celex_id") or "")
            if c in by_celex:
                by_celex[c].append(rec)
    rows: list[dict] = []
    for i, celex in enumerate(celex_ids[:N_GT]):
        recs = by_celex.get(celex) or []
        if not recs:
            raise SystemExit(f"No chunks for celex {celex} in {chunks_path}")
        mid = recs[len(recs) // 2]
        text = str(mid.get("page_content") or "")
        snippet = text[:280] if len(text) > 280 else text
        row: dict = {
            "id": f"gt_fixture_{i:05d}",
            "question": f"What does CELEX {celex} state regarding the matter described in the excerpt?",
            "reference": celex,
            "gold_snippet": snippet,
            "source_len_chars": len(text),
        }
        if dedup:
            row["answer"] = snippet[:120]
            row["source_chunk_positions"] = {"len_500_o50": len(recs) // 2}
            row["question_validation"] = {
                "supported": True,
                "reason": "fixture synthetic row",
            }
        rows.append(row)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _borrow_ground_truth(src: Path, dst: Path, allowed: set[str], n: int) -> None:
    rows: list[dict] = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if str(rec.get("reference")) in allowed:
                rows.append(rec)
            if len(rows) >= n:
                break
    if len(rows) < n:
        raise SystemExit(f"Only {len(rows)} GT rows in {src} match fixture CELEX ids")
    with open(dst, "w", encoding="utf-8") as f:
        for rec in rows[:n]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    if not (SRC_DATA / "train.jsonl").is_file():
        raise SystemExit(f"Missing {SRC_DATA / 'train.jsonl'}")

    if FIXTURE.exists():
        shutil.rmtree(FIXTURE)

    celex_all, celex_dedup = _sample_train()

    env = {
        **dict(__import__("os").environ),
        "TFM_DATA_DIR": str(FIXTURE),
        "TFM_CATEGORIES_JSON": str(SRC_DATA / "categories.json"),
    }
    py = sys.executable

    _run(
        [py, "-m", "Scripts.eval.build_chunk_indices", "--all", "--limit", str(N_DOCS)],
        env,
    )

    gt_std = FIXTURE / "ground_truth.jsonl"
    if not _try_ground_truth(env, "Scripts.eval.ground_truth_generate", gt_std):
        print("ground_truth_generate unavailable; writing synthetic GT", flush=True)
        _synthetic_ground_truth(
            celex_ids=celex_all[:N_GT],
            chunks_path=FIXTURE / "chunks_len_500_o50.jsonl",
            out=gt_std,
            dedup=False,
        )

    _run([py, "-m", "Scripts.eval.build_chunk_indices_dedup", "--top10", "--force"], env)

    gt_dedup = FIXTURE / "ground_truth_dedup_top10_100.jsonl"
    if not _try_ground_truth(env, "Scripts.eval.ground_truth_generate_dedup", gt_dedup):
        print("ground_truth_generate_dedup unavailable; writing synthetic GT", flush=True)
        _synthetic_ground_truth(
            celex_ids=celex_dedup[:N_GT],
            chunks_path=FIXTURE / "chunks_dedup_len_500_o50.jsonl",
            out=gt_dedup,
            dedup=True,
        )

    _run(
        [
            py,
            "-m",
            "Scripts.eval.top10.neighbor_index",
            "--strategies",
            "len_500_o50",
            "len_1000_o100",
            "len_1500_o150",
            "len_2000_o200",
            "rec_nn_priority",
        ],
        env,
    )
    _run([py, "-m", "Scripts.eval.top10.neighbor_index_dedup", "--top10"], env)

    for ckpt in FIXTURE.rglob("*.checkpoint.json"):
        ckpt.unlink()
    for name in ("eval_grid_checkpoint.json",):
        p = FIXTURE / "eval" / name
        if p.is_file():
            p.unlink()

    print(f"Fixture ready under {FIXTURE}")
    print("Verify: python tests/validate_fixture.py")


if __name__ == "__main__":
    main()
