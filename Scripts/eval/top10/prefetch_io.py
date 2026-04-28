"""Disk cache for two-phase top10 eval: embedding retrieval first, CE rerank later."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from ._shared import atomic_write_json

PREFETCH_BUNDLE_VERSION = 1
PREFETCH_META_NAME = "prefetch_meta.json"


def prefetch_query_path(root: Path, eval_id: str, cid: str, rid: str, qi: int) -> Path:
    return Path(root) / eval_id / cid / rid / f"{qi:06d}.json"


def count_prefetched_queries(root: Path, eval_id: str, pairs: tuple[tuple[str, str], ...], nq: int) -> int:
    n = 0
    for cid, rid in pairs:
        d = Path(root) / eval_id / cid / rid
        if not d.is_dir():
            continue
        for qi in range(nq):
            if (d / f"{qi:06d}.json").is_file():
                n += 1
    return n


def document_to_record(d: Document) -> dict[str, Any]:
    return {
        "page_content": d.page_content or "",
        "metadata": dict(d.metadata or {}),
    }


def record_to_document(rec: dict[str, Any]) -> Document:
    return Document(
        page_content=str(rec.get("page_content") or ""),
        metadata=dict(rec.get("metadata") or {}),
    )


def save_prefetch_payload(path: Path, payload: dict[str, Any]) -> None:
    out = dict(payload)
    out["prefetch_record_version"] = PREFETCH_BUNDLE_VERSION
    atomic_write_json(path, out)


def load_prefetch_payload(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_prefetch_bundle_meta(path: Path, meta: dict[str, Any]) -> None:
    atomic_write_json(
        path,
        {"prefetch_bundle_version": PREFETCH_BUNDLE_VERSION, "meta": meta},
    )


def load_prefetch_bundle_meta(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prefetch_meta_matches_disk(current: dict[str, Any], prefetch_meta_path: Path) -> None:
    if not prefetch_meta_path.is_file():
        raise SystemExit(
            f"Missing {prefetch_meta_path}. Run --prefetch-write for this eval first."
        )
    loaded = load_prefetch_bundle_meta(prefetch_meta_path)
    prev = loaded.get("meta") or {}
    keys = (
        "version",
        "eval_id",
        "ground_truth",
        "ground_truth_sha256",
        "manifest",
        "manifest_sha256",
        "pairs_fingerprint",
        "limit_queries",
        "final_k",
        "candidate_k",
    )
    for k in keys:
        if prev.get(k) != current.get(k):
            raise SystemExit(
                f"Prefetch bundle at {prefetch_meta_path} was built with different settings.\n"
                f"Mismatch on {k!r}: disk={prev.get(k)!r} current={current.get(k)!r}\n"
                "Re-run --prefetch-write with --no-resume-prefetch to rebuild, or align CLI args."
            )
    for k in ("neighbor_offsets", "multiquery_candidate_k"):
        if k in current and prev.get(k) != current.get(k):
            raise SystemExit(
                f"Prefetch meta mismatch on {k!r}: disk={prev.get(k)!r} current={current.get(k)!r}"
            )
