"""Selected (chunk_strategy, retriever) pairs from the user's image."""

from __future__ import annotations

import hashlib

from ..retrieval_strategies import RERANK_SUFFIX

# Eval 1: pair 10 is base `hyb_interleave` (no cross-encoder). Evals 2–4 upgrade it.
SELECTED_PAIRS_EVAL1: tuple[tuple[str, str], ...] = (
    ("len_1000_o100", "hyb_rrf_k60_ce_r50"),
    ("len_1000_o100", "hyb_rrf_k30_ce_r50"),
    ("len_1000_o100", "hyb_rrf_fetch40_ce_r50"),
    ("len_1000_o100", "hyb_weighted_norm_ce_r50"),
    ("len_1000_o100", "hyb_interleave_ce_r50"),
    ("len_1500_o150", "hyb_rrf_k60_ce_r50"),
    ("len_1500_o150", "hyb_rrf_k30_ce_r50"),
    ("len_500_o50", "hyb_fill_dense_then_bm25_ce_r50"),
    ("rec_nn_priority", "hyb_fill_dense_then_bm25_ce_r50"),
    ("len_2000_o200", "hyb_interleave"),
)


def cell_key(chunk_strategy: str, retriever: str) -> str:
    return f"{chunk_strategy}::{retriever}"


def pairs_fingerprint(pairs: tuple[tuple[str, str], ...]) -> str:
    raw = "\n".join(f"{c}::{r}" for c, r in pairs).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def retriever_for_eval234(chunk_strategy: str, retriever: str) -> str:
    """Force cross-encoder suffix for evals 2–4 (pair 10 becomes hyb_interleave_ce_r50)."""
    if retriever.endswith(RERANK_SUFFIX):
        return retriever
    return f"{retriever}{RERANK_SUFFIX}"


def distinct_chunk_strategies(pairs: tuple[tuple[str, str], ...]) -> tuple[str, ...]:
    seen: list[str] = []
    for c, _ in pairs:
        if c not in seen:
            seen.append(c)
    return tuple(seen)
