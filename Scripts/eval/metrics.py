"""Text normalization and retrieval hit / rank metrics (plan Part E)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.documents import Document


def normalize_text(s: str) -> str:
    """Collapse whitespace; lowercase for robust substring match."""
    s = s.lower()
    s = re.sub(r"\s+", " ", s.strip())
    return s


def first_hit_rank(
    docs: list[Document],
    reference_celex: str,
    gold_snippet: str,
) -> int | None:
    """
    Return 1-based rank of first chunk matching celex + gold snippet, or None if miss.
    """
    g = normalize_text(gold_snippet)
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        if str(meta.get("celex_id", "")) != str(reference_celex):
            continue
        if g and g in normalize_text(doc.page_content):
            return i
    return None


def aggregate_ranks(ranks: list[int | None]) -> dict[str, Any]:
    """Hit rate, MRR, and rank bucket counts (1..10 + miss)."""
    n = len(ranks)
    hits = [r for r in ranks if r is not None]
    hit_rate = len(hits) / n if n else 0.0
    mrr = sum(1.0 / r for r in hits) / n if n else 0.0
    buckets = {str(i): 0 for i in range(1, 11)}
    buckets["miss"] = 0
    for r in ranks:
        if r is None:
            buckets["miss"] += 1
        elif 1 <= r <= 10:
            buckets[str(r)] += 1
        else:
            buckets["miss"] += 1
    return {
        "n": n,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "buckets": buckets,
    }


def hit_at_k(ranks: list[int | None], k: int) -> float:
    """Fraction of queries with a hit at rank <= k (misses and ranks > k count as miss)."""
    if not ranks:
        return 0.0
    ok = sum(1 for r in ranks if r is not None and 1 <= r <= k)
    return ok / len(ranks)


def aggregate_ranks_topk(ranks: list[int | None], max_rank: int = 20) -> dict[str, Any]:
    """Hit rate, MRR, rank bucket counts (1..max_rank + miss), and Hit@3/5/10 when max_rank >= 10."""
    n = len(ranks)
    hits = [r for r in ranks if r is not None]
    hit_rate = len(hits) / n if n else 0.0
    mrr = sum(1.0 / r for r in hits) / n if n else 0.0
    buckets = {str(i): 0 for i in range(1, max_rank + 1)}
    buckets["miss"] = 0
    for r in ranks:
        if r is None:
            buckets["miss"] += 1
        elif 1 <= r <= max_rank:
            buckets[str(r)] += 1
        else:
            buckets["miss"] += 1
    out: dict[str, Any] = {
        "n": n,
        "hit_rate": hit_rate,
        "mrr": mrr,
        "buckets": buckets,
    }
    # Align with RAG context sizes (plan: report Hit@3, Hit@5, Hit@10, Hit@max_rank)
    for k in (3, 5, 10):
        if max_rank >= k:
            out[f"hit_at_{k}"] = hit_at_k(ranks, k)
    if max_rank >= 20:
        out["hit_at_20"] = hit_at_k(ranks, 20)
    elif max_rank > 0:
        out[f"hit_at_{max_rank}"] = hit_at_k(ranks, max_rank)
    return out
