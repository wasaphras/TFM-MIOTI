"""
Retriever variants for eval: dense, BM25, hybrid. Final list length k=10.

There are **10 base** retriever IDs (`RETRIEVER_IDS_BASE`) plus the same bases
with the ``_ce_r50`` suffix (**20** total in ``RETRIEVER_IDS``): retrieve
``RETRIEVAL_CANDIDATE_K`` candidates, cross-encode rerank, keep top 10.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from .. import config
from .rerank_cross_encoder import rerank_documents

FINAL_K = 10

RETRIEVER_IDS_BASE: tuple[str, ...] = (
    "dense_sim_k10",
    "dense_mmr_k10",
    "bm25_k10",
    "hyb_rrf_k60",
    "hyb_rrf_k30",
    "hyb_rrf_fetch40",
    "hyb_weighted_norm",
    "hyb_weighted_dense_70",
    "hyb_interleave",
    "hyb_fill_dense_then_bm25",
)

RERANK_SUFFIX = "_ce_r50"

RETRIEVER_IDS: tuple[str, ...] = RETRIEVER_IDS_BASE + tuple(
    f"{r}{RERANK_SUFFIX}" for r in RETRIEVER_IDS_BASE
)

RERANK_RETRIEVER_IDS: frozenset[str] = frozenset(
    f"{r}{RERANK_SUFFIX}" for r in RETRIEVER_IDS_BASE
)

# Retrievers that never call bm25_top in their main path; eval can run them
# before loading chunks JSONL / building BM25 (large RAM savings).
DENSE_ONLY_RETRIEVER_IDS: frozenset[str] = frozenset(
    {"dense_sim_k10", "dense_mmr_k10", "dense_sim_k10_ce_r50", "dense_mmr_k10_ce_r50"}
)

# Bases that only need Chroma (no BM25 / full chunks jsonl).
DENSE_BASE_IDS: frozenset[str] = frozenset({"dense_sim_k10", "dense_mmr_k10"})


def is_rerank_retriever(rid: str) -> bool:
    return rid in RERANK_RETRIEVER_IDS


def base_retriever_id(rid: str) -> str:
    if rid.endswith(RERANK_SUFFIX):
        return rid[: -len(RERANK_SUFFIX)]
    return rid


def needs_dense_only_context_for_base(base: str) -> bool:
    return base in DENSE_BASE_IDS


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b\w+\b", text.lower())


def doc_key(d: Document) -> str:
    m = d.metadata or {}
    uid = m.get("chunk_uid")
    if uid:
        return str(uid)
    celex = str(m.get("celex_id", "") or "")
    prefix = re.sub(r"\s+", " ", d.page_content[:240].lower().strip())
    return f"{celex}|{prefix}"


def dedupe_preserve_order(docs: list[Document]) -> list[Document]:
    seen: set[str] = set()
    out: list[Document] = []
    for d in docs:
        k = doc_key(d)
        if k in seen:
            continue
        seen.add(k)
        out.append(d)
    return out


class RetrievalContext:
    """Chroma store + optional same ordered chunk list for BM25 (lazy-built)."""

    def __init__(
        self, vectorstore: Chroma, documents: list[Document] | None
    ) -> None:
        self.vectorstore = vectorstore
        self.documents = documents
        self._tok: list[list[str]] | None = None
        self._bm25: BM25Okapi | None = None
        self._chunk_uid_index: dict[str, Document] | None = None

    def chunk_uid_map(self) -> dict[str, Document]:
        """
        Map chunk_uid -> Document for the loaded corpus. Built once per context (O(n) in corpus size);
        do not rebuild on every query — that caused multi-GB RSS growth on million-chunk strategies.
        """
        if self.documents is None:
            return {}
        if self._chunk_uid_index is None:
            m: dict[str, Document] = {}
            for d in self.documents:
                uid = str((d.metadata or {}).get("chunk_uid") or "")
                if uid:
                    m[uid] = d
            self._chunk_uid_index = m
        return self._chunk_uid_index

    def _ensure_bm25(self) -> None:
        if self.documents is None:
            raise RuntimeError("BM25 requested but chunk documents were not loaded.")
        if self._bm25 is not None:
            return
        self._tok = [tokenize(d.page_content) for d in self.documents]
        self._bm25 = BM25Okapi(self._tok)

    def bm25_top(self, query: str, k: int) -> list[tuple[Document, float]]:
        self._ensure_bm25()
        assert self._bm25 is not None and self.documents is not None
        q = tokenize(query)
        scores = self._bm25.get_scores(q)
        ranked = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]
        return [(self.documents[i], float(scores[i])) for i in ranked]

    def dense_top(self, query: str, k: int) -> list[Document]:
        return self.vectorstore.similarity_search(query, k=k)

    def dense_top_scores(
        self, query: str, k: int
    ) -> list[tuple[Document, float]]:
        """Return (doc, L2_distance) pairs. Lower distance = more similar."""
        return self.vectorstore.similarity_search_with_score(query, k=k)


def _dense_sim(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return ctx.dense_top(query, out_k)


def _dense_mmr(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    fetch_k = max(out_k * 2, 40)
    return ctx.vectorstore.max_marginal_relevance_search(
        query, k=out_k, fetch_k=fetch_k, lambda_mult=0.5
    )


def _bm25(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return [d for d, _ in ctx.bm25_top(query, out_k)]


def _rrf(
    ctx: RetrievalContext,
    query: str,
    k_rrf: int,
    k_fetch: int,
    out_k: int,
) -> list[Document]:
    k_fetch = max(k_fetch, out_k)
    dense = ctx.dense_top(query, k_fetch)
    bm25 = [d for d, _ in ctx.bm25_top(query, k_fetch)]
    scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for r, d in enumerate(dense, start=1):
        k = doc_key(d)
        doc_by_key[k] = d
        scores[k] = scores.get(k, 0.0) + 1.0 / (k_rrf + r)
    for r, d in enumerate(bm25, start=1):
        k = doc_key(d)
        doc_by_key[k] = d
        scores[k] = scores.get(k, 0.0) + 1.0 / (k_rrf + r)

    ranked = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_by_key[k] for k in ranked[:out_k]]


def _hyb_rrf_k60(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return _rrf(ctx, query, k_rrf=60, k_fetch=20, out_k=out_k)


def _hyb_rrf_k30(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return _rrf(ctx, query, k_rrf=30, k_fetch=20, out_k=out_k)


def _hyb_rrf_fetch40(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return _rrf(ctx, query, k_rrf=60, k_fetch=40, out_k=out_k)


def _minmax_norm(vals: dict[str, float]) -> dict[str, float]:
    if not vals:
        return {}
    xs = [v for v in vals.values() if v > 0]
    if not xs:
        return {k: 0.0 for k in vals}
    lo, hi = min(xs), max(xs)
    if hi <= lo:
        return {k: (1.0 if v > 0 else 0.0) for k, v in vals.items()}
    return {k: (v - lo) / (hi - lo) if v > 0 else 0.0 for k, v in vals.items()}


def _weighted_fusion(
    ctx: RetrievalContext, query: str, w_dense: float, w_bm25: float, out_k: int
) -> list[Document]:
    fetch = max(20, out_k)
    pairs_d = ctx.dense_top_scores(query, fetch)
    pairs_b = ctx.bm25_top(query, fetch)

    dense_raw: dict[str, float] = {}
    bm25_raw: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for d, dist in pairs_d:
        k = doc_key(d)
        doc_by_key[k] = d
        dense_raw[k] = 1.0 / (1.0 + float(dist))
    for d, sc in pairs_b:
        k = doc_key(d)
        doc_by_key[k] = d
        bm25_raw[k] = float(sc)

    nd = _minmax_norm(dense_raw)
    nb = _minmax_norm(bm25_raw)
    keys = set(nd) | set(nb)
    combined = {
        k: w_dense * nd.get(k, 0.0) + w_bm25 * nb.get(k, 0.0) for k in keys
    }
    ranked = sorted(combined.keys(), key=lambda x: combined[x], reverse=True)
    out: list[Document] = []
    seen: set[str] = set()
    for k in ranked:
        if k in seen:
            continue
        seen.add(k)
        out.append(doc_by_key[k])
        if len(out) >= out_k:
            break
    return out


def _hyb_weighted_norm(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return _weighted_fusion(ctx, query, 0.5, 0.5, out_k)


def _hyb_weighted_dense_70(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    return _weighted_fusion(ctx, query, 0.7, 0.3, out_k)


def _hyb_interleave(ctx: RetrievalContext, query: str, out_k: int) -> list[Document]:
    fetch = max(20, out_k)
    dense = ctx.dense_top(query, fetch)
    bm25 = [d for d, _ in ctx.bm25_top(query, fetch)]
    out: list[Document] = []
    seen: set[str] = set()
    for t in range(max(len(dense), len(bm25))):
        for lst in (dense, bm25):
            if t < len(lst):
                d = lst[t]
                k = doc_key(d)
                if k not in seen:
                    seen.add(k)
                    out.append(d)
                    if len(out) >= out_k:
                        return out
    return out


def _hyb_fill_dense_then_bm25(
    ctx: RetrievalContext, query: str, out_k: int
) -> list[Document]:
    dense_a_n = max(10, out_k)
    dense_a = ctx.dense_top(query, dense_a_n)
    bm_b = [d for d, _ in ctx.bm25_top(query, max(20, out_k))]
    dense_c = ctx.dense_top(query, max(20, out_k))
    out: list[Document] = []
    seen: set[str] = set()

    first_block = max(5, out_k // 2)
    first_block = min(first_block, len(dense_a))

    for d in dense_a[:first_block]:
        k = doc_key(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    for d in bm_b:
        if len(out) >= out_k:
            break
        k = doc_key(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    for d in dense_c:
        if len(out) >= out_k:
            break
        k = doc_key(d)
        if k not in seen:
            seen.add(k)
            out.append(d)
    return out[:out_k]


_RETRIEVERS: dict[str, Callable[[RetrievalContext, str, int], list[Document]]] = {
    "dense_sim_k10": _dense_sim,
    "dense_mmr_k10": _dense_mmr,
    "bm25_k10": _bm25,
    "hyb_rrf_k60": _hyb_rrf_k60,
    "hyb_rrf_k30": _hyb_rrf_k30,
    "hyb_rrf_fetch40": _hyb_rrf_fetch40,
    "hyb_weighted_norm": _hyb_weighted_norm,
    "hyb_weighted_dense_70": _hyb_weighted_dense_70,
    "hyb_interleave": _hyb_interleave,
    "hyb_fill_dense_then_bm25": _hyb_fill_dense_then_bm25,
}


def _pad_to_k(
    docs: list[Document], ctx: RetrievalContext, query: str, k: int
) -> list[Document]:
    seen = {doc_key(d) for d in docs}
    out = list(docs)
    if len(out) >= k:
        return out[:k]
    for d in ctx.dense_top(query, 80):
        key = doc_key(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
        if len(out) >= k:
            return out
    if ctx.documents is not None:
        for d, _ in ctx.bm25_top(query, 80):
            key = doc_key(d)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
            if len(out) >= k:
                return out
    return out[:k] if len(out) >= k else out


def run_retriever(
    name: str, ctx: RetrievalContext, query: str
) -> list[Document]:
    candidate_k = config.RETRIEVAL_CANDIDATE_K
    if name in RERANK_RETRIEVER_IDS:
        base = name[: -len(RERANK_SUFFIX)]
        if base not in _RETRIEVERS:
            raise ValueError(f"Unknown rerank retriever base: {base}")
        raw = _RETRIEVERS[base](ctx, query, candidate_k)
        unique = dedupe_preserve_order(raw)
        if not unique:
            docs: list[Document] = []
        else:
            docs = rerank_documents(query, unique, top_n=FINAL_K)
        docs = _pad_to_k(docs, ctx, query, FINAL_K)
    elif name in _RETRIEVERS:
        docs = _RETRIEVERS[name](ctx, query, FINAL_K)
        docs = _pad_to_k(docs, ctx, query, FINAL_K)
    else:
        raise ValueError(f"Unknown retriever: {name}")
    if len(docs) < FINAL_K:
        raise RuntimeError(
            f"Retriever {name} could not produce {FINAL_K} unique docs "
            f"(corpus too small or empty?). Got {len(docs)}."
        )
    return docs[:FINAL_K]


def load_documents_from_chunks_jsonl(path: str | Path) -> list[Document]:
    path = Path(path)
    docs: list[Document] = []
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Load chunks {path.name}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            docs.append(
                Document(
                    page_content=rec["page_content"],
                    metadata=rec.get("metadata") or {},
                )
            )
    return docs
