"""Parametric retriever: final_k=20, candidate_k configurable, dedupe before rerank."""

from __future__ import annotations

from langchain_core.documents import Document

from ... import config
from ..rerank_cross_encoder import rerank_documents, rerank_documents_with_scores
from ..retrieval_strategies import (
    RERANK_SUFFIX,
    RetrievalContext,
    _RETRIEVERS,
    _pad_to_k,
    base_retriever_id,
    dedupe_preserve_order,
    doc_key,
)
def run_retriever_k(
    name: str,
    ctx: RetrievalContext,
    query: str,
    *,
    final_k: int = 20,
    candidate_k: int | None = None,
    attach_rerank_scores: bool = False,
) -> list[Document]:
    """
    Run base retriever with ``candidate_k`` breadth, dedupe, optional CE rerank to ``final_k``.
    """
    base = base_retriever_id(name)
    if base not in _RETRIEVERS:
        raise ValueError(f"Unknown retriever base: {base}")
    cand = candidate_k if candidate_k is not None else max(
        int(config.RETRIEVAL_CANDIDATE_K), final_k * 5
    )
    raw = _RETRIEVERS[base](ctx, query, cand)
    unique = dedupe_preserve_order(raw)
    if name.endswith(RERANK_SUFFIX):
        if not unique:
            docs: list[Document] = []
        elif attach_rerank_scores:
            docs = [d for d, _ in rerank_documents_with_scores(query, unique, top_n=final_k)]
        else:
            docs = rerank_documents(query, unique, top_n=final_k)
        docs = _pad_to_k(docs, ctx, query, final_k)
    else:
        docs = unique[:final_k]
        docs = _pad_to_k(docs, ctx, query, final_k)
    if len(docs) < final_k:
        raise RuntimeError(
            f"Retriever {name} could not produce {final_k} unique docs "
            f"(corpus too small?). Got {len(docs)}."
        )
    return docs[:final_k]


def run_base_retriever_k(
    base_name: str,
    ctx: RetrievalContext,
    query: str,
    *,
    candidate_k: int = 100,
) -> list[Document]:
    """Base retriever only (no rerank), breadth ``candidate_k``, deduped."""
    if base_name not in _RETRIEVERS:
        raise ValueError(f"Unknown retriever: {base_name}")
    raw = _RETRIEVERS[base_name](ctx, query, candidate_k)
    return dedupe_preserve_order(raw)


def pad_from_candidate_pool(
    ranked: list[Document], pool: list[Document], k: int
) -> list[Document]:
    """
    Extend ``ranked`` up to ``k`` using retrieval order in ``pool`` (no second embedding pass).
    Slightly different from :func:`_pad_to_k` when the pool is short; matches intent of two-phase eval.
    """
    if len(ranked) >= k:
        return ranked[:k]
    seen = {doc_key(d) for d in ranked}
    out = list(ranked)
    for d in pool:
        if len(out) >= k:
            break
        key = doc_key(d)
        if key in seen:
            continue
        seen.add(key)
        out.append(d)
    return out[:k] if len(out) >= k else out


def finalize_from_candidates(
    rid: str,
    query: str,
    unique: list[Document],
    *,
    final_k: int = 20,
) -> list[Document]:
    """Cross-encode rerank + pad from the saved candidate pool (prefetch phase 2)."""
    unique = dedupe_preserve_order(unique)
    if rid.endswith(RERANK_SUFFIX):
        if not unique:
            docs: list[Document] = []
        else:
            docs = rerank_documents(query, unique, top_n=final_k)
        docs = pad_from_candidate_pool(docs, unique, final_k)
    else:
        docs = unique[:final_k]
        docs = pad_from_candidate_pool(docs, unique, final_k)
    if len(docs) < final_k:
        raise RuntimeError(
            f"Retriever {rid} could not produce {final_k} unique docs from prefetch pool "
            f"(corpus too small?). Got {len(docs)}."
        )
    return docs[:final_k]
