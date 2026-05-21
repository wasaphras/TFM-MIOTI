"""
Production RAG chat: dedup corpus, eval-winning retriever, Ollama embed, Gemini answer.
"""

from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from . import config
from .embeddings_chromadb import load_vectorstore
from .eval.retrieval_strategies import RetrievalContext, load_documents_from_chunks_jsonl
from .eval.top10.dedup_paths import (
    DEDUP_TRAIN_JSONL,
    chroma_persist_dir_dedup,
    chunks_jsonl_path_dedup,
)
from .eval.rerank_cross_encoder import rerank_documents_with_scores
from .eval.retrieval_strategies import (
    RERANK_SUFFIX,
    _RETRIEVERS,
    base_retriever_id,
    dedupe_preserve_order,
)
from .rag_stream_utils import stream_chunk_to_text
from .retriever import _build_rag_prompt, format_context_chunks_text

CHUNK_STRATEGY = os.environ.get("RAG_CHUNK_STRATEGY", "len_500_o50")
RETRIEVER = os.environ.get("RAG_RETRIEVER", "hyb_fill_dense_then_bm25_ce_r50")
FINAL_K = int(os.environ.get("RAG_FINAL_K", "20"))
CANDIDATE_K = int(os.environ.get("RAG_CANDIDATE_K", "100"))
MIN_RERANK_SCORE = float(os.environ.get("RAG_MIN_RERANK_SCORE", "0.3"))
HISTORY_MAX_MESSAGES = int(os.environ.get("RAG_HISTORY_MAX_MESSAGES", "20"))
HISTORY_MSG_MAX_CHARS = int(os.environ.get("RAG_HISTORY_MSG_MAX_CHARS", "4000"))
SNIPPET_MAX_CHARS = 400
DOC_PREVIEW_MAX_CHARS = 800
CITATION_FALLBACK = os.environ.get("RAG_CITATION_FALLBACK", "all").strip().lower()

_CITATION_RE = re.compile(r"\[(\d+)\]")

_ctx: RetrievalContext | None = None
_gemini_model: Any = None
_celex_docs: dict[str, dict[str, Any]] = {}


@dataclass(frozen=True)
class ContextChunk:
    chunk_uid: str
    celex_id: str
    categories_en: str
    text: str


@dataclass(frozen=True)
class HistoryTurn:
    user: str
    assistant: str
    context_chunks: tuple[ContextChunk, ...] = field(default_factory=tuple)


def _gemini_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GEMINI_API_KEY is empty. Add it to the project .env at the repo root."
        )
    return key


def _gemini_model_name() -> str:
    return os.environ.get("GEMINI_MODEL", "").strip() or "gemini-2.0-flash"


def _get_gemini_chat():
    global _gemini_model
    if _gemini_model is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        _gemini_model = ChatGoogleGenerativeAI(
            model=_gemini_model_name(),
            temperature=0.0,
            google_api_key=_gemini_api_key(),
        )
    return _gemini_model


def eurlex_url(celex_id: str) -> str:
    cid = (celex_id or "").strip()
    if not cid:
        return ""
    return f"https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{cid}"


def _load_celex_docs(path: Any) -> dict[str, dict[str, Any]]:
    """Load full document text keyed by celex_id from train_dedup.jsonl."""
    out: dict[str, dict[str, Any]] = {}
    p = path if hasattr(path, "is_file") else DEDUP_TRAIN_JSONL
    if not p.is_file():
        print(f"Warning: train dedup file missing at {p}; document previews disabled.")
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            celex = str(row.get("celex_id") or "").strip()
            if not celex:
                continue
            text = str(row.get("text") or "")
            labels = row.get("labels") or []
            out[celex] = {"text": text, "labels": labels}
    return out


def _document_preview(celex_id: str) -> str:
    rec = _celex_docs.get(celex_id) or {}
    text = str(rec.get("text") or "")
    if len(text) <= DOC_PREVIEW_MAX_CHARS:
        return text
    return text[:DOC_PREVIEW_MAX_CHARS] + "…"


def startup() -> None:
    """Load dedup Chroma + chunks for hybrid retrieval (call once at API lifespan)."""
    global _ctx, _celex_docs
    chroma_path = chroma_persist_dir_dedup(CHUNK_STRATEGY)
    chunks_path = chunks_jsonl_path_dedup(CHUNK_STRATEGY)
    if not chroma_path.is_dir():
        raise FileNotFoundError(
            f"Missing dedup Chroma at {chroma_path}. "
            "Run: python -m Scripts.eval.build_chunk_indices_dedup --top10"
        )
    if not chunks_path.is_file():
        raise FileNotFoundError(
            f"Missing dedup chunks at {chunks_path}. "
            "Run: python -m Scripts.eval.build_chunk_indices_dedup --top10"
        )
    print(f"Loading vectorstore from {chroma_path} ...")
    vs = load_vectorstore(chroma_path)
    print(f"Loading chunks from {chunks_path} ...")
    documents = load_documents_from_chunks_jsonl(chunks_path)
    _ctx = RetrievalContext(vs, documents)
    print(f"Loading full documents from {DEDUP_TRAIN_JSONL} ...")
    _celex_docs = _load_celex_docs(DEDUP_TRAIN_JSONL)
    print(
        f"RAG chat ready: chunk={CHUNK_STRATEGY} retriever={RETRIEVER} "
        f"final_k={FINAL_K} candidate_k={CANDIDATE_K} min_rerank={MIN_RERANK_SCORE} "
        f"celex_docs={len(_celex_docs)}"
    )


def shutdown() -> None:
    global _ctx, _gemini_model, _celex_docs
    _ctx = None
    _gemini_model = None
    _celex_docs = {}


def _require_ctx() -> RetrievalContext:
    if _ctx is None:
        raise RuntimeError("RAG chat service not initialized (call startup() first).")
    return _ctx


def _chunk_to_source_entry(rank: int, doc: Document) -> dict[str, Any]:
    meta = doc.metadata or {}
    content = doc.page_content or ""
    snippet = content[:SNIPPET_MAX_CHARS]
    if len(content) > SNIPPET_MAX_CHARS:
        snippet += "…"
    score = meta.get("rerank_score")
    return {
        "rank": rank,
        "chunk_uid": str(meta.get("chunk_uid") or ""),
        "rerank_score": float(score) if score is not None else None,
        "snippet": snippet,
    }


def docs_to_grouped_sources(docs: list[Document]) -> list[dict[str, Any]]:
    """Group retrieved chunks by CELEX into document-level source records."""
    by_celex: dict[str, list[tuple[int, Document]]] = defaultdict(list)
    for i, doc in enumerate(docs, 1):
        celex = str((doc.metadata or {}).get("celex_id") or "")
        by_celex[celex].append((i, doc))

    out: list[dict[str, Any]] = []
    for celex in sorted(by_celex.keys(), key=lambda c: (c == "", c)):
        items = by_celex[celex]
        first_meta = items[0][1].metadata or {}
        categories = str(first_meta.get("categories_en") or "")
        chunks = [_chunk_to_source_entry(rank, d) for rank, d in items]
        out.append(
            {
                "celex_id": celex,
                "categories_en": categories,
                "eurlex_url": eurlex_url(celex),
                "document_preview": _document_preview(celex),
                "chunks": chunks,
            }
        )
    return out


def rank_to_chunk_uid(docs: list[Document]) -> dict[int, str]:
    """Map 1-based source rank to chunk_uid for the current retrieval set."""
    return {
        i: str((d.metadata or {}).get("chunk_uid") or "")
        for i, d in enumerate(docs, 1)
    }


def parse_cited_ranks(answer: str) -> set[int]:
    """Extract 1-based source indices from [n] citations in the answer."""
    ranks: set[int] = set()
    for m in _CITATION_RE.finditer(answer):
        try:
            ranks.add(int(m.group(1)))
        except ValueError:
            continue
    return ranks


def resolve_used_chunk_uids(
    answer: str,
    docs: list[Document],
) -> list[str]:
    """
    Return chunk_uids cited in ``answer``, or all retrieved chunks if none cited
    and ``RAG_CITATION_FALLBACK=all``.
    """
    rank_map = rank_to_chunk_uid(docs)
    cited = parse_cited_ranks(answer)
    uids: list[str] = []
    seen: set[str] = set()
    if cited:
        for r in sorted(cited):
            uid = rank_map.get(r, "")
            if uid and uid not in seen:
                seen.add(uid)
                uids.append(uid)
        return uids
    if CITATION_FALLBACK == "all":
        for r in sorted(rank_map.keys()):
            uid = rank_map[r]
            if uid and uid not in seen:
                seen.add(uid)
                uids.append(uid)
    return uids


def context_chunks_from_docs(
    docs: list[Document],
    used_uids: list[str],
) -> list[dict[str, Any]]:
    """Build serializable context chunks for history from retrieved docs."""
    uid_set = set(used_uids)
    out: list[dict[str, Any]] = []
    for doc in docs:
        meta = doc.metadata or {}
        uid = str(meta.get("chunk_uid") or "")
        if uid not in uid_set:
            continue
        out.append(
            {
                "chunk_uid": uid,
                "celex_id": str(meta.get("celex_id") or ""),
                "categories_en": str(meta.get("categories_en") or ""),
                "text": doc.page_content or "",
            }
        )
    return out


def _filter_docs_by_rerank_score(docs: list[Document]) -> list[Document]:
    """
    Drop chunks below ``MIN_RERANK_SCORE`` on a **0–1 scale**.

    Cross-encoder logits are min–max normalized within the retrieved batch so
    0.3 means "below 30% of the spread for this query", not a raw logit cutoff.
    """
    scored: list[tuple[Document, float]] = []
    for d in docs:
        raw = (d.metadata or {}).get("rerank_score")
        if raw is None:
            continue
        try:
            scored.append((d, float(raw)))
        except (TypeError, ValueError):
            continue
    if not scored:
        return []

    raws = [s for _, s in scored]
    lo, hi = min(raws), max(raws)
    span = hi - lo

    kept: list[Document] = []
    for d, raw in scored:
        norm = 1.0 if span <= 0 else (raw - lo) / span
        if norm < MIN_RERANK_SCORE:
            continue
        meta = dict(d.metadata or {})
        meta["rerank_score_raw"] = raw
        meta["rerank_score"] = round(norm, 4)
        kept.append(Document(page_content=d.page_content, metadata=meta))
    return kept


def _retrieve_reranked_for_chat(
    ctx: RetrievalContext,
    query: str,
    *,
    candidate_k: int,
    final_k: int,
) -> list[Document]:
    """Rerank with scores, filter by normalized threshold, cap at ``final_k`` (no unscored pad)."""
    base = base_retriever_id(RETRIEVER)
    if base not in _RETRIEVERS:
        raise ValueError(f"Unknown retriever base: {base}")
    raw = _RETRIEVERS[base](ctx, query, candidate_k)
    unique = dedupe_preserve_order(raw)
    if not unique:
        return []

    if RETRIEVER.endswith(RERANK_SUFFIX):
        ranked = [
            d
            for d, _ in rerank_documents_with_scores(
                query, unique, top_n=len(unique)
            )
        ]
    else:
        ranked = unique

    filtered = _filter_docs_by_rerank_score(ranked)
    return filtered[:final_k]


def _truncate(text: str, max_chars: int) -> str:
    t = text.strip()
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 1] + "…"


def _normalize_history(
    history: Sequence[HistoryTurn | dict[str, Any]] | None,
) -> list[HistoryTurn]:
    if not history:
        return []
    out: list[HistoryTurn] = []
    for item in history:
        if isinstance(item, HistoryTurn):
            user = item.user
            assistant = item.assistant
            chunks = item.context_chunks
        elif isinstance(item, dict):
            user = str(item.get("user") or "")
            assistant = str(item.get("assistant") or "")
            raw_chunks = item.get("context_chunks") or []
            chunks = tuple(
                ContextChunk(
                    chunk_uid=str(c.get("chunk_uid") or ""),
                    celex_id=str(c.get("celex_id") or ""),
                    categories_en=str(c.get("categories_en") or ""),
                    text=str(c.get("text") or ""),
                )
                for c in raw_chunks
                if isinstance(c, dict) and c.get("chunk_uid")
            )
        else:
            continue
        user = _truncate(user, HISTORY_MSG_MAX_CHARS)
        assistant = _truncate(assistant, HISTORY_MSG_MAX_CHARS)
        if not user or not assistant:
            continue
        out.append(HistoryTurn(user=user, assistant=assistant, context_chunks=chunks))
    if len(out) > HISTORY_MAX_MESSAGES:
        out = out[-HISTORY_MAX_MESSAGES:]
    return out


def retrieve_for_chat(
    query: str,
    history: Sequence[HistoryTurn | dict[str, Any]] | None = None,
) -> tuple[list[Document], list[dict[str, Any]]]:
    """Run retrieval on the current query only; return docs and grouped source records."""
    del history  # retrieval must not use conversation history
    ctx = _require_ctx()
    search_q = query.strip()
    docs = _retrieve_reranked_for_chat(
        ctx,
        search_q,
        candidate_k=CANDIDATE_K,
        final_k=FINAL_K,
    )
    sources = docs_to_grouped_sources(docs)
    return docs, sources


def _chat_system_prompt() -> str:
    return (
        config.SYSTEM_PROMPT
        + "\n\nThe user may ask follow-up questions in the same conversation. "
        "Use prior turns to understand what they refer to, but answer ONLY from "
        "the retrieved context in the latest message."
    )


def build_messages(
    query: str,
    docs: list[Document],
    history: Sequence[HistoryTurn | dict[str, Any]] | None = None,
) -> list[SystemMessage | HumanMessage | AIMessage]:
    """System + prior turns (user, saved chunks, assistant) + current RAG prompt."""
    messages: list[SystemMessage | HumanMessage | AIMessage] = [
        SystemMessage(content=_chat_system_prompt()),
    ]
    for turn in _normalize_history(history):
        messages.append(HumanMessage(content=turn.user))
        if turn.context_chunks:
            chunk_dicts = [
                {
                    "chunk_uid": c.chunk_uid,
                    "celex_id": c.celex_id,
                    "categories_en": c.categories_en,
                    "text": c.text,
                }
                for c in turn.context_chunks
            ]
            messages.append(
                HumanMessage(content=format_context_chunks_text(chunk_dicts))
            )
        messages.append(AIMessage(content=turn.assistant))
    messages.append(HumanMessage(content=_build_rag_prompt(query, docs)))
    return messages


def stream_answer(
    query: str,
    docs: list[Document],
    history: Sequence[HistoryTurn | dict[str, Any]] | None = None,
) -> Iterator[str]:
    """Yield text deltas from Gemini for the RAG prompt built from ``docs``."""
    chat = _get_gemini_chat()
    messages = build_messages(query, docs, history=history)
    for chunk in chat.stream(messages):
        delta = stream_chunk_to_text(chunk)
        if delta:
            yield delta
