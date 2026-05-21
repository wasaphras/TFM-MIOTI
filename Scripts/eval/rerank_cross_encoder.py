"""Lazy-loaded cross-encoder for reranking (query, passage) pairs in eval."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document

from .. import config

_model: Any = None


def _resolve_rerank_device() -> str | None:
    """
    None = try CUDA first, then fall back to CPU on OOM.
    Non-empty string = force that device (config / env ``RERANK_DEVICE``).
    """
    d = (config.RERANK_DEVICE or "").strip()
    return d if d else None


def _load_cross_encoder(device: str) -> Any:
    from sentence_transformers import CrossEncoder as CE

    return CE(config.RERANK_MODEL, device=device)


def _is_gpu_oom(exc: BaseException) -> bool:
    """PyTorch versions differ: OOM may be ``torch.OutOfMemoryError`` or ``RuntimeError``."""
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        return True
    return False


def get_cross_encoder() -> Any:
    global _model
    if _model is not None:
        return _model

    import torch

    explicit = _resolve_rerank_device()
    if explicit:
        _model = _load_cross_encoder(explicit)
        return _model

    if not torch.cuda.is_available():
        _model = _load_cross_encoder("cpu")
        return _model

    torch.cuda.empty_cache()
    try:
        _model = _load_cross_encoder("cuda")
    except Exception as e:
        if not _is_gpu_oom(e):
            raise
        torch.cuda.empty_cache()
        print(
            "Cross-encoder CUDA OOM; loading reranker on CPU "
            "(export RERANK_DEVICE=cpu to skip the GPU attempt on small or shared GPUs)."
        )
        _model = _load_cross_encoder("cpu")
    return _model


def unload_cross_encoder() -> None:
    """Drop cached CrossEncoder and release GPU memory (e.g. between eval chunk strategies)."""
    global _model
    if _model is None:
        return
    ce = _model
    _model = None
    try:
        inner = getattr(ce, "model", None)
        if inner is not None:
            try:
                import torch

                if torch.cuda.is_available():
                    try:
                        inner = inner.to("cpu")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                del inner
            except Exception:
                pass
    except Exception:
        pass
    try:
        del ce
    except Exception:
        pass
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception:
        pass


def _predict_rerank_scores(query: str, docs: list[Document]) -> list[float]:
    """Cross-encoder scores for (query, passage) pairs; same order as ``docs``."""
    if not docs:
        return []
    max_chars = config.RERANK_PASSAGE_MAX_CHARS
    model = get_cross_encoder()
    pairs: list[list[str]] = []
    for d in docs:
        text = d.page_content or ""
        if len(text) > max_chars:
            text = text[:max_chars]
        pairs.append([query, text])
    bs = max(1, int(config.RERANK_PREDICT_BATCH_SIZE))
    raw = model.predict(
        pairs,
        show_progress_bar=False,
        convert_to_numpy=True,
        batch_size=bs,
    )
    return [float(raw[i]) for i in range(len(docs))]


def rerank_documents_with_scores(
    query: str, docs: list[Document], top_n: int
) -> list[tuple[Document, float]]:
    """Rerank by CE score; attach ``rerank_score`` to each returned document metadata."""
    if not docs or top_n <= 0:
        return []
    scores = _predict_rerank_scores(query, docs)
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    out: list[tuple[Document, float]] = []
    for i in order[:top_n]:
        d = docs[i]
        meta = dict(d.metadata or {})
        meta["rerank_score"] = scores[i]
        out.append((Document(page_content=d.page_content, metadata=meta), scores[i]))
    return out


def rerank_documents(
    query: str, docs: list[Document], top_n: int
) -> list[Document]:
    """Score (query, chunk) pairs; return up to top_n documents by descending score."""
    return [d for d, _ in rerank_documents_with_scores(query, docs, top_n)]


def reset_cross_encoder_for_tests() -> None:
    """Drop cached model (e.g. between tests)."""
    unload_cross_encoder()
