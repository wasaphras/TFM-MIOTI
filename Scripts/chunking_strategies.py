"""Ten chunking strategies: five length-first and five delimiter-first (max 2000 chars)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

from langchain_core.documents import Document
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

import pandas as pd

from . import config

# Default separator ladder for length-first family (isolates size/overlap only).
LENGTH_SEPARATORS = ["\n\n", "\n", " ", ""]

CHUNK_STRATEGY_IDS: tuple[str, ...] = (
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


def row_meta(row) -> dict:
    labels_en = row["labels_en"]
    if not isinstance(labels_en, list):
        labels_en = (
            list(labels_en)
            if hasattr(labels_en, "__iter__") and not isinstance(labels_en, str)
            else []
        )
    raw_labels = row.get("labels") or []
    if not isinstance(raw_labels, list):
        raw_labels = (
            [raw_labels] if raw_labels is not None and str(raw_labels) != "nan" else []
        )
    celex = str(row.get("celex_id", "") or "")
    return {
        "celex_id": celex,
        "label_ids": ",".join(str(x) for x in raw_labels),
        "categories_en": ", ".join(labels_en),
    }


def _recursive_split_oversize(
    text: str,
    base_meta: dict,
    chunk_size: int = 2000,
    chunk_overlap: int = 0,
    separators: list[str] | None = None,
) -> list[Document]:
    """Split a single oversized segment to ≤ chunk_size using RecursiveCharacter."""
    seps = separators or LENGTH_SEPARATORS
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=seps,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents([Document(page_content=text, metadata=dict(base_meta))])


def _merge_paragraphs_nn(text: str, base_meta: dict, max_chars: int = 2000) -> list[Document]:
    """
    Split on \\n\\n, greedily merge consecutive paragraphs until adding the next
    would exceed max_chars. Paragraphs longer than max_chars are recursively split.
    """
    parts = text.split("\n\n")
    chunks: list[Document] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if not current:
            return
        merged = "\n\n".join(current)
        current = []
        if len(merged) <= max_chars:
            chunks.append(Document(page_content=merged, metadata=dict(base_meta)))
        else:
            chunks.extend(_recursive_split_oversize(merged, base_meta, 2000, 0))

    for p in parts:
        p = p.strip()
        if not p:
            continue
        candidate = "\n\n".join(current + [p]) if current else p
        if len(p) > max_chars:
            flush()
            chunks.extend(_recursive_split_oversize(p, base_meta, 2000, 0))
        elif len(candidate) <= max_chars:
            current.append(p)
        else:
            flush()
            current = [p]
    flush()
    return chunks


def _merge_lines_n(text: str, base_meta: dict, max_chars: int = 2000) -> list[Document]:
    """Split on \\n, merge consecutive lines into batches ≤ max_chars."""
    lines = text.split("\n")
    chunks: list[Document] = []
    current: list[str] = []

    def flush():
        nonlocal current
        if not current:
            return
        merged = "\n".join(current)
        current = []
        if len(merged) <= max_chars:
            chunks.append(Document(page_content=merged, metadata=dict(base_meta)))
        else:
            chunks.extend(_recursive_split_oversize(merged, base_meta, 2000, 0))

    for line in lines:
        line = line.rstrip()
        if not line and not current:
            continue
        candidate = "\n".join(current + [line]) if current else line
        if len(line) > max_chars:
            flush()
            chunks.extend(_recursive_split_oversize(line, base_meta, 2000, 0))
        elif len(candidate) <= max_chars:
            current.append(line)
        else:
            flush()
            current = [line]
    flush()
    return chunks


def _assign_chunk_uids(docs: list[Document], strategy_id: str, start_index: int = 0) -> None:
    for i, doc in enumerate(docs):
        idx = start_index + i
        celex = str(doc.metadata.get("celex_id", "") or "")
        raw = f"{celex}|{strategy_id}|{idx}".encode("utf-8")
        doc.metadata["chunk_uid"] = hashlib.sha256(raw).hexdigest()[:32]
        doc.metadata["chunk_strategy"] = strategy_id


_STRATEGY_FUNCS: dict[str, Callable[[str, dict], list[Document]]] = {}


def _register(sid: str):
    def deco(fn: Callable[[str, dict], list[Document]]):
        _STRATEGY_FUNCS[sid] = fn
        return fn

    return deco


@_register("len_500_o50")
def _len_500_o50(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=LENGTH_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("len_1000_o100")
def _len_1000_o100(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=LENGTH_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("len_1500_o150")
def _len_1500_o150(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150,
        separators=LENGTH_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("len_2000_o0")
def _len_2000_o0(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=0,
        separators=LENGTH_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("len_2000_o200")
def _len_2000_o200(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        separators=LENGTH_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("para_nn_merge")
def _para_nn_merge(text: str, meta: dict) -> list[Document]:
    return _merge_paragraphs_nn(text, meta, 2000)


@_register("line_n_merge")
def _line_n_merge(text: str, meta: dict) -> list[Document]:
    return _merge_lines_n(text, meta, 2000)


@_register("char_nn_only")
def _char_nn_only(text: str, meta: dict) -> list[Document]:
    """
    Split on \\n\\n first (CharacterTextSplitter). Unlike RecursiveCharacter,
    CharacterTextSplitter can emit segments *longer* than chunk_size when there
    is no separator match (e.g. one huge paragraph) — that breaks embedding
    context limits; oversized pieces are split with RecursiveCharacter.
    """
    sp = CharacterTextSplitter(
        separator="\n\n",
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    base_meta = dict(meta)
    docs = sp.split_documents([Document(page_content=text, metadata=dict(base_meta))])
    out: list[Document] = []
    max_chars = 2000
    overlap = 100
    for d in docs:
        if len(d.page_content) <= max_chars:
            out.append(d)
        else:
            out.extend(
                _recursive_split_oversize(
                    d.page_content,
                    base_meta,
                    chunk_size=max_chars,
                    chunk_overlap=overlap,
                    separators=["\n", " ", ""],
                )
            )
    return out


@_register("rec_nn_priority")
def _rec_nn_priority(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


@_register("rec_legal_markers")
def _rec_legal_markers(text: str, meta: dict) -> list[Document]:
    sp = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=150,
        separators=["\nArticle ", "\nCHAPTER ", "\n\n", "\n", " ", ""],
        length_function=len,
        is_separator_regex=False,
    )
    return sp.split_documents([Document(page_content=text, metadata=dict(meta))])


def chunk_one_document(text: str, base_meta: dict, strategy_id: str) -> list[Document]:
    """Chunk a single document's text (for ground-truth containment checks)."""
    if strategy_id not in _STRATEGY_FUNCS:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")
    docs = _STRATEGY_FUNCS[strategy_id](text, base_meta)
    _assign_chunk_uids(docs, strategy_id, start_index=0)
    return docs


def chunk_documents_with_strategy(
    df: pd.DataFrame,
    strategy_id: str,
    save_chunks: bool = False,
    chunks_out_path: str | None = None,
) -> list[Document]:
    """
    Split all rows in df using the named strategy. Adds chunk_uid and chunk_strategy.
    """
    if "labels_en" not in df.columns:
        raise ValueError("DataFrame must include labels_en (preprocess_for_rag).")
    if strategy_id not in _STRATEGY_FUNCS:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

    all_chunks: list[Document] = []
    global_i = 0
    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"Chunk documents ({strategy_id})",
        unit="doc",
    ):
        meta = row_meta(row)
        docs = _STRATEGY_FUNCS[strategy_id](row["text"], meta)
        _assign_chunk_uids(docs, strategy_id, start_index=global_i)
        global_i += len(docs)
        all_chunks.extend(docs)

    if save_chunks and chunks_out_path:
        import json

        with open(chunks_out_path, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(
                tqdm(all_chunks, desc="Save chunks JSONL", unit="chunk")
            ):
                f.write(
                    json.dumps(
                        {
                            "chunk_id": i,
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return all_chunks


def write_chunks_jsonl_from_df(
    df: pd.DataFrame,
    strategy_id: str,
    out_path: str | Path,
) -> int:
    """
    Chunk ``df`` with ``strategy_id`` and write JSONL incrementally (no full
    in-memory list of all chunks). Returns number of chunk lines written.
    """
    if "labels_en" not in df.columns:
        raise ValueError("DataFrame must include labels_en (preprocess_for_rag).")
    if strategy_id not in _STRATEGY_FUNCS:
        raise ValueError(f"Unknown strategy_id: {strategy_id}")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    global_i = 0
    chunk_id = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for _, row in tqdm(
            df.iterrows(),
            total=len(df),
            desc=f"Chunk+write JSONL ({strategy_id})",
            unit="doc",
        ):
            meta = row_meta(row)
            docs = _STRATEGY_FUNCS[strategy_id](row["text"], meta)
            _assign_chunk_uids(docs, strategy_id, start_index=global_i)
            global_i += len(docs)
            for chunk in docs:
                f.write(
                    json.dumps(
                        {
                            "chunk_id": chunk_id,
                            "page_content": chunk.page_content,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                chunk_id += 1
    return chunk_id


def chroma_persist_dir(strategy_id: str) -> str:
    from .eval.corpus_layout import STANDARD

    return STANDARD.chroma_persist_dir_str(strategy_id)
