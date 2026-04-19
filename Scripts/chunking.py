"""
Chunking module: splits documents into smaller chunks for embedding and retrieval.

Expects a DataFrame produced by preprocess_for_rag() (column `labels_en`).
English categories and celex_id are stored in chunk metadata only (not in
page_content). Rebuild Chroma after changing chunking or metadata.

For the 10-strategy evaluation grid, see `chunking_strategies.py`.
"""

import json

from langchain_core.documents import Document
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter

import pandas as pd

from . import config


def _save_chunks(chunks: list[Document], path: str | None = None) -> None:
    """Save chunks to JSONL for analysis."""
    out_path = path or config.CHUNKS_JSONL
    with open(out_path, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(
            tqdm(chunks, desc="Save chunks.jsonl", unit="chunk")
        ):
            record = {
                "chunk_id": i,
                "page_content": chunk.page_content,
                "metadata": chunk.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def chunk_documents(df: pd.DataFrame, save_chunks: bool = True) -> list[Document]:
    """
    Split DataFrame text column into LangChain Document chunks.
    Requires `labels_en` from preprocess_for_rag(). Metadata per chunk: celex_id,
    label_ids, categories_en (comma-separated English labels).
    """
    if "labels_en" not in df.columns:
        raise ValueError(
            "DataFrame must include 'labels_en'. Run preprocess_for_rag() after load_data()."
        )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=config.CHUNK_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    all_chunks: list[Document] = []
    for _, row in tqdm(
        df.iterrows(), total=len(df), desc="Chunk documents", unit="doc"
    ):
        labels_en = row["labels_en"]
        if not isinstance(labels_en, list):
            labels_en = list(labels_en) if hasattr(labels_en, "__iter__") and not isinstance(labels_en, str) else []

        raw_labels = row.get("labels") or []
        if not isinstance(raw_labels, list):
            raw_labels = [raw_labels] if raw_labels is not None and str(raw_labels) != "nan" else []

        celex = str(row.get("celex_id", "") or "")
        meta = {
            "celex_id": celex,
            "label_ids": ",".join(str(x) for x in raw_labels),
            "categories_en": ", ".join(labels_en),
        }
        full_doc = Document(page_content=row["text"], metadata=dict(meta))
        chunks = text_splitter.split_documents([full_doc])
        all_chunks.extend(chunks)

    if save_chunks:
        _save_chunks(all_chunks)

    return all_chunks


if __name__ == "__main__":
    from .data_extraction_load import main as load_main
    from .preprocess import preprocess_for_rag

    df = preprocess_for_rag(load_main())
    documents = chunk_documents(df)
    print(f"Created {len(documents)} LangChain document chunks.")
    print(f"Chunks saved to {config.CHUNKS_JSONL}")
