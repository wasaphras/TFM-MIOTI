"""
Map EuroVOC label IDs in the loaded DataFrame to human-readable names using categories.json.
Run after load_data() and before chunk_documents().
"""

import json
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

from . import config


def load_categories(path: Path | str | None = None) -> dict:
    """Load the EuroVOC id → multilingual labels map from JSON."""
    p = Path(path or config.CATEGORIES_JSON)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def resolve_label_ids_to_en(
    ids: list[str] | None,
    categories: dict,
    lang: str | None = None,
) -> list[str]:
    """
    Resolve each label id to the string for `lang` (default: config.RAG_LABEL_LANG).
    Unknown ids are kept as their original string so nothing is dropped silently.
    """
    lang = lang or config.RAG_LABEL_LANG
    if not ids:
        return []
    out: list[str] = []
    for lid in ids:
        key = str(lid)
        entry = categories.get(key)
        if entry is None:
            out.append(key)
            continue
        label = entry.get(lang) or entry.get("en")
        out.append(label if label is not None else key)
    return out


def _normalize_label_cell(raw) -> list[str]:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        return [raw]
    return [str(x) for x in raw]


def preprocess_for_rag(
    df: pd.DataFrame,
    categories_path: Path | str | None = None,
) -> pd.DataFrame:
    """
    Add column `labels_en`: English (or RAG_LABEL_LANG) category names for each row's `labels`.
    Original `labels` (EuroVOC ids) are unchanged.
    """
    categories = load_categories(categories_path)
    lang = config.RAG_LABEL_LANG

    df = df.copy()
    tqdm.pandas(desc="Preprocess labels_en", unit="row")
    df["labels_en"] = df["labels"].progress_apply(
        lambda cell: resolve_label_ids_to_en(
            _normalize_label_cell(cell), categories, lang
        )
    )
    return df


if __name__ == "__main__":
    from .data_extraction_load import main as load_main

    frame = load_main()
    frame = preprocess_for_rag(frame)
    print(frame[["celex_id", "labels", "labels_en"]].head().to_string())
