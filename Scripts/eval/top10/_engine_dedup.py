"""Dedup corpus: same eval engine as _engine with DEDUP layout."""

from __future__ import annotations

from ..corpus_layout import DEDUP
from ._engine import run_checkpointed_eval as _run_checkpointed_eval


def run_checkpointed_eval(**kwargs):
    kwargs.setdefault("layout", DEDUP)
    return _run_checkpointed_eval(**kwargs)
