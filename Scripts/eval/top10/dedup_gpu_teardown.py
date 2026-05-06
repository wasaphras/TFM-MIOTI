"""Release GPU/RAM between dedup eval subprocesses.

This module runs in its **own** Python interpreter (see ``run_dedup_top10_evals.sh``).
``unload_cross_encoder()`` / ``release_chroma_process_cache()`` only affect *this*
process; they do not reclaim VRAM from a sibling that already exited (the driver
does that when the child exits). They still help if this interpreter ever held
models, and they align with a consistent teardown sequence.

Environment:

- ``DEDUP_EVAL_OLLAMA_STOP`` — if truthy, run ``ollama stop <EMBEDDING_MODEL>``
  so the Ollama embedding server releases GPU memory before a prefetch-read phase
  loads the cross-encoder on CUDA. Requires ``ollama`` on ``PATH``. Opt-in
  because it stops that model for all clients until the next embed loads it.

This process also runs ``torch.cuda.synchronize()`` and ``empty_cache()`` when CUDA
is available (independent of ``EVAL_CUDA_EMPTY_CACHE``, which only gates extra
cache clears inside the eval child loops).
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys


def _truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


def _warn(msg: str) -> None:
    print(f"dedup_gpu_teardown: {msg}", file=sys.stderr)


def maybe_stop_ollama_embedding_model() -> None:
    if not _truthy_env("DEDUP_EVAL_OLLAMA_STOP"):
        return
    try:
        from ... import config as project_config
    except Exception as exc:
        _warn(f"ollama stop skipped (config import): {exc}")
        return
    model = project_config.EMBEDDING_MODEL
    try:
        subprocess.run(
            ["ollama", "stop", model],
            check=False,
            timeout=120,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        _warn(f"ollama stop failed ({model!r}): {exc}")


def main() -> None:
    try:
        from ..rerank_cross_encoder import unload_cross_encoder

        unload_cross_encoder()
    except Exception as exc:
        _warn(f"unload_cross_encoder: {exc}")
    try:
        from ...embeddings_chromadb import release_chroma_process_cache

        release_chroma_process_cache()
    except Exception as exc:
        _warn(f"release_chroma_process_cache: {exc}")

    gc.collect()

    maybe_stop_ollama_embedding_model()
    gc.collect()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
    except Exception as exc:
        _warn(f"torch cuda sync/empty: {exc}")

    gc.collect()


if __name__ == "__main__":
    main()
