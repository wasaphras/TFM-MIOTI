#!/usr/bin/env bash
# Idempotent dedup top-10 eval1 + eval2 (prefetch-write then prefetch-read), then merge.
# Skips steps that already finished. Does NOT delete checkpoints/CSVs unless you pass --reset.
#
# Memory contract (GPU): each phase is a separate ``python -m`` process. When that
# child exits, its PyTorch CUDA allocations are released by the driver. Prefetch-write
# uses Chroma + Ollama (/api/embed); prefetch-read uses only the cross-encoder on disk
# candidates. Ollama runs in a separate daemon and often keeps the embedding model on
# GPU after prefetch-write; the next child may then OOM when loading the reranker on
# the same GPU. After each prefetch-write (or skip), this script runs
# ``python -m Scripts.eval.top10.dedup_gpu_teardown``; set ``DEDUP_EVAL_OLLAMA_STOP=1``
# to run ``ollama stop`` on ``config.EMBEDDING_MODEL`` there (requires ``ollama`` on PATH).
#
# Usage:
#   bash Scripts/eval/run_dedup_top10_evals.sh
#   bash Scripts/eval/run_dedup_top10_evals.sh --reset   # wipe dedup eval outputs + prefetch + merge, then run all
#   DEDUP_EVAL_RESET=1 bash ...   # same as --reset
# Optional: DEDUP_EVAL_SLEEP_BETWEEN_STEPS=6  # seconds after each teardown (default 4)
# Optional: DEDUP_EVAL_OLLAMA_STOP=1          # free Ollama GPU VRAM before prefetch-read (see dedup_gpu_teardown)
# Conda: if ``conda`` is on PATH and env ``Data`` exists, Python steps use
#   ``conda run -n Data --no-capture-output python`` (override name with ``DEDUP_CONDA_ENV``).
#   Set ``DEDUP_NO_CONDA=1`` to force plain ``python`` on PATH.
#
# Prereqs (once): build_chunk_indices_dedup --top10, ground_truth_generate_dedup, neighbor_index_dedup --top10
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"

PY=(python)
if [[ "${DEDUP_NO_CONDA:-}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  _ce="${DEDUP_CONDA_ENV:-Data}"
  if [[ -n "$_ce" ]] && conda env list 2>/dev/null | awk -v e="$_ce" '$1==e {found=1} END{exit !found}'; then
    PY=(conda run -n "$_ce" --no-capture-output python)
    echo "== Using conda env for Python: $_ce"
  fi
fi

teardown() {
  echo "== Teardown (CE + Chroma cache + CUDA; optional Ollama stop if DEDUP_EVAL_OLLAMA_STOP=1)"
  "${PY[@]}" -m Scripts.eval.top10.dedup_gpu_teardown || true
  sleep "${DEDUP_EVAL_SLEEP_BETWEEN_STEPS:-4}"
}

load_state() {
  eval "$("${PY[@]}" -m Scripts.eval.top10.dedup_eval_state --export-sh)"
}

if [[ "${1:-}" == "--reset" ]] || [[ "${DEDUP_EVAL_RESET:-}" == "1" ]]; then
  echo "== --reset: removing dedup eval outputs, prefetch trees, and merged CSV"
  rm -rf Data/eval_top10_dedup/prefetch/eval1_baseline
  rm -rf Data/eval_top10_dedup/prefetch/eval2_neighbors
  rm -f Data/eval_top10_dedup/eval1_baseline/checkpoint.json
  rm -f Data/eval_top10_dedup/eval1_baseline/results_summary.csv
  rm -f Data/eval_top10_dedup/eval1_baseline/rank_breakdown_long.csv
  rm -f Data/eval_top10_dedup/eval1_baseline/hit_rate_pivot.csv
  rm -f Data/eval_top10_dedup/eval2_neighbors/checkpoint.json
  rm -f Data/eval_top10_dedup/eval2_neighbors/results_summary.csv
  rm -f Data/eval_top10_dedup/eval2_neighbors/rank_breakdown_long.csv
  rm -f Data/eval_top10_dedup/eval2_neighbors/hit_rate_pivot.csv
  rm -f Data/eval_top10_dedup/results_summary_baseline_neighbors.csv
  echo "== Reset done."
fi

echo "== Repo: $ROOT"
load_state
echo "== State: nq=$NQ  e1_prefetch_write=$E1_PREFETCH_WRITE_DONE  e1_prefetch_read=$E1_PREFETCH_READ_DONE  e2_prefetch_write=$E2_PREFETCH_WRITE_DONE  e2_prefetch_read=$E2_PREFETCH_READ_DONE  merge=$MERGE_DONE"

if [[ "$NQ" -eq 0 ]]; then
  echo "ERROR: No ground-truth rows at Data/ground_truth_dedup_top10_100.jsonl (or file missing)." >&2
  exit 1
fi

# --- Eval 1 ---
if [[ "$E1_PREFETCH_WRITE_DONE" != "1" ]]; then
  echo ""
  echo "== [eval1] prefetch-write (resume-safe; Chroma + BM25; no CE)"
  EVAL_CUDA_EMPTY_CACHE=1 "${PY[@]}" -m Scripts.eval.top10.run_eval1_baseline_dedup --prefetch-write
else
  echo ""
  echo "== [eval1] prefetch-write SKIPPED (already complete)"
fi
teardown

if [[ "$E1_PREFETCH_READ_DONE" != "1" ]]; then
  echo ""
  echo "== [eval1] prefetch-read (resume-safe; rerank from disk)"
  EVAL_CUDA_EMPTY_CACHE=1 "${PY[@]}" -m Scripts.eval.top10.run_eval1_baseline_dedup --prefetch-read
else
  echo ""
  echo "== [eval1] prefetch-read SKIPPED (results_summary.csv already complete)"
fi
teardown

# --- Eval 2 ---
if [[ "$E2_PREFETCH_WRITE_DONE" != "1" ]]; then
  echo ""
  echo "== [eval2] prefetch-write (resume-safe)"
  EVAL_CUDA_EMPTY_CACHE=1 "${PY[@]}" -m Scripts.eval.top10.run_eval2_neighbors_dedup --prefetch-write
else
  echo ""
  echo "== [eval2] prefetch-write SKIPPED (already complete)"
fi
teardown

if [[ "$E2_PREFETCH_READ_DONE" != "1" ]]; then
  echo ""
  echo "== [eval2] prefetch-read (resume-safe)"
  EVAL_CUDA_EMPTY_CACHE=1 "${PY[@]}" -m Scripts.eval.top10.run_eval2_neighbors_dedup --prefetch-read
else
  echo ""
  echo "== [eval2] prefetch-read SKIPPED (results_summary.csv already complete)"
fi
teardown

# --- Merge ---
if [[ "$MERGE_DONE" != "1" ]]; then
  echo ""
  echo "== [merge] merge_top10_summaries_dedup"
  "${PY[@]}" -m Scripts.eval.merge_top10_summaries_dedup
else
  echo ""
  echo "== [merge] SKIPPED (merged CSV already present)"
fi

load_state
echo ""
echo "== Final state: e1_prefetch_write=$E1_PREFETCH_WRITE_DONE  e1_prefetch_read=$E1_PREFETCH_READ_DONE  e2_prefetch_write=$E2_PREFETCH_WRITE_DONE  e2_prefetch_read=$E2_PREFETCH_READ_DONE  merge=$MERGE_DONE"
echo "== Outputs:"
echo "    Data/eval_top10_dedup/eval1_baseline/results_summary.csv"
echo "    Data/eval_top10_dedup/eval2_neighbors/results_summary.csv"
echo "    Data/eval_top10_dedup/results_summary_baseline_neighbors.csv"
