#!/usr/bin/env bash
# End-to-end smoke test on tests/fixture/Data (10 docs, 2 GT questions).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

export TFM_DATA_DIR="${TFM_DATA_DIR:-$REPO_ROOT/tests/fixture/Data}"
export TFM_CATEGORIES_JSON="${TFM_CATEGORIES_JSON:-$REPO_ROOT/Data/categories.json}"
# Small GPUs often OOM on bge-reranker-v2-m3; CPU rerank is fine for smoke.
export RERANK_DEVICE="${RERANK_DEVICE:-cpu}"
export EVAL_CUDA_EMPTY_CACHE="${EVAL_CUDA_EMPTY_CACHE:-1}"

PYTHON="${PYTHON:-python}"
QUICK=1
FULL_GRID=0
REBUILD=0
SKIP_RAGAS=0

for arg in "$@"; do
  case "$arg" in
    --quick) QUICK=1 ;;
    --full-grid) FULL_GRID=1; QUICK=0 ;;
    --rebuild) REBUILD=1 ;;
    --skip-ragas) SKIP_RAGAS=1 ;;
    -h|--help)
      echo "Usage: $0 [--full-grid] [--quick] [--rebuild] [--skip-ragas]"
      echo "  Default: 3 base retrievers (fast). --full-grid: 20 retrievers, 200 summary rows."
      echo "  TFM_DATA_DIR defaults to tests/fixture/Data"
      exit 0
      ;;
    *) echo "Unknown arg: $arg" >&2; exit 2 ;;
  esac
done

pass=0
fail=0

stage() {
  local name="$1"
  shift
  local t0
  t0=$(date +%s)
  echo ""
  echo "=== $name ==="
  if "$@"; then
    local dt=$(( $(date +%s) - t0 ))
    echo "PASS ($name) ${dt}s"
    pass=$((pass + 1))
  else
    echo "FAIL ($name)"
    fail=$((fail + 1))
  fi
}

stage validate_fixture "$PYTHON" tests/validate_fixture.py

stage sanity_fixture bash -c '
  test -f "'"$TFM_DATA_DIR"'/train.jsonl" &&
  test -f "'"$TFM_DATA_DIR"'/eval_corpus_manifest.json" &&
  test -f "'"$TFM_DATA_DIR"'/ground_truth.jsonl" &&
  test -f "'"$TFM_DATA_DIR"'/train_dedup.jsonl" &&
  test -f "'"$TFM_DATA_DIR"'/eval_corpus_manifest_dedup.json" &&
  test -f "'"$TFM_DATA_DIR"'/ground_truth_dedup_top10_100.jsonl" &&
  test -f "'"$TFM_DATA_DIR"'/chunks_len_500_o50.jsonl" &&
  test -f "'"$TFM_DATA_DIR"'/chroma_chunk_len_500_o50/chroma.sqlite3"
'

if [[ "$REBUILD" -eq 1 ]]; then
  stage rebuild_fixture "$PYTHON" tests/build_fixture.py
fi

if [[ "$QUICK" -eq 0 ]]; then
  RETRIEVERS=(
    dense_sim_k10 dense_mmr_k10 bm25_k10
    hyb_rrf_k60 hyb_rrf_k30 hyb_rrf_fetch40
    hyb_weighted_norm hyb_weighted_dense_70
    hyb_interleave hyb_fill_dense_then_bm25
    dense_sim_k10_ce_r50 dense_mmr_k10_ce_r50 bm25_k10_ce_r50
    hyb_rrf_k60_ce_r50 hyb_rrf_k30_ce_r50 hyb_rrf_fetch40_ce_r50
    hyb_weighted_norm_ce_r50 hyb_weighted_dense_70_ce_r50
    hyb_interleave_ce_r50 hyb_fill_dense_then_bm25_ce_r50
  )
else
  RETRIEVERS=(dense_sim_k10 bm25_k10 hyb_rrf_k60)
fi

stage grid_eval "$PYTHON" -m Scripts.eval.run_grid_eval \
  --limit-queries 2 \
  --no-resume \
  --retrievers "${RETRIEVERS[@]}"

_grid_expected_rows() {
  if [[ "$FULL_GRID" -eq 1 ]]; then echo 200; elif [[ "$QUICK" -eq 1 ]]; then echo 30; else echo 200; fi
}

stage grid_rows bash -c '
  n=$(tail -n +2 "'"$TFM_DATA_DIR"'/eval/results_summary.csv" | wc -l | tr -d " ")
  expected='"$(_grid_expected_rows)"'
  echo "results_summary rows: $n (expected $expected)"
  test "$n" -eq "$expected"
'

stage top10_all "$PYTHON" -m Scripts.eval.top10.run_all \
  --limit-queries 2 \
  --no-resume

stage dedup_eval1 "$PYTHON" -m Scripts.eval.top10.run_eval1_baseline_dedup \
  --limit-queries 2 \
  --no-resume

stage dedup_eval2 "$PYTHON" -m Scripts.eval.top10.run_eval2_neighbors_dedup \
  --limit-queries 2 \
  --no-resume

stage dedup_merge "$PYTHON" -m Scripts.eval.merge_top10_summaries_dedup

if [[ "$SKIP_RAGAS" -eq 0 ]]; then
  RAGAS_OUT="$TFM_DATA_DIR/eval_top10_dedup/llm_triad_len500_hyb_fill"
  rm -f "$RAGAS_OUT/rag_responses.jsonl" "$RAGAS_OUT/ragas_scores.jsonl" 2>/dev/null || true

  stage ragas_gen "$PYTHON" -m Scripts.eval.llm_triad.generate_rag_responses \
    --limit-queries 2 \
    --out "$RAGAS_OUT/rag_responses.jsonl"

  stage ragas_judge "$PYTHON" -m Scripts.eval.llm_triad.judge_rag_triad \
    --in "$RAGAS_OUT/rag_responses.jsonl" \
    --out "$RAGAS_OUT/ragas_scores.jsonl" \
    --ground-truth "$TFM_DATA_DIR/ground_truth_dedup_top10_100.jsonl" \
    --limit-queries 2 \
    --provider ollama
else
  echo "Skipping Ragas (--skip-ragas)"
fi

echo ""
echo "Smoke test finished: $pass passed, $fail failed"
if [[ "$fail" -gt 0 ]]; then
  exit 1
fi
