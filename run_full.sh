#!/usr/bin/env bash
# Full mode: thesis corpus under Data/ (gitignored). Uses all documents in train.jsonl.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Full corpus paths (defaults in Scripts/config.py)
unset TFM_DATA_DIR
unset TFM_CATEGORIES_JSON

PYTHON="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Full mode (thesis-scale experiments)

  Corpus : Data/train.jsonl  (download via Hugging Face, not in git)
  GT     : typically 100 questions (ground_truth_generate --n 100)
  Data   : project Data/  (do not set TFM_DATA_DIR)

Run one step:
  ./run_full.sh ingest          Download / refresh train.jsonl (needs HF_TOKEN)
  ./run_full.sh index           Build 10 chunk JSONLs + Chroma (slow; Ollama)
  ./run_full.sh ground-truth    Generate 100 GT questions
  ./run_full.sh grid            Main 10x20 retrieval grid
  ./run_full.sh top10           Top-10 evals 1-4 (k=20, 100 GT default)
  ./run_full.sh dedup-index     Dedup chunk+Chroma filter (needs train_dedup.jsonl)
  ./run_full.sh dedup-gt        Dedup ground truth (100 rows default)
  ./run_full.sh dedup-eval      Dedup top-10 eval1+2 (bash driver)

Run in order (prints commands only; does not execute):
  ./run_full.sh plan

Prerequisites:
  pip install -r requirements.txt
  ollama pull qwen3-embedding:4b && ollama pull llama3.2
  export HF_TOKEN=...   # first download only

See README.md Steps 1-6 for detail.
EOF
}

step_ingest() {
  echo "=== ingest: MultiEURLEX -> Data/train.jsonl ==="
  "$PYTHON" -m Scripts.data_extraction_load
}

step_index() {
  echo "=== index: 10 strategies -> Data/chunks_* + Data/chroma_chunk_* ==="
  "$PYTHON" -m Scripts.eval.build_chunk_indices --all
}

step_ground_truth() {
  echo "=== ground-truth: Data/ground_truth.jsonl (n=100) ==="
  "$PYTHON" -m Scripts.eval.ground_truth_generate --n 100
}

step_grid() {
  echo "=== grid: Data/eval/*.csv (10 strategies x 20 retrievers) ==="
  "$PYTHON" -m Scripts.eval.run_grid_eval
}

step_top10() {
  echo "=== top10: eval1-4 under Data/eval_top10/ ==="
  "$PYTHON" -m Scripts.eval.top10.neighbor_index --strategies \
    len_500_o50 len_1000_o100 len_1500_o150 len_2000_o200 rec_nn_priority
  "$PYTHON" -m Scripts.eval.top10.run_all
}

step_dedup_index() {
  echo "=== dedup-index: filter to Data/train_dedup.jsonl CELEX set ==="
  "$PYTHON" -m Scripts.eval.build_chunk_indices_dedup --top10
}

step_dedup_gt() {
  echo "=== dedup-gt: Data/ground_truth_dedup_top10_100.jsonl ==="
  "$PYTHON" -m Scripts.eval.ground_truth_generate_dedup
}

step_dedup_eval() {
  echo "=== dedup-eval: eval_top10_dedup baseline + neighbors ==="
  bash Scripts/eval/run_dedup_top10_evals.sh
}

plan() {
  cat <<'EOF'
Full pipeline (run from repo root, in order):

  ./run_full.sh ingest
  ./run_full.sh index
  ./run_full.sh ground-truth
  ./run_full.sh grid
  ./run_full.sh top10
  ./run_full.sh dedup-index
  ./run_full.sh dedup-gt
  ./run_full.sh dedup-eval

Optional: Ragas on dedup track (see README "Ragas evaluation").
EOF
}

echo "=== TFM full mode ==="
echo "  Data directory: $REPO_ROOT/Data"
echo ""

case "${1:-help}" in
  ingest) step_ingest ;;
  index) step_index ;;
  ground-truth|gt) step_ground_truth ;;
  grid) step_grid ;;
  top10) step_top10 ;;
  dedup-index) step_dedup_index ;;
  dedup-gt) step_dedup_gt ;;
  dedup-eval) step_dedup_eval ;;
  plan) plan ;;
  help|-h|"") usage ;;
  *)
    echo "Unknown step: $1" >&2
    usage >&2
    exit 2
    ;;
esac
