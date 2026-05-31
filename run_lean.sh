#!/usr/bin/env bash
# Lean mode: 10-document corpus + 2 ground-truth questions (committed fixture).
# Does not download MultiEURLEX or use Data/train.jsonl.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

export TFM_DATA_DIR="${TFM_DATA_DIR:-$REPO_ROOT/tests/fixture/Data}"
export TFM_CATEGORIES_JSON="${TFM_CATEGORIES_JSON:-$REPO_ROOT/Data/categories.json}"
export RERANK_DEVICE="${RERANK_DEVICE:-cpu}"
export EVAL_CUDA_EMPTY_CACHE="${EVAL_CUDA_EMPTY_CACHE:-1}"

PYTHON="${PYTHON:-python}"

usage() {
  cat <<'EOF'
Lean mode (default for clones and CI)

  Corpus : tests/fixture/Data  (10 docs, ~32 MB, in git)
  GT     : 2 questions per track (standard + dedup)
  Data   : TFM_DATA_DIR -> fixture (not Data/)

Commands:
  ./run_lean.sh              Run full lean pipeline check (smoke test)
  ./run_lean.sh --verify     Fixture files/sizes only (not pip or Ollama)
  ./run_lean.sh --rebuild    Rebuild fixture from Data/train.jsonl, then smoke test
  ./run_lean.sh --full-grid  Smoke test with 20 retrievers (200-row grid summary)
  ./run_lean.sh --skip-ragas Skip Ragas judge stage

  ./run_lean.sh --help       This message

Rebuild fixture manually:
  python tests/build_fixture.py   # needs Ollama + local Data/train.jsonl
EOF
}

for arg in "$@"; do
  if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
    usage
    exit 0
  fi
done

echo "=== TFM lean mode ==="
echo "  TFM_DATA_DIR=$TFM_DATA_DIR"
echo ""

if [[ "${1:-}" == "--verify" ]]; then
  shift
  "$PYTHON" tests/validate_fixture.py
  test -f "$TFM_DATA_DIR/train.jsonl"
  test -f "$TFM_DATA_DIR/ground_truth.jsonl"
  echo "Lean verify OK"
  exit 0
fi

exec bash tests/run_smoke_test.sh "$@"
