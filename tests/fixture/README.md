# Smoke-test fixture (10 documents)

Prebuilt corpus for CI and clone verification. **Not** a slice of the 15 GB full-corpus Chroma trees under `Data/chroma_chunk_*`.

| What | Size (typical) |
|------|----------------|
| Whole fixture | ~30-35 MB |
| Each `chroma_chunk_<strategy>/` | ~2-3 MB |
| Full thesis corpus Chroma (do not copy here) | ~5-15 GB per strategy |

## Contents

- `train.jsonl` - 10 English documents (fixed seed 42)
- `train_dedup.jsonl` - 8 of those CELEX ids
- `chunks_*.jsonl` + `chroma_chunk_*` - all 10 strategies, embedded via Ollama
- `chunks_dedup_*` + `chroma_chunk_dedup_*` - dedup track (top-10 strategies)
- `ground_truth.jsonl` and `ground_truth_dedup_top10_100.jsonl` - 2 rows each
- `neighbor_index/` and `neighbor_index_dedup/` - pickle sidecars for eval 2

Eval CSVs under `eval/` and `eval_top10*` are optional smoke outputs; regenerate with `bash tests/run_smoke_test.sh`.

## Rebuild (from full `Data/train.jsonl` on this machine)

Requires Ollama (`qwen3-embedding:4b`, `llama3.2` for optional LLM GT):

```bash
conda run -n Data --no-capture-output python tests/build_fixture.py
python tests/validate_fixture.py
```

This **re-embeds** 10 docs; it does **not** copy multi-GB Chroma from `Data/`.

## Run smoke test

```bash
bash tests/run_smoke_test.sh --skip-ragas
# bash tests/run_smoke_test.sh --full-grid --skip-ragas   # 200-row grid (slow on CPU rerank)
```

Uses `TFM_DATA_DIR=$(pwd)/tests/fixture/Data` by default.
