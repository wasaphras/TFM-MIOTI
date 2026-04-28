# TFM -- MultiEURLEX RAG Evaluation

Python pipeline to download English EU-law documents from [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex), chunk them with **10 strategies**, embed each into its own **Chroma** vector database (via Ollama), generate **ground-truth** evaluation questions with an LLM, and score **20 retrieval variants** per chunk DB: **10 base** retrievers (dense, BM25, hybrid fusion) plus **10 cross-encoder reranked** counterparts (`*_ce_r50`, same fusion logic after reranking the candidate pool).

---

## How to run -- exact commands, step by step

> **All commands run from the project root (`TFM/`).** The steps must be executed **in order**.
>
> **What you are building:**
>
>
> | Concept                 | Count   | What it is                                                                 |
> | ----------------------- | ------- | -------------------------------------------------------------------------- |
> | Chunk databases         | **10**  | One Chroma vector DB per chunking strategy                               |
> | Ground-truth questions  | **100** | Example size: LLM-generated questions (`--n 100`)                        |
> | Retriever variants      | **20**  | 10 **base** IDs + the same 10 with **`_ce_r50`** cross-encoder rerank       |
>
>
> **Default grid** (`run_grid_eval` with no `--retrievers`): **10 × 20 × N** per-query retrieval calls (N = number of ground-truth rows). With N = **100**, that is **20 000** scored queries. `results_summary.csv` has **200 rows** (one per chunk strategy × retriever). `rank_breakdown_long.csv` has **20 000** rows (200 combos × 100 questions).
>
> **Base-only grid** (pass exactly the **10** base IDs via `--retrievers`, no `*_ce_r50`): **10 × 10 × N** calls; with N = 100 → **10 000** scored queries, **100** summary rows, **10 000** breakdown rows.

---

### Step 0 -- One-time prerequisites

```bash
# 1. Pull the two Ollama models
ollama pull qwen3-embedding:4b    # embedding model
ollama pull llama3.2               # LLM for ground-truth generation

# 2. Only if Data/train.jsonl does NOT exist yet (first time):
export HF_TOKEN=your_huggingface_token
```

---

### Step 1 -- Download the dataset

Downloads MultiEURLEX and writes `Data/train.jsonl` (1 000 English documents). **Skips automatically** if the file already exists.

```bash
python -m Scripts.data_extraction_load
```

**Creates:** `Data/train.jsonl`, `Data/dataset_info.json`

If you cloned the repo from GitHub, `train.jsonl` is usually **not** in the tree (it is gitignored as a large file). Step 1 is still required once per machine.

---

### Step 2 -- Build the 10 chunk databases

Chunks every document with 10 different strategies, writes one JSONL per strategy, then embeds each into its own Chroma directory via Ollama. **This is the slowest step.**

```bash
python -m Scripts.eval.build_chunk_indices --all
```

**Creates (10 of each):**


| File / directory                 | Example                                                  |
| -------------------------------- | -------------------------------------------------------- |
| `Data/chunks_<strategy>.jsonl`   | `Data/chunks_len_500_o50.jsonl`                          |
| `Data/chroma_chunk_<strategy>/`  | `Data/chroma_chunk_len_500_o50/`                         |
| `Data/eval_corpus_manifest.json` | One shared file listing which CELEX doc IDs are in scope |


The **10 chunking strategy IDs** are:

```
len_500_o50  len_1000_o100  len_1500_o150  len_2000_o0  len_2000_o200
para_nn_merge  line_n_merge  char_nn_only  rec_nn_priority  rec_legal_markers
```

The **20 retriever IDs** (defaults in `Scripts/eval/retrieval_strategies.py`) are the **10 bases** plus each with **`_ce_r50`**:

```
dense_sim_k10              dense_sim_k10_ce_r50
dense_mmr_k10              dense_mmr_k10_ce_r50
bm25_k10                   bm25_k10_ce_r50
hyb_rrf_k60                hyb_rrf_k60_ce_r50
hyb_rrf_k30                hyb_rrf_k30_ce_r50
hyb_rrf_fetch40            hyb_rrf_fetch40_ce_r50
hyb_weighted_norm          hyb_weighted_norm_ce_r50
hyb_weighted_dense_70     hyb_weighted_dense_70_ce_r50
hyb_interleave             hyb_interleave_ce_r50
hyb_fill_dense_then_bm25   hyb_fill_dense_then_bm25_ce_r50
```

For each `*_ce_r50` variant, the pipeline retrieves **`RETRIEVAL_CANDIDATE_K`** candidates (default **50** from `config.py` / env), reranks them with the cross-encoder, then keeps the top **`FINAL_K` = 10** for scoring—same final list length as the base retrievers.

**Resume behavior:** already-completed strategies are auto-skipped. Add `--force` to rebuild from scratch.

> **Quick-test variant** -- only use the first N documents (much faster, useful for verifying the pipeline works):
>
> ```bash
> python -m Scripts.eval.build_chunk_indices --all --limit 50
> ```
>
> If you change `--limit` later, you **must re-run Step 3** because the corpus scope changed.

---

### Step 3 -- Generate 100 ground-truth questions

The LLM (`llama3.2` via Ollama) reads snippets from the indexed documents and writes one evaluation question per snippet. `--n 100` means 100 accepted questions.

```bash
python -m Scripts.eval.ground_truth_generate --n 100
```

**Requires:** `Data/eval_corpus_manifest.json` from Step 2 (questions only reference documents that are actually in the databases).

**Creates:** `Data/ground_truth.jsonl` -- 100 rows, each containing `question`, `gold_snippet`, and `reference` (CELEX ID).

---

### Step 4 -- Run the evaluation grid (default: 10 × 20)

**Default** (omit `--retrievers`): runs **all 20** retrievers on each of the **10** chunk databases—includes every `*_ce_r50` reranked variant (needs the cross-encoder stack from `sentence-transformers` / PyTorch; **GPU recommended**).

```bash
python -m Scripts.eval.run_grid_eval
```

**Base-only (10 retrievers, no rerank)** — faster, less VRAM: pass exactly the **10** base IDs so cross-encoder cells are not scheduled:

```bash
python -m Scripts.eval.run_grid_eval \
  --retrievers \
    dense_sim_k10 \
    dense_mmr_k10 \
    bm25_k10 \
    hyb_rrf_k60 \
    hyb_rrf_k30 \
    hyb_rrf_fetch40 \
    hyb_weighted_norm \
    hyb_weighted_dense_70 \
    hyb_interleave \
    hyb_fill_dense_then_bm25
```

**Creates (in `Data/eval/`):**


| File                      | Content                                                                                                                                 |
| ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `results_summary.csv`     | **200 rows** if you ran all 20 retrievers (**100 rows** if you passed only the 10 bases): one row per chunk strategy × retriever, with hit rate, MRR, rank stats |
| `rank_breakdown_long.csv` | Per-query rank detail: **20 000** rows for 20 retrievers × 100 GT questions (or **10 000** for 10 bases × 100)                          |
| `hit_rate_pivot.csv`      | Wide table: chunk strategies as rows, one column per retriever you ran, cells = hit rates                                             |


**Resume / checkpoint:** Progress is saved after **every ground-truth query** to `Data/eval/eval_grid_checkpoint.json` (atomic write). If you stop with Ctrl+C, kill the process, or the machine crashes, re-run the **same** command (same ground-truth file, `--limit-queries`, `--chunk-strategies`, `--retrievers`). The run continues where it left off. If you change any of those inputs or edit `ground_truth.jsonl`, delete the checkpoint or pass `--no-resume` to start over. CSV reports are written only when the full grid finishes.

**Memory:** The eval loop loads one Chroma index per chunk strategy, then discards it (`gc.collect()`, Chroma cache release). BM25 needs the full `chunks_<strategy>.jsonl` in RAM for that strategy. For each chunk strategy, **all non–cross-encoder retrievers run first**, then cross-encoder `*_ce_r50` cells; the reranker is **unloaded from GPU** after those cells so the next strategy does not keep the model resident. Optional: set `EVAL_CUDA_EMPTY_CACHE=1` to call `torch.cuda.empty_cache()` after each strategy when using GPU rerankers. If you still hit CUDA OOM during rerank forward passes, lower `RERANK_PREDICT_BATCH_SIZE` (default `32`) or use a smaller `RERANK_MODEL`.

---

### Step 5 -- Top-10 four-eval pipeline (k=20, 100 GT questions)

This is a **separate** experiment from the default grid in Step 4. It runs a **pruned** set of `(chunk_strategy, retriever)` pairs (see `Scripts/eval/top10/pairs.py` → `SELECTED_PAIRS_EVAL1`) on **`Data/ground_truth.jsonl`** with **final list length 20**, metrics **Hit@3/5/10/20** + MRR in `results_summary.csv`, buckets **1–20 + miss**, and **`per_query_ranks.csv`** (one row per query × cell for rank movement analysis).

| Eval | Script | Idea |
| ---- | ------ | ---- |
| **1** | `Scripts.eval.top10.run_eval1_baseline` | Baseline: dedupe, retrieve/rerank to top 20. All pairs use `*_ce_r50` in the default grid. |
| **2** | `Scripts.eval.top10.run_eval2_neighbors` | Base retrieve → **neighbor** expansion (default offsets `-2,-1,1,2`; override with `--neighbor-offsets`; limit which seeds get neighbors with `--neighbor-seed-top N`) → dedupe → cross-encoder rerank → 20. |
| **3** | `Scripts.eval.top10.run_eval3_enhanced` | LLM rewrites each **original question** into a longer formal EU-law question; retrieve with the enhanced text; rerank → 20. |
| **4** | `Scripts.eval.top10.run_eval4_multiquery` | Enhanced + **2 variants**; base retrieval runs for **original question first**, then enhanced + variants (default), **union**, dedupe, cross-encoder rerank with the **original** question → 20. Pass `--no-original-in-retrieval` for the legacy 3-query-only pool. |

**Tuning helpers** (after a full or partial eval has written `results_summary.csv`):

```bash
# Dry-run commands for candidate_k × baseline vs neighbors; add --execute to run
python -m Scripts.eval.top10.run_tuning_sweeps candidate
python -m Scripts.eval.top10.run_tuning_sweeps neighbor

# Compare per-query ranks between two eval output directories
python -m Scripts.eval.top10.report_rank_comparison \
  Data/eval_top10/eval1_baseline Data/eval_top10/eval2_neighbors \
  --label-a baseline --label-b neighbors --out Data/eval_top10/rank_delta.csv

# Oracle: where does gold appear? (small --limit-queries recommended)
python -m Scripts.eval.top10.diagnose_oracle --limit-queries 20
```

Merge summaries (includes new `hit_at_*` columns when present):

```bash
python -m Scripts.eval.merge_top10_summaries
```

**One-time index / stats** (after Step 2, before evals):

```bash
python -m Scripts.eval.top10.neighbor_index --strategies \
  len_500_o50 len_1000_o100 len_1500_o150 len_2000_o200 rec_nn_priority
```

`Data/chunk_stats.json` (median chunk length → target word count for LLM prompts) is created automatically on first run of eval 3 or 4.

**Outputs** (gitignored by default):

- `Data/eval_top10/eval1_baseline/`, `eval2_neighbors/`, `eval3_enhanced/`, `eval4_multiquery/` — each with `results_summary.csv` (one row per pair; includes **hit_at_3**, **hit_at_5**, **hit_at_10**, **hit_at_20** when `final_k` ≥ 20), `per_query_ranks.csv`, `rank_breakdown_long.csv`, `hit_rate_pivot.csv`, and `checkpoint.json` while incomplete.
- `Data/enhanced_questions/<strategy>.jsonl` (+ optional `.checkpoint.json`)
- `Data/multi_query_questions/<strategy>.jsonl` (+ optional `.checkpoint.json`)
- `Data/neighbor_index/<strategy>.pkl`

**Resume:** Re-run the **same** command; progress is saved after **each** ground-truth question. Use `--no-resume` on one eval to discard **only** that eval’s checkpoint and CSVs.

**Run all four in order** (forwards extra CLI args such as `--limit-queries 5` to every step):

```bash
python -m Scripts.eval.top10.run_all
```

Or individually:

```bash
python -m Scripts.eval.top10.run_eval1_baseline
python -m Scripts.eval.top10.run_eval2_neighbors
python -m Scripts.eval.top10.run_eval3_enhanced
python -m Scripts.eval.top10.run_eval4_multiquery
```

Default `--ground-truth` is `Data/ground_truth.jsonl`. Requires Ollama for embeddings; evals 2–4 load the cross-encoder (`sentence-transformers`). Evals 3–4 require Ollama for `config.LLM_MODEL` (default `llama3.2`).

**Memory:** The shared engine drops retrieved document lists after each query, runs `gc.collect()` every 25 queries, unloads the cross-encoder and releases the Chroma process cache when **switching chunk strategy**, and cleans up again in a `finally` block on exit or Ctrl+C. Optional: `EVAL_CUDA_EMPTY_CACHE=1` to call `torch.cuda.empty_cache()` after those steps (same as the Step 4 grid).

**Phased pipeline (recommended on small GPU/RAM):** Split work so you never hold **LLM + full BM25 corpus + Chroma + cross-encoder** at once.

1. **LLM only** (eval 3/4): write `Data/enhanced_questions/` and `Data/multi_query_questions/` — no embeddings, no chunk JSONL in RAM beyond chunk_stats.

   ```bash
   python -m Scripts.eval.top10.materialize_llm_inputs --eval both
   ```

   Use `--eval eval3` or `--eval eval4` if you only need one tree. Same `--ground-truth`, `--limit-queries`, and `--pairs` as the evals. **Resume:** re-run the same command; completed question ids are skipped. If Ollama returns bad JSON or slows down, use `--llm-max-attempts 16 --retry-sleep-seconds 3`, or `--heuristic-fallback` for variants only (rule-based paraphrases; flagged in the log).

2. **Embedding prefetch** (all evals): store per-query candidate lists under `Data/eval_top10/prefetch/<eval_id>/` — Chroma + BM25 only; **no cross-encoder**; eval 3/4 read queries from disk (`require_*`) and do **not** call the LLM.

   ```bash
   python -m Scripts.eval.top10.run_eval1_baseline --prefetch-write
   python -m Scripts.eval.top10.run_eval2_neighbors --prefetch-write
   python -m Scripts.eval.top10.run_eval3_enhanced --prefetch-write
   python -m Scripts.eval.top10.run_eval4_multiquery --prefetch-write
   ```

   Or `python -m Scripts.eval.top10.run_all --prefetch-write` (forwards args to each step).

3. **Rerank / metrics** (no Chroma): cross-encoder only on GPU if you want.

   ```bash
   python -m Scripts.eval.top10.run_all --prefetch-read
   ```

`--no-resume-prefetch` deletes that eval’s prefetch subtree before a write. Padding to `final_k` in the read phase uses the **saved candidate pool** only (not a second embedding pass), which can differ slightly from live `_pad_to_k` when the pool is very small.

**Live eval (no prefetch):** Evals 3–4 still call the LLM during the main loop; use that only if you have enough RAM. JSONL indexes are cached in memory per file so rows are not re-read from disk on every question.

---

### Cheat sheet -- copy-paste the full pipeline

```bash
# Step 1: dataset
python -m Scripts.data_extraction_load

# Step 2: 10 chunk databases (chunk + embed)
python -m Scripts.eval.build_chunk_indices --all

# Step 3: 100 ground-truth questions
python -m Scripts.eval.ground_truth_generate --n 100

# Step 4a (full): 10 chunk strategies × 20 retrievers × 100 GT questions
python -m Scripts.eval.run_grid_eval

# Step 4b (lighter): 10 × 10 bases only — same GT file
python -m Scripts.eval.run_grid_eval \
  --retrievers dense_sim_k10 dense_mmr_k10 bm25_k10 \
    hyb_rrf_k60 hyb_rrf_k30 hyb_rrf_fetch40 \
    hyb_weighted_norm hyb_weighted_dense_70 \
    hyb_interleave hyb_fill_dense_then_bm25
```

### Smoke test -- quick verification with tiny corpus

```bash
python -m Scripts.eval.build_chunk_indices --all --limit 10
python -m Scripts.eval.ground_truth_generate --n 10
python -m Scripts.eval.run_grid_eval --limit-queries 10 \
  --retrievers dense_sim_k10 bm25_k10 hyb_rrf_k60
```

---

### Optional: interactive RAG (not part of the eval grid)

```bash
python -m Scripts.main --prompt "Your question" --limit 50
python run_api.py   # or: python -m Scripts.api
```

---

## Prerequisites


| Requirement                | Notes                                                                                                                                |
| -------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Conda env**              | Create one and install deps: `pip install -r requirements.txt` (installs `sentence-transformers` and PyTorch for eval reranking)     |
| **Ollama**                 | Pull embedding + chat models: see Step 0 above                                                                                       |
| **GPU (optional)**         | Recommended for the full 10×20 eval with default reranker `BAAI/bge-reranker-v2-m3`; CPU works but is slower                         |
| **HF_TOKEN**               | Hugging Face token (env var read by `config.py`). **Do not commit tokens.** Needed when Step 1 must download the dataset tarball if `Data/train.jsonl` is missing. The downloader also mentions `HUGGINGFACE_HUB_TOKEN` in its warning text; either can satisfy the Hub depending on library behavior, but this repo passes `HF_TOKEN` explicitly. |
| **Data/categories.json**   | EuroVOC id-to-labels map (required for `preprocess_for_rag`)                                                                          |


Default models in config (adjust if you use others):

- Embeddings: `qwen3-embedding:4b` -- `ollama pull qwen3-embedding:4b`
- LLM (GT questions + RAG answers): `llama3.2` -- `ollama pull llama3.2`

### Embedding vector size (dimensions)

`Scripts/config.py` sets `EMBEDDING_DIMENSIONS = 2560`, passed to Ollama's `/api/embed` as the `dimensions` field (Matryoshka / truncated width).

For `qwen3-embedding:4b` on current Ollama, the native maximum is 2560: requesting a larger `dimensions` still yields length 2560. Smaller values truncate the vector (cheaper index, less detail). `None` would omit `dimensions` and use the server default (still 2560 for this model).

---

## Corpus manifest contract

`Data/eval_corpus_manifest.json` lists the CELEX IDs of every document loaded when you ran `build_chunk_indices` (same scope as `train.jsonl` + optional `--limit`).

1. **build_chunk_indices** writes this file **after** loading the dataframe, **before** chunking.
2. **ground_truth_generate** **requires** the manifest and only samples documents whose `celex_id` is in the list.
3. **run_grid_eval** **requires** the manifest and aborts if any GT `reference` CELEX is missing (no silent all-zero eval from scope drift).

If you change `--limit`, re-run chunk index build (Step 2), then regenerate `ground_truth.jsonl` (Step 3).

---

## Directory map


| Path                                    | Role                                                            |
| --------------------------------------- | --------------------------------------------------------------- |
| `Data/train.jsonl`                      | English docs (celex_id, text, labels)                           |
| `Data/categories.json`                  | EuroVOC labels for `preprocess_for_rag`                         |
| `Data/eval_corpus_manifest.json`        | Eval corpus scope (CELEX list from last `build_chunk_indices`)  |
| `Data/chunks_<strategy>.jsonl`          | Chunks for BM25 + chunk_uid alignment                           |
| `Data/chroma_chunk_<strategy>/`         | Chroma persistence per chunking strategy                        |
| `Data/ground_truth.jsonl`               | Eval queries + gold snippet + reference CELEX                   |
| `Data/eval/*.csv`                       | Grid eval outputs (written when a full grid finishes)           |
| `Scripts/config.py`                     | Paths, model names, collection name, `DOC_LIMIT` (default 1000 docs) |
| `Scripts/data_extraction_load.py`       | HF download -> `train.jsonl`                                    |
| `Scripts/preprocess.py`                 | Adds `labels_en` from categories                                |
| `Scripts/chunking.py`                   | Single-strategy chunking for `main`/legacy                      |
| `Scripts/chunking_strategies.py`        | **10** eval chunking strategies                                 |
| `Scripts/embeddings_chromadb.py`        | Ollama embed -> Chroma                                          |
| `Scripts/retriever.py`                  | Dense retrieve + RAG answer                                     |
| `Scripts/main.py`                       | End-to-end demo pipeline                                        |
| `Scripts/api.py`                        | FastAPI `/chat`                                                 |
| `Scripts/eval/build_chunk_indices.py`   | Chunk all strategies, embed, resume-safe                        |
| `Scripts/eval/ground_truth_generate.py` | LLM questions from in-corpus snippets                           |
| `Scripts/eval/retrieval_strategies.py`  | **20** retrievers: 10 baselines + 10 `*_ce_r50` rerank variants |
| `Scripts/eval/rerank_cross_encoder.py`  | Cross-encoder rerank (`RERANK_MODEL` in config)                 |
| `Scripts/eval/metrics.py`               | Hit rate, MRR, rank buckets                                     |
| `Scripts/eval/run_grid_eval.py`         | Full grid + CSV outputs                                         |

**Clones and Git:** Large generated paths are listed in `.gitignore` (including `Data/train.jsonl`, `Data/chunks_*.jsonl`, every `Data/chroma_*` directory, and `Data/eval/`). A fresh clone has the **scripts and small metadata** in `Data/`; run **Step 1** and **Step 2** locally to recreate `train.jsonl`, chunk JSONLs, and Chroma before ground truth or eval.

---

## CLI entry points


| Command                                                                  | Purpose                                                |
| ------------------------------------------------------------------------ | ------------------------------------------------------ |
| `python -m Scripts.data_extraction_load`                                 | Ensure `train.jsonl` exists (download if needed)       |
| `python -m Scripts.preprocess`                                           | Smoke-test preprocess (loads data via extraction main) |
| `python -m Scripts.eval.build_chunk_indices --all`                       | Build all strategies (+ manifest)                      |
| `python -m Scripts.eval.build_chunk_indices --only-strategy len_2000_o0` | Single strategy                                        |
| `python -m Scripts.eval.build_chunk_indices --all --force`               | Full rebuild JSONL + Chroma (no skip)                  |
| `python -m Scripts.eval.ground_truth_generate --n 100`                   | Write `ground_truth.jsonl` (100 questions)             |
| `python -m Scripts.eval.run_grid_eval`                                   | Run grid (**default: all 20 retrievers**); writes `Data/eval/*.csv` when complete |
| `python -m Scripts.eval.run_grid_eval --limit-queries 100`               | Use first 100 rows of GT (if file has more)            |
| `python -m Scripts.main`                                                 | RAG answer one prompt                                  |
| `python -m Scripts.api`                                                  | Start API server                                       |
| `python -m Scripts.embeddings_chromadb`                                  | Dev: chunk + embed default Chroma                      |
| `python -m Scripts.chunking`                                             | Dev: default chunker -> `chunks.jsonl`                 |


Common flags:

- **build_chunk_indices**: `--limit N`, `--force`, `--all`, `--only-strategy <id>`
- **ground_truth_generate**: `--n`, `--seed`, `--min-doc-chars`, `--snippet-min-chars`, `--snippet-max-chars`, `--manifest`, `--out`
- **run_grid_eval**: `--ground-truth`, `--out`, `--manifest`, `--retrievers` (subset; default = all 20), `--chunk-strategies`, `--limit-queries`, `--no-resume`, `--checkpoint`

---

## Troubleshooting

### All hit rates are 0.0

Usually **GT CELEX IDs are not in the indexed corpus**. Fix:

1. Run `build_chunk_indices` with the intended `--limit` (or full corpus).
2. Regenerate GT: `python -m Scripts.eval.ground_truth_generate --n 100`
3. Run `run_grid_eval` again.

If GT still references docs outside the manifest, eval **exits with an error** listing missing CELEX IDs.

### `Missing eval_corpus_manifest.json`

Run `build_chunk_indices` at least once for your current `train.jsonl` scope. GT generation and eval both require the manifest.

### Resume vs `--force`

- **Default (no `--force`)**: For each strategy, if `chunks_<id>.jsonl` line count equals Chroma collection count, that strategy is **skipped**. If embedding stopped early, the run **resumes** from the existing vector count. Regenerating the JSONL for a strategy **clears** that strategy's Chroma so vectors stay aligned with chunk IDs.
- **`--force`**: Deletes strategy Chroma dirs and rebuilds JSONL + embeddings for selected strategies.

### Memory (large corpora)

- **Chunking**: `build_chunk_indices` writes each strategy's JSONL **incrementally** (per document), so the full chunk list is never held in RAM.
- **Embedding**: Vectors are built by **streaming** the JSONL in `EMBEDDING_BATCH_SIZE` batches; Chroma is flushed every `CHROMA_WRITE_THRESHOLD` rows (512), each flush in `CHROMA_ADD_SUBBATCH_SIZE` (128) slices with retries to avoid HNSW compaction errors.
- The pipeline loads the full preprocessed dataframe once for all strategies; use `--limit` or `--only-strategy` to lower that footprint. After each strategy, `gc.collect()` runs.

### Embedding context / chunk size

All eval strategies cap segments at **2000 characters** (with recursive splits for oversized pieces). `char_nn_only` post-splits any `CharacterTextSplitter` overflow so nothing exceeds the cap.

### Hugging Face download fails

Set `export HF_TOKEN=...` and ensure you accept the dataset terms on Hugging Face if required.

### Reranker model download or OOM

The default `BAAI/bge-reranker-v2-m3` is downloaded from Hugging Face on first eval use. If you run out of memory, set a lighter model:

```bash
export RERANK_MODEL=BAAI/bge-reranker-base
# or
export RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

Optional: `export RETRIEVAL_CANDIDATE_K=40` to reduce candidate pool size.

### Chroma `InternalError` / HNSW compaction during `build_chunk_indices`

If you see errors like **Failed to apply logs to the hnsw segment writer**: (1) ensure you are on a recent `chromadb`; (2) delete the `Data/chroma_chunk_<strategy>/` directory for the failing strategy and **resume** (JSONL is kept; only Chroma is rebuilt). The embedder uses small sub-batches and retries to reduce this class of failure.

---

## Environment variables


| Variable                   | Used for                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `HF_TOKEN`                 | `huggingface_hub` download in `data_extraction_load`                                               |
| `RERANK_MODEL`             | Cross-encoder checkpoint ID (default: `BAAI/bge-reranker-v2-m3`)                                   |
| `RERANK_DEVICE`            | `cpu` or `cuda` / `cuda:0` to pin the reranker; if unset, tries CUDA then falls back to CPU on OOM |
| `RERANK_PREDICT_BATCH_SIZE` | Batch size for cross-encoder `predict` (default `32`; lower reduces peak VRAM during rerank)      |
| `RETRIEVAL_CANDIDATE_K`    | First-stage pool size before rerank (default `50`)                                                 |
| `RERANK_PASSAGE_MAX_CHARS` | Truncate chunk text passed to the reranker (default `8000`)                                        |
| `EVAL_CUDA_EMPTY_CACHE`    | Set to `1` / `true` to call `torch.cuda.empty_cache()` after each chunk strategy in the eval grid |


Model names stay in `Scripts/config.py` (no secrets).