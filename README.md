# TFM -- MultiEURLEX RAG Evaluation

Python pipeline to download English EU-law documents from [MultiEURLEX](https://huggingface.co/datasets/coastalcph/multi_eurlex), chunk them with **10 strategies**, embed each into its own **Chroma** vector database (via Ollama), generate **ground-truth** evaluation questions with an LLM, and score **20 retrieval variants** per chunk DB: **10 base** retrievers (dense, BM25, hybrid fusion) plus **10 cross-encoder reranked** counterparts (`*_ce_r50`, same fusion logic after reranking the candidate pool).

**Detailed reference:** for every script module, dedup/top-10 tracks, and `Data/` layout, see [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

---

## Two ways to run

| Mode | Corpus | Ground truth | Data directory | Entry point |
|------|--------|--------------|----------------|-------------|
| **Lean** | **10** docs (in git) | **2** questions | `tests/fixture/Data/` | [`./run_lean.sh`](run_lean.sh) |
| **Full** | **~55 000** docs (download) | **100** questions (typical) | `Data/` (gitignored) | [`./run_full.sh`](run_full.sh) |

**Lean** is for clones, reviewers, and CI: prebuilt Chroma (~2 MB per strategy), no Hugging Face download.  
**Full** is for thesis-scale results: download MultiEURLEX, build indices locally, run the experiments in README Steps 1-6.

```bash
# One-time setup (both modes)
pip install -r requirements.txt
ollama pull qwen3-embedding:4b
ollama pull llama3.2

# Lean: verify the whole stack on the small fixture (~1-2 h on CPU rerank)
./run_lean.sh --skip-ragas
./run_lean.sh --verify          # seconds: fixture files + sizes only

# Full: one step at a time (see also ./run_full.sh plan)
export HF_TOKEN=your_token      # first download only
./run_full.sh ingest
./run_full.sh index             # slow
./run_full.sh ground-truth
./run_full.sh grid
```

Lean mode sets `TFM_DATA_DIR` to the fixture. Full mode uses default `Data/` (unset `TFM_DATA_DIR`). See [docs/DELIVERY_CHECKLIST.md](docs/DELIVERY_CHECKLIST.md).

---

## How to run -- exact commands, step by step (full mode)

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

Downloads MultiEURLEX and writes `Data/train.jsonl` (up to `DOC_LIMIT` English documents, default **55 000**). **Skips automatically** if the file already exists.

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

**Optional — top-5 hybrid slice:** [`Scripts/eval/run_top5_eval.py`](Scripts/eval/run_top5_eval.py) runs five `hyb_*_ce_r50` retrievers on **`len_1000_o100` only**, with a dedicated ground-truth file (see `--ground-truth` default in that module). Same checkpoint semantics as Step 4; details in [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md).

**Resume / checkpoint:** Progress is saved after **every ground-truth query** to `Data/eval/eval_grid_checkpoint.json` (atomic write). If you stop with Ctrl+C, kill the process, or the machine crashes, re-run the **same** command (same ground-truth file, `--limit-queries`, `--chunk-strategies`, `--retrievers`). The run continues where it left off. If you change any of those inputs or edit `ground_truth.jsonl`, delete the checkpoint or pass `--no-resume` to start over. CSV reports are written only when the full grid finishes.

**Memory:** The eval loop loads one Chroma index per chunk strategy, then discards it (`gc.collect()`, Chroma cache release). BM25 needs the full `chunks_<strategy>.jsonl` in RAM for that strategy. For each chunk strategy, **all non–cross-encoder retrievers run first**, then cross-encoder `*_ce_r50` cells; the reranker is **unloaded from GPU** after those cells so the next strategy does not keep the model resident. Optional: set `EVAL_CUDA_EMPTY_CACHE=1` to call `torch.cuda.empty_cache()` after each strategy when using GPU rerankers. If you still hit CUDA OOM during rerank forward passes, lower `RERANK_PREDICT_BATCH_SIZE` (default `32`) or use a smaller `RERANK_MODEL`.

---

### Step 5 -- Top-10 four-eval pipeline (k=20, 100 GT questions)

This is a **separate** experiment from the default grid in Step 4. It runs **10 hand-picked** `(chunk_strategy, retriever)` pairs on **`Data/ground_truth.jsonl`** (100 questions) with **final list length 20** and metrics buckets **1–20 + miss**.

| Eval | Script | Idea |
| ---- | ------ | ---- |
| **1** | `Scripts.eval.top10.run_eval1_baseline` | Baseline: dedupe, then retrieve/rerank to top 20. The last pair uses **base** `hyb_interleave` only (no cross-encoder); all others use `*_ce_r50`. |
| **2** | `Scripts.eval.top10.run_eval2_neighbors` | Base retrieve (breadth 100) → expand each hit with **±1, ±2** neighbor chunks in the same CELEX (sidecar index) → dedupe → cross-encoder rerank → 20. **All 10** pairs use `*_ce_r50` (including upgraded `hyb_interleave_ce_r50`). |
| **3** | `Scripts.eval.top10.run_eval3_enhanced` | LLM rewrites each **original question** (gold snippet **not** shown) into a longer formal EU-law question; retrieve with the enhanced text; dedupe; rerank → 20. |
| **4** | `Scripts.eval.top10.run_eval4_multiquery` | From the enhanced question, the LLM emits **2 variants**; run base retrieval **independently** for enhanced + 2 variants (`--per-query-candidate-k` default **80**), **union**, dedupe by `chunk_uid`, cross-encoder rerank using the **original** question → 20. |

**One-time index / stats** (after Step 2, before evals):

```bash
python -m Scripts.eval.top10.neighbor_index --strategies \
  len_500_o50 len_1000_o100 len_1500_o150 len_2000_o200 rec_nn_priority
```

`Data/chunk_stats.json` (median chunk length → target word count for LLM prompts) is created automatically on first run of eval 3 or 4.

**Outputs** (gitignored by default):

- `Data/eval_top10/eval1_baseline/`, `eval2_neighbors/`, `eval3_enhanced/`, `eval4_multiquery/` — each with `results_summary.csv` (10 rows), `rank_breakdown_long.csv` (10 × 100 rank buckets), `hit_rate_pivot.csv`, and `checkpoint.json` while incomplete.
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

### Step 6 -- Dedup corpus and top-10 dedup eval (optional parallel track)

This is **not** Steps 4–5. It uses a **smaller curated document list** (`Data/train_dedup.jsonl`) and separate chunk/Chroma directories (`chunks_dedup_*`, `chroma_chunk_dedup_*`), then runs the **same style of top-10 / k=20** retrieval experiments with checkpoint + prefetch support. Full detail: [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) (§4, §8).

**Requirements:** Complete **Step 2** for the chunk strategies involved (dedup build **streams** existing `chunks_*.jsonl` / chroma—it does **not** re-call Ollama). You still need **`Data/train_dedup.jsonl`** and **`HF_TOKEN`** as for the rest of the pipeline.

1. Filter indices to dedup CELEX IDs (strategies appearing in curated pairs):

   ```bash
   python -m Scripts.eval.build_chunk_indices_dedup --top10
   ```

2. Generate dedup ground truth (answers + validation metadata; writes `Data/ground_truth_dedup_top10_100.jsonl`):

   ```bash
   python -m Scripts.eval.ground_truth_generate_dedup
   ```

3. Neighbor index for dedup eval 2 (strategies in curated pairs only):

   ```bash
   python -m Scripts.eval.top10.neighbor_index_dedup --top10
   ```

4. **Recommended:** orchestrate GPU phases (conda env `Data` if present, prefetch-write then prefetch-read, optional Ollama stop between phases):

   ```bash
   bash Scripts/eval/run_dedup_top10_evals.sh
   ```

   Or run [`Scripts.eval.top10.run_eval1_baseline_dedup`](Scripts/eval/top10/run_eval1_baseline_dedup.py) / [`run_eval2_neighbors_dedup`](Scripts/eval/top10/run_eval2_neighbors_dedup.py) manually with `--prefetch-write` / `--prefetch-read` exactly like Step 5’s phased pattern.

5. Merge baseline + neighbors summaries:

   ```bash
   python -m Scripts.eval.merge_top10_summaries_dedup
   ```

**Ragas:** On this dedup track, constrained RAG + Ragas metrics are documented under **“Ragas evaluation”** later in this README (same prerequisites as dedup prefetch `eval1_baseline`).


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

### Lean mode details

[`./run_lean.sh`](run_lean.sh) sets `TFM_DATA_DIR=tests/fixture/Data` and runs the same eval stages as full mode on **10 documents** and **2** ground-truth rows. See [`tests/fixture/README.md`](tests/fixture/README.md).

| Stage | Lean default |
|-------|----------------|
| Grid | 3 retrievers x 10 strategies x 2 GT (`--full-grid` for 20 retrievers, 200 summary rows) |
| Top-10 | Evals 1-4 |
| Dedup | Eval1 + eval2 + merge (8-doc dedup subset) |
| Ragas | On unless `--skip-ragas` |

Rebuild fixture: `python tests/build_fixture.py` (needs local `Data/train.jsonl` + Ollama).

---

### Ragas evaluation (two-phase LLM triad on the dedup corpus)

This complements hit-rate / MRR with **Ragas** scores on the retrieval + generation pipeline. Defaults target the **best cell** observed in dedup baseline eval (`len_500_o50`, `hyb_fill_dense_then_bm25_ce_r50`, top **20** after cross-encoder rerank, candidate breadth 100)—see `Data/eval_top10_dedup/eval1_baseline/results_summary.csv`.

**Prerequisites:**

- Dedup corpus artifacts: `Data/chunks_dedup_len_500_o50.jsonl`, `Data/chroma_chunk_dedup_len_500_o50/`, plus `Scripts.eval.top10.dedup_paths` paths.
- Embedding prefetch aligned with **`eval1_baseline`** speeds up Step 1: `Data/eval_top10_dedup/prefetch/eval1_baseline/` (optional; falls back to live Chroma + rerank).

**Frozen defaults (generation):** `--chunk-strategy len_500_o50`, `--retriever hyb_fill_dense_then_bm25_ce_r50`, `--final-k 20`, `--candidate-k 100`, ground truth `Data/ground_truth_dedup_top10_100.jsonl`.

1. **Step A — Replay retrieval and write constrained RAG answers** (writes `rag_responses_v2` JSONL including `ground_truth_answer` / `reference_contexts`):

   ```bash
   python -m Scripts.eval.llm_triad.generate_rag_responses
   ```

2. **Step B — Ragas judge** (**default `--metrics local`**, **default `--provider auto`** → Gemini when `GEMINI_API_KEY` is present in `.env`, else Ollama). Judges + embeddings route through Gemini API + `GEMINI_MODEL` / `GEMINI_EMBEDDING_MODEL` (defaults in script) when using Gemini:

   ```bash
   pip install langchain-google-genai python-dotenv   # listed in requirements.txt
   ```

   Put in project root **`.env`** (already gitignored), plain values:

   ```
   GEMINI_API_KEY=<your-api-key>
   GEMINI_MODEL=gemini-2.0-flash
   GEMINI_EMBEDDING_MODEL=gemini-embedding-001
   ```

   Then run Step B as usual (no extra flags needed if `GEMINI_API_KEY` is set):

   ```bash
   python -m Scripts.eval.llm_triad.judge_rag_triad \
     --in Data/eval_top10_dedup/llm_triad_len500_hyb_fill/rag_responses.jsonl \
     --out Data/eval_top10_dedup/llm_triad_len500_hyb_fill/ragas_scores.jsonl \
     --ground-truth Data/ground_truth_dedup_top10_100.jsonl
   ```

   Force Ollama for Ragas explicitly: **`--provider ollama`**. Force Gemini: **`--provider gemini`** (fails fast if key missing).

   **`--ground-truth`:** optional for v2 rows that already carry `ground_truth_answer` and non-empty `reference_contexts`, but **recommended** for lineage and for `local` if any row lacks those fields (the script skips with `ragas_skip_reason`). For `--metrics full`, it also backfills answers for reference-based LLM metrics.

**Interpretation caveats:** The field `ground_truth.answer` (propagated as `ground_truth_answer`) is **LLM-authored** when the dedup ground-truth file was built; scores that compare to it (e.g. `answer_similarity`, `answer_correctness` in **full** mode) are relative to that proxy, not independent human labels. Ragas reproducibility depends on **`ragas_provider` + model IDs** logged in `.meta.json` (Gemini chat + embedding IDs or Ollama model names); keep them stable across thesis runs.

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
| **Conda / Python**       | Recommend **Python 3.11+**. Install deps: `pip install -r requirements.txt` (includes LangChain stack, **chromadb**, **rank_bm25**, **sentence-transformers** → typically pulls **PyTorch** for CUDA/CPU reranking, **ragas==0.4.3**, **rapidfuzz** for non-LLM Ragas distances, **datasets**, **python-dotenv**, **langchain-google-genai**) |
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

**Dedup manifest:** `Data/eval_corpus_manifest_dedup.json` is the CELEX scope written by `build_chunk_indices_dedup`; `ground_truth_generate_dedup`, dedup top-10 evals, and `llm_triad` default paths must stay consistent with that file.

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
| `Data/train_dedup.jsonl`                | Dedup training subset (CELEX-filtered docs)                    |
| `Data/eval_corpus_manifest_dedup.json` | CELEX scope after `build_chunk_indices_dedup`                 |
| `Data/chunks_dedup_<strategy>.jsonl`    | Dedup chunk lines (subset of `chunks_<strategy>.jsonl`)        |
| `Data/chroma_chunk_dedup_<strategy>/`    | Dedup Chroma DBs                                              |
| `Data/ground_truth_dedup_top10_100.jsonl` | Dedup eval (+ LLM `answer` for Ragas lineage)                |
| `Data/neighbor_index_dedup/`            | Pickled neighbor indices for dedup eval 2                      |
| `Data/eval_top10/`                      | Standard top-10 four-eval outputs + prefetch                  |
| `Data/eval_top10_dedup/`                | Dedup top-10 eval (eval1/eval2) + prefetch + merges          |
| `Data/eval/*.csv`                       | Grid eval outputs (written when a full grid finishes)           |
| `Scripts/config.py`                     | Paths, model names, `DOC_LIMIT` (default 55000), `TFM_DATA_DIR` override |
| `Scripts/data_extraction_load.py`       | HF download -> `train.jsonl`                                    |
| `Scripts/preprocess.py`                 | Adds `labels_en` from categories                                |
| `Scripts/chunking.py`                   | Single-strategy chunking for `main`/legacy                      |
| `Scripts/chunking_strategies.py`        | **10** eval chunking strategies                                 |
| `Scripts/embeddings_chromadb.py`        | Ollama embed -> Chroma                                          |
| `Scripts/retriever.py`                  | Dense retrieve + RAG answer                                     |
| `Scripts/main.py`                       | End-to-end demo pipeline                                        |
| `Scripts/api.py`                        | FastAPI `/chat/stream` (SSE) + `/health`                          |
| `Scripts/rag_chat_service.py`           | Dedup RAG retrieval + Gemini streaming for webapp                 |
| `webapp/`                               | Vite + React chat UI                                            |
| `Scripts/eval/build_chunk_indices_dedup.py` | Dedup index filter (streams standard chunk files → dedup paths) |
| `Scripts/eval/ground_truth_generate_dedup.py` | Dedup ground-truth builder                          |
| `Scripts/eval/merge_top10_summaries_dedup.py` | Merge dedup baseline + neighbors summaries           |
| `Scripts/eval/ground_truth_generate.py` | LLM questions from in-corpus snippets                           |
| `Scripts/eval/retrieval_strategies.py`  | **20** retrievers: 10 baselines + 10 `*_ce_r50` rerank variants |
| `Scripts/eval/rerank_cross_encoder.py`  | Cross-encoder rerank (`RERANK_MODEL` in config)                 |
| `Scripts/eval/metrics.py`               | Hit rate, MRR, rank buckets                                     |
| `Scripts/eval/run_grid_eval.py`         | Full grid + CSV outputs                                         |
| `Scripts/eval/llm_triad/`               | Rag replay + Ragas scoring (`generate_rag_responses`, `judge_rag_triad`) |

**Clones and Git:** Heavy paths under `Data/` are gitignored (`train*.jsonl`, `chunks_*.jsonl`, `chroma_chunk_*`, etc.). A fresh clone includes **scripts**, small metadata under `Data/`, committed **eval CSVs**, and the **smoke fixture** under `tests/fixture/Data/`. For full thesis runs, execute Steps **1–2** locally; use `bash tests/run_smoke_test.sh` to verify the stack without downloading the full corpus.

**Reproducing committed thesis tables:** see [`docs/REPRODUCING_RESULTS.md`](docs/REPRODUCING_RESULTS.md).
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
| `python -m Scripts.api`                                                  | Start RAG chat API (dedup stack + Gemini streaming)    |
| `python -m Scripts.embeddings_chromadb`                                  | Dev: chunk + embed default Chroma                      |
| `python -m Scripts.chunking`                                             | Dev: default chunker -> `chunks.jsonl`                 |
| `python -m Scripts.eval.build_chunk_indices_dedup --top10`               | Dedup chunk JSONLs + Chroma (filter from standard artifacts) |
| `python -m Scripts.eval.ground_truth_generate_dedup`                     | Dedup ground truth JSONL (100-row style)                       |
| `python -m Scripts.eval.top10.neighbor_index_dedup --top10`           | Dedup neighbor index PKLs for paired strategies                |
| `python -m Scripts.eval.merge_top10_summaries_dedup`                   | Merge `eval_top10_dedup` baseline + neighbors CSVs           |
| `bash Scripts/eval/run_dedup_top10_evals.sh`                            | Scripted dedup eval1/eval2 with prefetch splits                |
Common flags:

- **build_chunk_indices**: `--limit N`, `--force`, `--all`, `--only-strategy <id>`
- **ground_truth_generate**: `--n`, `--seed`, `--min-doc-chars`, `--snippet-min-chars`, `--snippet-max-chars`, `--manifest`, `--out`
- **run_grid_eval**: `--ground-truth`, `--out`, `--manifest`, `--retrievers` (subset; default = all 20), `--chunk-strategies`, `--limit-queries`, `--no-resume`, `--checkpoint`
- **build_chunk_indices_dedup**: `--top10` (subset to chunk strategies in curated pairs) or `--strategies` list
- **judge_rag_triad**: `--provider auto|ollama|gemini` (auto = gemini when `GEMINI_API_KEY` is set), `--gemini-model` / `--gemini-embedding-model`, `--metrics local|minimal|full` (default **local**), `--context-max-chars`, `--ground-truth` (recommended), `--ragas-timeout` (default 900s), `--ragas-max-workers` (default **1**), `--batch-size` (default 1 row per `evaluate()`), `--ollama-num-ctx` / `--ollama-num-predict` / `--ollama-plain-output` (Ollama only), `--resume`

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

### Ragas judge: `RagasOutputParserException`, `fix_output_format failed to parse`

This happens only on **`minimal`/`full`** (faithfulness, context precision/recall, answer relevancy, etc.). The judge LLM must return **strict JSON** for Ragas prompts; smaller Ollama chats often reply with prose or **truncated JSON**.

**Prefer `--metrics local`** (default): no judge chat—only embeddings + lexical context overlap (`rapidfuzz`). If you stay on **`full`/`minimal`**: **`--provider gemini`** avoids many Ollama JSON parse failures. On Ollama only: omit `--ollama-plain-output`, widen **`--ollama-num-ctx`**, **`--ollama-num-predict`**, **`--ragas-timeout`**, or use a **`--judge-model`** that follows JSON reliably.

### Chroma `InternalError` / HNSW compaction during `build_chunk_indices`

If you see errors like **Failed to apply logs to the hnsw segment writer**: (1) ensure you are on a recent `chromadb`; (2) delete the `Data/chroma_chunk_<strategy>/` directory for the failing strategy and **resume** (JSONL is kept; only Chroma is rebuilt). The embedder uses small sub-batches and retries to reduce this class of failure.

---

## Webapp (Vite + React chat UI)

ChatGPT-style UI in [`webapp/`](webapp/) against the **dedup eval winner**: `len_500_o50` + `hyb_fill_dense_then_bm25_ce_r50` (k=20, candidate 100), Ollama `qwen3-embedding:4b` for retrieval, **Gemini** for streamed answers.

### Prerequisites

1. Steps **1–2** and dedup indices: `python -m Scripts.eval.build_chunk_indices_dedup --top10` (needs `Data/chroma_chunk_dedup_len_500_o50/` and `Data/chunks_dedup_len_500_o50.jsonl`).
2. `ollama pull qwen3-embedding:4b` (Ollama running).
3. Copy [`.env.example`](.env.example) to `.env` at the project root and set `GEMINI_API_KEY` (never commit `.env`).

### Run (three services: Ollama + API + UI)

You need **both** Ollama (embeddings at query time) **and** the FastAPI backend (retrieval + Gemini). `ollama serve` alone will still show **Failed to fetch** in the browser.

**Terminal 1 — Ollama** (if not already running):

```bash
ollama serve
```

**Terminal 2 — API** (from project root; use the conda env that has `requirements.txt` installed):

```bash
conda run -n Data --no-capture-output python -m Scripts.api
# or: python -m Scripts.api   # when your active env already has fastapi, chromadb, etc.
```

Wait until you see `RAG chat service ready` and `Uvicorn running on http://0.0.0.0:8000`. Check: `curl http://localhost:8000/health` → `{"status":"ok"}`.

**Terminal 3 — UI:**

```bash
cd webapp
cp .env.example .env   # optional: VITE_API_BASE_URL=http://localhost:8000
npm install
npm run dev
```

Open `http://localhost:5173`. Answers render as markdown; **Sources** are grouped by CELEX document with EUR-Lex links and expandable full-document previews, plus per-chunk excerpts.

**`POST /chat/stream` body:** `{ "query": "...", "history": [{ "user": "...", "assistant": "...", "context_chunks": [{ "chunk_uid", "celex_id", "categories_en", "text" }] }] }`. Retrieval uses `query` only; prior turns (user + cited chunks + assistant) are injected before the LLM. SSE events: `sources` (document-grouped JSON), `token`, `done` (`used_chunk_uids`, `context_chunks`), `error`.

### Verify

| Check | Expected |
| ----- | -------- |
| `GET http://localhost:8000/health` | `{"status":"ok"}` |
| Ask a legislation question | Streamed markdown answer; sources grouped by CELEX with `eurlex_url` |
| Missing `GEMINI_API_KEY` | SSE `error` event pointing to `.env` |
| Missing dedup Chroma | API fails at startup with path + `build_chunk_indices_dedup` hint |

Optional env: `RAG_CHUNK_STRATEGY`, `RAG_RETRIEVER`, `RAG_FINAL_K`, `RAG_CANDIDATE_K`, `GEMINI_MODEL`, `CORS_ORIGINS` (see `.env.example`).

### Webapp: "Failed to fetch"

| Cause | Fix |
| ----- | --- |
| API not running | Start `python -m Scripts.api` (see above); verify `curl http://localhost:8000/health` |
| Wrong Python env | `ModuleNotFoundError: fastapi` → use `conda run -n Data python -m Scripts.api` |
| Wrong `VITE_API_BASE_URL` | `webapp/.env` should be `http://localhost:8000` (restart `npm run dev` after edits) |
| GPU OOM on first question | Add to root `.env`: `RERANK_DEVICE=cpu` and restart API |

---

## Environment variables


| Variable                   | Used for                                                                                           |
| -------------------------- | -------------------------------------------------------------------------------------------------- |
| `TFM_DATA_DIR`             | Override data root (default `<repo>/Data`; smoke test uses `tests/fixture/Data`)                  |
| `TFM_CATEGORIES_JSON`      | Path to `categories.json` (default `<repo>/Data/categories.json`)                                 |
| `TFM_DOC_LIMIT`            | Cap when sampling MultiEURLEX on first download (default `55000`)                                 |
| `HF_TOKEN`                 | `huggingface_hub` download in `data_extraction_load`                                               |
| `GEMINI_API_KEY`           | Loaded from project `.env`; **webapp chat API** + `judge_rag_triad` when `--provider` is gemini/auto |
| `GEMINI_MODEL`             | Gemini chat for webapp + Ragas judge; default `gemini-2.0-flash` if unset |
| `RAG_CHUNK_STRATEGY`       | Webapp retrieval chunk id (default `len_500_o50`) |
| `RAG_RETRIEVER`            | Webapp retriever id (default `hyb_fill_dense_then_bm25_ce_r50`) |
| `RAG_FINAL_K` / `RAG_CANDIDATE_K` | Webapp top-k after rerank / candidate pool (default 20 / 100) |
| `CORS_ORIGINS`             | Comma-separated origins for FastAPI CORS (default `http://localhost:5173`) |
| `GEMINI_EMBEDDING_MODEL`    | Gemini embedding ID (local + embedding metrics); default `gemini-embedding-001` if unset |
| `RERANK_MODEL`             | Cross-encoder checkpoint ID (default: `BAAI/bge-reranker-v2-m3`)                                   |
| `RERANK_DEVICE`            | `cpu` or `cuda` / `cuda:0` to pin the reranker; if unset, tries CUDA then falls back to CPU on OOM |
| `RERANK_PREDICT_BATCH_SIZE` | Batch size for cross-encoder `predict` (default `32`; lower reduces peak VRAM during rerank)      |
| `RETRIEVAL_CANDIDATE_K`    | First-stage pool size before rerank (default `50`)                                                 |
| `RERANK_PASSAGE_MAX_CHARS` | Truncate chunk text passed to the reranker (default `8000`)                                        |
| `EVAL_CUDA_EMPTY_CACHE`    | Set to `1` / `true` to call `torch.cuda.empty_cache()` after each chunk strategy in the eval grid |
| `DEDUP_EVAL_OLLAMA_STOP`    | Dedup bash driver: optional `ollama stop` after prefetch-write (`run_dedup_top10_evals.sh`) |
| `DEDUP_EVAL_RESET` / `--reset` | Wipe dedup prefetch + eval checkpoints (see shell script header) |
| `DEDUP_CONDA_ENV`           | Override conda env name (default **`Data`**) used by dedup bash driver |
| `DEDUP_NO_CONDA`            | Set `1` to use plain `python` on PATH in dedup bash driver           |


Model names stay in `Scripts/config.py` (no secrets).