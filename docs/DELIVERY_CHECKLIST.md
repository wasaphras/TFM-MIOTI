# Delivery checklist (school hand-in)

## Repository contents

- [ ] All code under `Scripts/` and `webapp/` matches thesis description
- [ ] `requirements.txt` installs on Python 3.11+
- [ ] `.env.example` copied to `.env` locally (never commit `.env`)
- [ ] `docs/ACADEMIC_USE.md` - fill in author name and institution
- [ ] `docs/REPRODUCING_RESULTS.md` - maps committed CSVs to experiments
- [ ] `tests/fixture/Data/` committed (~32 MB, 10-doc Chroma ~2 MB each)

## Lean mode (10 docs, 2 GT, no download)

```bash
# 1. Python 3.11+ env active
pip install -r requirements.txt

# 2. Ollama (separate terminal)
ollama serve
ollama pull qwen3-embedding:4b
ollama pull llama3.2

# 3. Optional if default python is wrong
# export PYTHON=...   or conda activate Data

# 4. Fixture check (does NOT validate pip/Ollama)
./run_lean.sh --verify

# 5. Full lean pipeline (~1–2 h CPU rerank)
./run_lean.sh --skip-ragas   # or omit flag to include Ollama-only Ragas
```

Expected: `Lean verify OK`; smoke test ends with `8 passed, 0 failed` (Ragas optional; judge uses `--provider ollama`).

## Full mode (thesis corpus, local Data/)

```bash
export HF_TOKEN=...
./run_full.sh plan
./run_full.sh ingest
./run_full.sh index
# ... see README.md Steps 3-6 or run_full.sh help
```

## Common mistakes

| Mistake | Symptom |
|---------|---------|
| Running `./run_lean.sh` without pip/Ollama | `ModuleNotFoundError`, connection refused to Ollama |
| Treating `--verify` as full prereq check | Verify passes but full run fails on imports/Ollama |
| Ollama not serving | Embed/index/Ragas gen fails at `/api/embed` or chat |
| Copying `Data/chroma_chunk_*` into fixture | Fixture >1 GB per strategy |
| Missing `TFM_CATEGORIES_JSON` | Preprocess fails on fixture |
| GPU rerank on 6 GB card | OOM; use `RERANK_DEVICE=cpu` for smoke |
