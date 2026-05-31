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
pip install -r requirements.txt
ollama pull qwen3-embedding:4b
./run_lean.sh --verify
./run_lean.sh --skip-ragas
```

Expected: `Lean verify OK`; smoke test ends with `8 passed, 0 failed` (Ragas optional).

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
| Copying `Data/chroma_chunk_*` into fixture | Fixture >1 GB per strategy |
| Missing `TFM_CATEGORIES_JSON` | Preprocess fails on fixture |
| GPU rerank on 6 GB card | OOM; use `RERANK_DEVICE=cpu` for smoke |
