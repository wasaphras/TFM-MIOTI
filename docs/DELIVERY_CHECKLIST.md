# Delivery checklist (school hand-in)

## Repository contents

- [ ] All code under `Scripts/` and `webapp/` matches thesis description
- [ ] `requirements.txt` installs on Python 3.11+
- [ ] `.env.example` copied to `.env` locally (never commit `.env`)
- [ ] `docs/ACADEMIC_USE.md` - fill in author name and institution
- [ ] `docs/REPRODUCING_RESULTS.md` - maps committed CSVs to experiments
- [ ] `tests/fixture/Data/` committed (~32 MB, 10-doc Chroma ~2 MB each)

## Verify without full MultiEURLEX download

```bash
pip install -r requirements.txt
ollama pull qwen3-embedding:4b
python tests/validate_fixture.py
bash tests/run_smoke_test.sh --skip-ragas
```

Expected: `validate_fixture.py` prints OK; smoke test ends with `7 passed, 0 failed` (Ragas optional).

## Full thesis corpus (local only, gitignored)

```bash
export HF_TOKEN=...
python -m Scripts.data_extraction_load
python -m Scripts.eval.build_chunk_indices --all
# ... see README.md Steps 3-6
```

## Common mistakes

| Mistake | Symptom |
|---------|---------|
| Copying `Data/chroma_chunk_*` into fixture | Fixture >1 GB per strategy |
| Missing `TFM_CATEGORIES_JSON` | Preprocess fails on fixture |
| GPU rerank on 6 GB card | OOM; use `RERANK_DEVICE=cpu` for smoke |
