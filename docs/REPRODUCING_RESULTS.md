# Reproducing thesis evaluation outputs

Committed CSVs under `Data/` correspond to these tracks:

| Path | Track | Contents |
|------|-------|----------|
| `Data/eval/results_summary.csv` | Main grid | 10 chunk strategies x 20 retrievers, 100 GT questions, final k=10 |
| `Data/eval_top10/eval1_baseline/` | Top-10 eval 1 | 10 curated pairs, k=20, 100 questions |
| `Data/eval_top10/eval2_neighbors/` | Top-10 eval 2 | Neighbor expansion + CE |
| `Data/eval_top10/eval3_enhanced/` | Top-10 eval 3 | LLM-enhanced queries |
| `Data/eval_top10/eval4_multiquery/` | Top-10 eval 4 | Multi-query fusion |
| `Data/eval_top10_dedup/eval1_baseline/` | Dedup eval 1 | Same pair set on dedup corpus |
| `Data/eval_top10_dedup/eval2_neighbors/` | Dedup eval 2 | Dedup neighbors |
| `Data/eval_top10_dedup/llm_triad_len500_hyb_fill/` | Ragas | RAG answers + Ragas scores on dedup winner cell |

Regenerate locally: follow numbered steps in [README.md](../README.md). Quick stack check without the full corpus:

```bash
bash tests/run_smoke_test.sh
```
