"""
Step 2: Load rag_responses JSON/JSONL and score with RAGAS.

``minimal`` metrics: faithfulness, answer_relevancy, nv_context_relevance (NVIDIA dual-judge).

``full`` adds reference-based metrics (requires ``ground_truth_answer`` on each row or
``--ground-truth`` merge): context_precision, context_recall, answer_correctness.

Uses LangChain Ollama chat + embeddings like the rest of the TFM stack.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextRelevance,
    answer_correctness,
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from ... import config
from ..top10._shared import atomic_write_json, load_ground_truth, sha256_file


def load_rag_artifact(path: Path) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    path = Path(path)
    if path.suffix.lower() == ".jsonl" or path.name.endswith(".jsonl"):
        items: list[dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return None, items
    with open(path, encoding="utf-8") as f:
        blob = json.load(f)
    meta = blob.get("meta") if isinstance(blob.get("meta"), dict) else None
    items = blob.get("items")
    if not isinstance(items, list):
        raise ValueError(f"Expected {{'items': [...]}} in {path}")
    return meta, [x for x in items if isinstance(x, dict)]


def _load_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.is_file():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec.get("id"), str):
                done.add(rec["id"])
    return done


def _contexts_from_docs(docs: list[dict[str, Any]]) -> list[str]:
    """Plain strings for RAGAS ``retrieved_contexts`` (one list element per chunk)."""
    out: list[str] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        text = str(d.get("page_content") or "").strip()
        if text:
            out.append(text)
    return out


def _reference_answer(row: dict[str, Any]) -> str:
    v = str(row.get("ground_truth_answer") or "").strip()
    return v


def _gt_index(gt_path: Path) -> dict[str, dict[str, Any]]:
    gt = load_ground_truth(Path(gt_path))
    return {str(r.get("id") or ""): r for r in gt if r.get("id")}


def augment_items_from_ground_truth(items: list[dict[str, Any]], gt_path: Path) -> None:
    """Fill ``ground_truth_answer`` / ``reference_contexts`` from GT when missing (legacy v1 rows)."""
    by_id = _gt_index(gt_path)
    for row in items:
        qid = str(row.get("id") or "")
        if qid not in by_id:
            continue
        g = by_id[qid]
        if not _reference_answer(row):
            row["ground_truth_answer"] = str(g.get("answer") or "")
        rc = row.get("reference_contexts")
        if not isinstance(rc, list) or not any(str(x).strip() for x in rc):
            gs = str(row.get("gold_snippet") or g.get("gold_snippet") or "").strip()
            row["reference_contexts"] = [gs] if gs else []


def _build_metrics(mode: str) -> list[Any]:
    base: list[Any] = [
        faithfulness,
        answer_relevancy,
        ContextRelevance(),
    ]
    if mode == "minimal":
        return base
    if mode == "full":
        return base + [context_precision, context_recall, answer_correctness]
    raise ValueError(f"Unknown metrics mode: {mode!r}")


def metric_names(metrics: list[Any]) -> list[str]:
    out: list[str] = []
    for m in metrics:
        name = getattr(m, "name", None)
        if isinstance(name, str):
            out.append(name)
    return out


def _float_metric(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
        if math.isnan(x):
            return None
        return x
    except (TypeError, ValueError):
        return None


def extract_metric_scores(score_row: dict[str, Any], names: list[str]) -> dict[str, float | None]:
    return {n: _float_metric(score_row.get(n)) for n in names}


def _overall_mean(scores: dict[str, float | None]) -> float | None:
    vals = [v for v in scores.values() if v is not None]
    if not vals:
        return None
    return round(sum(vals) / len(vals), 6)


def write_csv_summary(
    csv_path: Path,
    rows: list[dict[str, Any]],
    metric_cols: list[str],
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", *metric_cols, "overall_score", "ragas_skip_reason"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _classify_skip(row: dict[str, Any], *, full: bool) -> str | None:
    docs = row.get("retrieved_documents")
    if not isinstance(docs, list):
        return "missing_retrieved_documents"
    ctxs = _contexts_from_docs(docs)
    if not ctxs:
        return "empty_retrieved_contexts"
    if full and not _reference_answer(row):
        return "missing_ground_truth_answer"
    return None


def write_summary_stats_json(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    metric_cols: list[str],
    extra: dict[str, Any],
) -> None:
    per_metric: dict[str, Any] = {}
    for col in metric_cols:
        vals: list[float] = []
        for r in rows:
            if r.get("ragas_skip_reason"):
                continue
            v = _float_metric(r.get(col))
            if v is not None:
                vals.append(v)
        if vals:
            per_metric[col] = {
                "mean": round(statistics.mean(vals), 6),
                "stdev": round(statistics.stdev(vals), 6) if len(vals) > 1 else 0.0,
                "n_non_nan": len(vals),
            }
        else:
            per_metric[col] = {"mean": None, "stdev": None, "n_non_nan": 0}
    atomic_write_json(
        path,
        {**extra, "per_metric": per_metric},
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in", dest="in_path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="RAGAS scores JSONL append path")
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Summary CSV (rewritten from all rows in --out at end)",
    )
    p.add_argument(
        "--ground-truth",
        type=Path,
        default=None,
        help="Optional dedup GT JSONL: merge answer/gold_snippet into rows missing v2 fields",
    )
    p.add_argument("--limit-queries", type=int, default=None, metavar="N")
    p.add_argument(
        "--metrics",
        choices=("minimal", "full"),
        default="full",
        help="full = default Ragas-style bundle incl. reference-based metrics",
    )
    p.add_argument(
        "--llm-model",
        "--judge-model",
        dest="llm_model",
        default=config.LLM_MODEL,
        help="Ollama chat model for RAGAS metrics (alias: --judge-model)",
    )
    p.add_argument(
        "--embedding-model",
        default=config.EMBEDDING_MODEL,
        help="Ollama embedding model (required for answer_relevancy / answer_correctness similarity)",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--ragas-timeout",
        type=int,
        default=300,
        metavar="SEC",
        help="Per-operation timeout for RAGAS",
    )
    p.add_argument(
        "--ragas-max-retries",
        type=int,
        default=10,
        metavar="N",
        help="Passed to RunConfig.max_retries",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Optional batch size passed to ragas.evaluate",
    )
    p.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Pass raise_exceptions=True into ragas.evaluate",
    )
    args = p.parse_args()

    in_path = Path(args.in_path)
    _, items = load_rag_artifact(in_path)
    if args.limit_queries is not None:
        items = items[: args.limit_queries]

    gt_path = Path(args.ground_truth) if args.ground_truth else None
    if gt_path is not None:
        augment_items_from_ground_truth(items, gt_path)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    in_sha = sha256_file(in_path)
    gt_sha = sha256_file(gt_path) if gt_path and gt_path.is_file() else None

    try:
        import ragas as _ragas

        ragas_ver = getattr(_ragas, "__version__", "unknown")
    except Exception:
        ragas_ver = "unknown"

    metrics = _build_metrics(args.metrics)
    m_names = metric_names(metrics)

    meta = {
        "schema": "ragas_judge_scores_v2",
        "framework": "ragas",
        "ragas_version": ragas_ver,
        "input_path": str(in_path.resolve()),
        "input_sha256": in_sha,
        "ground_truth_path": str(gt_path.resolve()) if gt_path else None,
        "ground_truth_sha256": gt_sha,
        "ground_truth_answer_is_llm_authored": True,
        "llm_model": args.llm_model,
        "embedding_model": args.embedding_model,
        "metrics_mode": args.metrics,
        "metrics": m_names,
        "notes": (
            "reference-based metrics use ground_truth_answer (LLM-authored, from GT file). "
            "context_precision / context_recall / answer_correctness require that field or --ground-truth merge. "
            "nv_context_relevance is the ContextRelevance metric name in ragas 0.4.x."
        ),
    }
    meta_path = out_path.with_suffix(out_path.suffix + ".meta.json")

    done = _load_done_ids(out_path) if args.resume else set()
    mode = "a" if args.resume and out_path.is_file() else "w"
    if mode == "w" or not meta_path.is_file():
        meta["n_items_total"] = len(items)
        atomic_write_json(meta_path, meta)

    pending: list[tuple[str, dict[str, Any]]] = []
    for row in items:
        qid = str(row.get("id") or "")
        if qid in done:
            continue
        pending.append((qid, row))

    n_fail = 0
    n_skipped = 0
    full_mode = args.metrics == "full"

    if pending:
        llm = ChatOllama(model=args.llm_model, temperature=0.0)
        emb = OllamaEmbeddings(model=args.embedding_model)
        r_llm = LangchainLLMWrapper(llm)
        r_emb = LangchainEmbeddingsWrapper(emb)
        run_cfg = RunConfig(
            timeout=args.ragas_timeout,
            max_retries=args.ragas_max_retries,
        )

        chunk_size = args.batch_size or len(pending)
        with open(out_path, mode, encoding="utf-8") as fout:
            offset = 0
            while offset < len(pending):
                batch = pending[offset : offset + chunk_size]
                offset += len(batch)

                skip_records: list[tuple[str, dict[str, Any], str]] = []
                eval_slice: list[tuple[str, dict[str, Any]]] = []
                for qid, row in batch:
                    sr = _classify_skip(row, full=full_mode)
                    if sr:
                        skip_records.append((qid, row, sr))
                    else:
                        eval_slice.append((qid, row))

                for qid, row, sr in skip_records:
                    n_skipped += 1
                    blank_scores = {name: None for name in m_names}
                    overall = None
                    out_rec: dict[str, Any] = {
                        "id": qid,
                        "question": row.get("question"),
                        **blank_scores,
                        "overall_score": overall,
                        "ragas_skip_reason": sr,
                        "ragas_version": ragas_ver,
                        "llm_model": args.llm_model,
                        "embedding_model": args.embedding_model,
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    fout.flush()

                if not eval_slice:
                    continue

                ds = Dataset.from_dict(
                    {
                        "user_input": [str(r[1].get("question") or "") for r in eval_slice],
                        "response": [str(r[1].get("rag_answer") or "") for r in eval_slice],
                        "retrieved_contexts": [
                            _contexts_from_docs(
                                r[1].get("retrieved_documents")
                                if isinstance(r[1].get("retrieved_documents"), list)
                                else []
                            )
                            for r in eval_slice
                        ],
                        "reference": [
                            _reference_answer(r[1]) if full_mode else ""
                            for r in eval_slice
                        ],
                    }
                )
                ev_kwargs: dict[str, Any] = {
                    "metrics": metrics,
                    "llm": r_llm,
                    "embeddings": r_emb,
                    "run_config": run_cfg,
                    "show_progress": True,
                    "raise_exceptions": args.debug_metrics,
                }
                if args.batch_size is not None:
                    ev_kwargs["batch_size"] = args.batch_size
                ev = evaluate(ds, **ev_kwargs)
                scores_list = ev.scores or []

                for bi, (qid, row) in enumerate(eval_slice):
                    score_row = scores_list[bi] if bi < len(scores_list) else {}
                    scores = extract_metric_scores(score_row, m_names)
                    if all(v is None for v in scores.values()):
                        n_fail += 1
                    overall = _overall_mean(scores)
                    out_rec = {
                        "id": qid,
                        "question": row.get("question"),
                        **scores,
                        "overall_score": overall,
                        "ragas_skip_reason": None,
                        "ragas_version": ragas_ver,
                        "llm_model": args.llm_model,
                        "embedding_model": args.embedding_model,
                    }
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    fout.flush()

    all_rows: list[dict[str, Any]] = []
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_rows.append(json.loads(line))

    csv_path = args.out_csv
    if csv_path is None:
        csv_path = out_path.with_name(out_path.stem + "_summary.csv")
    write_csv_summary(Path(csv_path), all_rows, m_names)

    stats_path = out_path.with_name(out_path.stem + "_summary_stats.json")
    write_summary_stats_json(
        stats_path,
        rows=all_rows,
        metric_cols=m_names,
        extra={
            "input_sha256": in_sha,
            "ground_truth_sha256": gt_sha,
            "ragas_version": ragas_ver,
            "llm_model": args.llm_model,
            "embedding_model": args.embedding_model,
            "metrics_mode": args.metrics,
            "n_rows_output": len(all_rows),
            "metrics_all_nan_rows": n_fail,
            "skipped_rows": n_skipped,
        },
    )

    meta_end = dict(meta)
    meta_end["n_lines_written"] = len(all_rows)
    meta_end["metrics_all_nan_rows"] = n_fail
    meta_end["skipped_rows"] = n_skipped
    meta_end["summary_csv"] = str(Path(csv_path).resolve())
    meta_end["summary_stats_json"] = str(stats_path.resolve())
    atomic_write_json(meta_path, meta_end)

    print(f"Wrote {out_path} ({len(all_rows)} rows), CSV {csv_path}, stats {stats_path}")
    if n_skipped:
        print(f"Note: {n_skipped} rows skipped (see ragas_skip_reason in JSONL / CSV)")
    if n_fail:
        print(f"Warning: {n_fail} evaluated rows had all metric values missing (NaN/errors)")


if __name__ == "__main__":
    main()
