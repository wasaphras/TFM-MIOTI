"""
Step 2: Load rag_responses JSON/JSONL and score with RAGAS.

``minimal`` metrics: faithfulness, answer_relevancy, nv_context_relevance (NVIDIA dual-judge).

``full`` adds reference-based metrics (requires ``ground_truth_answer`` on each row or
``--ground-truth`` merge): context_precision, context_recall, answer_correctness.

``local`` (default) prefers **embedding + lexical** metrics — no judge chat LLM —
``answer_similarity`` + ``non_llm_context_precision_with_reference``
+ ``non_llm_context_recall``. Use this when Ollama chat models repeatedly fail Ragas JSON
prompts (`OutputParserException` on faithfulness / context_precision, etc.). ``minimal``
/ ``full`` still need a stronger JSON-compliant judge (`--judge-model`).

Uses LangChain embeddings + optional chat judge: **Ollama** (default when no
``GEMINI_API_KEY``) or **Google Gemini** via project ``.env`` (``GEMINI_API_KEY``,
``GEMINI_MODEL``) and ``--provider gemini`` / ``auto``.

Concurrency: Ragas submits ``len(metrics)`` jobs **per dataset row**. Defaults are tuned
for a **single-GPU**: **one GT row per** ``evaluate()`` and **``ragas-max-workers=1``**.

LLM-backed metrics expect **structured JSON**. For ``minimal``/``full`` on Ollama, weak
local models often fail — use ``--provider gemini`` or ``--metrics local``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    ContextRelevance,
    NonLLMContextPrecisionWithReference,
    NonLLMContextRecall,
    answer_correctness,
    answer_relevancy,
    answer_similarity,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig

from ... import config
from ..top10._shared import atomic_write_json, load_ground_truth, sha256_file

# Ensure `.env` is loaded when this module is the entry point (config already loads it on import).
config.load_project_dotenv()


def _resolve_provider_flag(flag: str) -> str:
    if flag != "auto":
        return flag
    return "gemini" if os.environ.get("GEMINI_API_KEY", "").strip() else "ollama"


def _gemini_api_key_or_raise() -> str:
    k = os.environ.get("GEMINI_API_KEY", "").strip()
    if not k:
        raise SystemExit(
            "Provider is gemini but GEMINI_API_KEY is empty. "
            "Add it to the project .env (project root) or export it in the shell."
        )
    return k


def _resolve_gemini_chat_model(args: argparse.Namespace) -> str:
    if getattr(args, "gemini_model", None):
        s = str(args.gemini_model).strip()
        if s:
            return s
    env_m = os.environ.get("GEMINI_MODEL", "").strip()
    return env_m or "gemini-2.0-flash"


def _resolve_gemini_embedding_model(args: argparse.Namespace) -> str:
    if getattr(args, "gemini_embedding_model", None):
        s = str(args.gemini_embedding_model).strip()
        if s:
            return s
    env_m = os.environ.get("GEMINI_EMBEDDING_MODEL", "").strip()
    # Google AI `embedContent`; `text-embedding-004` / legacy IDs often 404 on v1beta — use current GA id.
    return env_m or "gemini-embedding-001"


def build_judge_chat_gemini(args: argparse.Namespace) -> Any:
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=_resolve_gemini_chat_model(args),
        temperature=0.0,
        google_api_key=_gemini_api_key_or_raise(),
    )


def build_judge_chat(args: argparse.Namespace, *, provider: str) -> Any:
    if provider == "ollama":
        return build_judge_chat_ollama(args)
    return build_judge_chat_gemini(args)


def build_ragas_embeddings(args: argparse.Namespace, *, provider: str) -> Any:
    if provider == "ollama":
        return OllamaEmbeddings(model=args.embedding_model)
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(
        model=_resolve_gemini_embedding_model(args),
        google_api_key=_gemini_api_key_or_raise(),
    )


def resolve_effective_model_labels(args: argparse.Namespace, *, provider: str) -> tuple[str, str]:
    """(chat_or_judge_display_name, embedding_display_name) for logs and CSV lines."""
    if provider == "gemini":
        return _resolve_gemini_chat_model(args), _resolve_gemini_embedding_model(args)
    return str(args.llm_model), str(args.embedding_model)


def build_judge_chat_ollama(args: argparse.Namespace) -> ChatOllama:
    """ChatOllama tuned for Ragas JSON prompts (reduces RagasOutputParserException)."""
    kw: dict[str, Any] = {
        "model": args.llm_model,
        "temperature": 0.0,
        "num_ctx": max(512, int(args.ollama_num_ctx)),
    }
    if not args.ollama_plain_output:
        kw["format"] = "json"
    npred = getattr(args, "ollama_num_predict", None)
    if npred is not None and int(npred) > 0:
        kw["num_predict"] = max(1, int(npred))
    return ChatOllama(**kw)


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


def _contexts_from_docs(
    docs: list[dict[str, Any]],
    *,
    max_chars: int | None = None,
) -> list[str]:
    """Plain strings for RAGAS ``retrieved_contexts`` (one list element per chunk)."""
    out: list[str] = []
    for d in docs:
        if not isinstance(d, dict):
            continue
        text = str(d.get("page_content") or "").strip()
        if text:
            if max_chars is not None and len(text) > max_chars > 0:
                text = text[:max_chars] + "\n[truncated]"
            out.append(text)
    return out


def _reference_answer(row: dict[str, Any]) -> str:
    v = str(row.get("ground_truth_answer") or "").strip()
    return v


def _gt_index(gt_path: Path) -> dict[str, dict[str, Any]]:
    gt = load_ground_truth(Path(gt_path))
    return {str(r.get("id") or ""): r for r in gt if r.get("id")}


def _reference_contexts_list(row: dict[str, Any]) -> list[str]:
    rc = row.get("reference_contexts")
    if isinstance(rc, list):
        xs = [str(x).strip() for x in rc if str(x).strip()]
        if xs:
            return xs
    gs = str(row.get("gold_snippet") or "").strip()
    return [gs] if gs else []


def _maybe_truncate_text(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[truncated]"


def _truncate_context_list(ctxs: list[str], max_chars: int | None) -> list[str]:
    return [_maybe_truncate_text(c, max_chars) for c in ctxs]


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
    if mode == "local":
        return [
            answer_similarity,
            NonLLMContextPrecisionWithReference(),
            NonLLMContextRecall(),
        ]
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


def metrics_need_chat_llm(metrics: list[Any]) -> bool:
    try:
        from ragas.metrics.base import MetricWithLLM

        return any(isinstance(m, MetricWithLLM) for m in metrics)
    except Exception:
        return True


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


def _classify_skip(row: dict[str, Any], *, metrics_mode: str) -> str | None:
    docs = row.get("retrieved_documents")
    if not isinstance(docs, list):
        return "missing_retrieved_documents"
    ctxs = _contexts_from_docs(docs)
    if not ctxs:
        return "empty_retrieved_contexts"
    if metrics_mode == "full" and not _reference_answer(row):
        return "missing_ground_truth_answer"
    if metrics_mode == "local":
        if not _reference_answer(row):
            return "missing_ground_truth_answer"
        if not _reference_contexts_list(row):
            return "missing_reference_contexts"
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
        "--provider",
        choices=("auto", "ollama", "gemini"),
        default="auto",
        help=(
            "Ragas judge + embeddings backend: "
            "auto picks gemini when GEMINI_API_KEY is set (after loading .env), otherwise ollama."
        ),
    )
    p.add_argument(
        "--metrics",
        choices=("local", "minimal", "full"),
        default="local",
        help=(
            "local = answer_similarity + non-LLM context precision/recall (no judge chat; "
            "needs ground_truth_answer + reference_contexts). "
            "minimal/full = LLM metrics (fragile on small Ollama models)."
        ),
    )
    p.add_argument(
        "--llm-model",
        "--judge-model",
        dest="llm_model",
        default=config.LLM_MODEL,
        help="Chat model name when --provider is ollama (alias: --judge-model)",
    )
    p.add_argument(
        "--embedding-model",
        default=config.EMBEDDING_MODEL,
        help=(
            "Embedding model when --provider is ollama. "
            "For gemini, use GEMINI_EMBEDDING_MODEL or --gemini-embedding-model."
        ),
    )
    p.add_argument(
        "--gemini-model",
        default=None,
        metavar="ID",
        help="Override GEMINI_MODEL (.env) for Gemini chat (ignored for ollama).",
    )
    p.add_argument(
        "--gemini-embedding-model",
        default=None,
        metavar="ID",
        help="Override Gemini embeddings model (default: gemini-embedding-001).",
    )
    p.add_argument(
        "--context-max-chars",
        type=int,
        default=0,
        metavar="N",
        help=(
            "If >0, truncate each retrieved AND each reference context string to N chars "
            "(suffix [truncated]). 0 = no truncation."
        ),
    )
    p.add_argument(
        "--ollama-plain-output",
        action="store_true",
        help=(
            "Disable Ollama JSON format (no format=json). Ragas expects JSON-shaped "
            "answers; only use this if JSON mode breaks your model."
        ),
    )
    p.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=16384,
        metavar="TOK",
        help=(
            "Judge model context size. Small values truncate long metric prompts mid-JSON "
            "and cause Ragas parse errors. Lower if you hit OOM."
        ),
    )
    p.add_argument(
        "--ollama-num-predict",
        type=int,
        default=8192,
        metavar="TOK",
        help=(
            "Max new tokens for judge chat (minimal/full). 0 = omit (Ollama default). "
            "Ignored for --metrics local (no judge LLM)."
        ),
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--ragas-timeout",
        type=int,
        default=900,
        metavar="SEC",
        help="Per-metric-job timeout for RAGAS (single LLM call chain; increase if Ollama is slow)",
    )
    p.add_argument(
        "--ragas-max-retries",
        type=int,
        default=10,
        metavar="N",
        help="Passed to RunConfig.max_retries",
    )
    p.add_argument(
        "--ragas-max-workers",
        type=int,
        default=1,
        metavar="N",
        help=(
            "RAGAS Executor parallelism (default 1 = strictly sequential metric jobs; "
            "raise only if you want concurrent calls and VRAM allows)"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help=(
            "How many ground-truth rows per ragas.evaluate() call (default 1). "
            "Use >>1 only if models are fast enough; larger values multiply parallel jobs."
        ),
    )
    p.add_argument(
        "--debug-metrics",
        action="store_true",
        help="Pass raise_exceptions=True into ragas.evaluate",
    )
    args = p.parse_args()

    provider = _resolve_provider_flag(args.provider)
    if provider == "gemini":
        _gemini_api_key_or_raise()
    effective_llm, effective_emb = resolve_effective_model_labels(args, provider=provider)

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
    needs_chat_llm = metrics_need_chat_llm(metrics)
    ctx_limit = int(args.context_max_chars)
    ctx_max = ctx_limit if ctx_limit > 0 else None

    meta = {
        "schema": "ragas_judge_scores_v3",
        "framework": "ragas",
        "ragas_version": ragas_ver,
        "input_path": str(in_path.resolve()),
        "input_sha256": in_sha,
        "ground_truth_path": str(gt_path.resolve()) if gt_path else None,
        "ground_truth_sha256": gt_sha,
        "ground_truth_answer_is_llm_authored": True,
        "ragas_provider": provider,
        "llm_model": effective_llm,
        "embedding_model": effective_emb,
        "ollama_judge_chat_model_if_applicable": args.llm_model if provider == "ollama" else None,
        "ollama_embedding_model_if_applicable": args.embedding_model if provider == "ollama" else None,
        "metrics_mode": args.metrics,
        "metrics_needs_judge_chat_llm": needs_chat_llm,
        "metrics": m_names,
        "evaluate_rows_per_batch": max(1, int(args.batch_size)),
        "ragas_timeout_sec": args.ragas_timeout,
        "ragas_max_retries": args.ragas_max_retries,
        "ragas_max_workers": max(1, int(args.ragas_max_workers)),
        "ollama_format_json": not args.ollama_plain_output,
        "ollama_num_ctx": max(512, int(args.ollama_num_ctx)),
        "ollama_num_predict": args.ollama_num_predict,
        "context_max_chars": ctx_limit,
        "notes": (
            "metrics=local: answer_similarity + non_llm_context_precision_with_reference + "
            "non_llm_context_recall (embedding + rapidfuzz; no judge chat). "
            "Requires ground_truth_answer + non-empty reference_contexts per row. "
            "Provider auto: picks gemini if GEMINI_API_KEY is set (.env OK), else Ollama. "
            "metrics=minimal/full: use --provider gemini for reliable structured JSON judging, "
            "or weaker Ollama chats may OutputParserException. "
            "Default batch_size=1 and ragas_max_workers=1."
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
    need_reference_answer = args.metrics in ("full", "local")

    if pending:
        judge_llm = build_judge_chat(args, provider=provider) if needs_chat_llm else None
        judge_emb = build_ragas_embeddings(args, provider=provider)
        run_cfg = RunConfig(
            timeout=args.ragas_timeout,
            max_retries=args.ragas_max_retries,
            max_workers=max(1, int(args.ragas_max_workers)),
        )

        chunk_size = max(1, int(args.batch_size))
        with open(out_path, mode, encoding="utf-8") as fout:
            offset = 0
            while offset < len(pending):
                batch = pending[offset : offset + chunk_size]
                offset += len(batch)

                skip_records: list[tuple[str, dict[str, Any], str]] = []
                eval_slice: list[tuple[str, dict[str, Any]]] = []
                for qid, row in batch:
                    sr = _classify_skip(row, metrics_mode=args.metrics)
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
                        "llm_model": effective_llm,
                        "embedding_model": effective_emb,
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
                                else [],
                                max_chars=ctx_max,
                            )
                            for r in eval_slice
                        ],
                        "reference": [
                            (_reference_answer(r[1]) if need_reference_answer else "")
                            for r in eval_slice
                        ],
                        "reference_contexts": [
                            (
                                _truncate_context_list(_reference_contexts_list(r[1]), ctx_max)
                                if args.metrics == "local"
                                else []
                            )
                            for r in eval_slice
                        ],
                    }
                )
                ev_kwargs: dict[str, Any] = {
                    "metrics": metrics,
                    "embeddings": judge_emb,
                    "run_config": run_cfg,
                    "show_progress": True,
                    "raise_exceptions": args.debug_metrics,
                }
                if needs_chat_llm:
                    ev_kwargs["llm"] = judge_llm
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
                        "llm_model": effective_llm,
                        "embedding_model": effective_emb,
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
            "ragas_provider": provider,
            "llm_model": effective_llm,
            "embedding_model": effective_emb,
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
