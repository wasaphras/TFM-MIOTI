"""LLM rewrite (eval 3) and two query variants (eval 4), incremental JSONL + checkpoint."""

from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from ... import config
from ._shared import append_jsonl_record, atomic_write_json

ENHANCED_DIR = config.DATA_DIR / "enhanced_questions"
MULTI_DIR = config.DATA_DIR / "multi_query_questions"

# One in-memory index per JSONL path (avoid re-reading the whole file on every GT row).
_enhanced_by_path: dict[str, dict[str, dict]] = {}
_multi_by_path: dict[str, dict[str, dict]] = {}


def _default_checkpoint(path: Path) -> Path:
    return path.with_name(path.name + ".checkpoint.json")


def _load_jsonl_index(path: Path) -> dict[str, dict]:
    by_id: dict[str, dict] = {}
    if not path.is_file():
        return by_id
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_id[str(rec["id"])] = rec
    return by_id


def _enhanced_table(path: Path) -> dict[str, dict]:
    key = str(path.resolve())
    if key not in _enhanced_by_path:
        _enhanced_by_path[key] = _load_jsonl_index(path)
    return _enhanced_by_path[key]


def _multi_table(path: Path) -> dict[str, dict]:
    key = str(path.resolve())
    if key not in _multi_by_path:
        _multi_by_path[key] = _load_jsonl_index(path)
    return _multi_by_path[key]


def clear_question_disk_caches() -> None:
    """Drop cached JSONL indexes (frees RAM between bulk phases). Reloads from disk on next access."""
    _enhanced_by_path.clear()
    _multi_by_path.clear()
    gc.collect()


def _enhance_prompt(question: str, target_words: int) -> str:
    return (
        "You are an EU legal-text retrieval assistant. Rewrite the user's question "
        "into a richer, formally worded English legal question of about "
        f"{target_words} words. Use precise legal/regulatory phrasing typical of EU "
        "directives and regulations (e.g. \"this Regulation\", \"the competent authority\", "
        "\"Article\", \"Member States\"). Preserve the original intent. Do not invent facts. "
        "Do not include an answer.\n\n"
        "Reply with a single JSON object and nothing else, format: "
        '{"enhanced_question": "..."}\n\n'
        f"Original question: {question}"
    )


def _variants_prompt(enhanced_question: str, target_words: int) -> str:
    return (
        "Given a formal EU-legal question, write TWO alternative phrasings that ask about "
        "the SAME underlying legal fact but use different legal vocabulary, sentence structure, "
        "or focus angle. Keep each about "
        f"{target_words} words. Do not invent facts.\n\n"
        "Reply with a single JSON object and nothing else, format: "
        '{"variants": ["...", "..."]}\n\n'
        f"Question: {enhanced_question}"
    )


def _parse_json_obj(content: str) -> dict[str, Any] | None:
    """Parse a JSON object; tolerate markdown fences, prose around JSON, trailing commas."""
    content = (content or "").strip()
    if not content:
        return None
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()

    candidates: list[str] = [content]
    if "{" in content and "}" in content:
        start = content.find("{")
        end = content.rfind("}") + 1
        if end > start:
            inner = content[start:end]
            if inner not in candidates:
                candidates.append(inner)

    for cand in candidates:
        fixed = cand
        for _ in range(3):
            try:
                obj = json.loads(fixed)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
            fixed = re.sub(r",\s*([}\]])", r"\1", fixed)
    return None


def _parse_json_any(content: str) -> Any | None:
    """Parse JSON object or top-level array (models sometimes return only the array)."""
    d = _parse_json_obj(content)
    if d is not None:
        return d
    content = (content or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n?```$", "", content).strip()
    if "[" in content and "]" in content:
        start = content.find("[")
        end = content.rfind("]") + 1
        if end > start:
            try:
                arr = json.loads(content[start:end])
                if isinstance(arr, list):
                    return arr
            except json.JSONDecodeError:
                pass
    return None


def _pair_from_variant_list(vs: Any) -> tuple[str, str] | None:
    if not isinstance(vs, list) or len(vs) < 2:
        return None
    a, b = vs[0], vs[1]
    if not isinstance(a, str) or not isinstance(b, str):
        return None
    a, b = a.strip(), b.strip()
    min_len = 10
    if len(a) >= min_len and len(b) >= min_len and a.lower() != b.lower():
        return a, b
    return None


def _variants_from_model_output(content: str) -> tuple[str, str] | None:
    parsed = _parse_json_any(content)
    if isinstance(parsed, dict):
        pair = _pair_from_variant_list(parsed.get("variants"))
        if pair:
            return pair
    if isinstance(parsed, list):
        return _pair_from_variant_list(parsed)
    return None


def _heuristic_variant_pair(enhanced_question: str) -> tuple[str, str] | None:
    """Last-resort variants when the LLM keeps failing (eval quality may be lower)."""
    eq = (enhanced_question or "").strip()
    if len(eq) < 25:
        return None
    parts = re.split(r"(?<=[.!?])\s+", eq)
    chunks = [p.strip() for p in parts if len(p.strip()) > 12]
    if len(chunks) >= 2:
        v1, v2 = chunks[0], " ".join(chunks[1:])
        if len(v2) >= 12:
            return v1, v2
    v2 = "From the perspective of EU legal obligations: " + eq[:3000]
    if len(v2) >= 25 and v2.strip().lower() != eq.lower():
        return eq, v2.strip()
    return None


class QuestionEnhancer:
    def __init__(self, temperature_enhance: float = 0.2, temperature_variants: float = 0.3):
        self._llm_enhance = ChatOllama(model=config.LLM_MODEL, temperature=temperature_enhance)
        self._llm_variants = ChatOllama(model=config.LLM_MODEL, temperature=temperature_variants)

    def close(self) -> None:
        """Release LLM clients (helps RAM between bulk phases)."""
        self._llm_enhance = None  # type: ignore[assignment]
        self._llm_variants = None  # type: ignore[assignment]
        gc.collect()

    def enhance(
        self,
        question: str,
        target_words: int,
        max_attempts: int = 8,
        *,
        retry_sleep_s: float = 0.0,
    ) -> str | None:
        if self._llm_enhance is None:
            raise RuntimeError("QuestionEnhancer is closed")
        prompt = _enhance_prompt(question, target_words)
        extras = (
            "",
            "\n\nOutput valid JSON only.",
            "\n\nSingle JSON object. No markdown. Key: enhanced_question.",
        )
        for i in range(max_attempts):
            if i > 0 and retry_sleep_s > 0:
                time.sleep(retry_sleep_s)
            p = prompt + extras[min(i, len(extras) - 1)]
            try:
                msg = self._llm_enhance.invoke([HumanMessage(content=p)])
                data = _parse_json_obj(str(msg.content or ""))
                if data and isinstance(data.get("enhanced_question"), str):
                    q = data["enhanced_question"].strip()
                    if len(q) > 20:
                        return q
            except Exception:
                continue
        return None

    def variants(
        self,
        enhanced_question: str,
        target_words: int,
        max_attempts: int = 10,
        *,
        retry_sleep_s: float = 1.5,
    ) -> tuple[str, str] | None:
        if self._llm_variants is None:
            raise RuntimeError("QuestionEnhancer is closed")
        base = _variants_prompt(enhanced_question, target_words)
        suffixes = (
            "",
            "\n\nOutput valid JSON only. Format: {\"variants\": [\"...\", \"...\"]}",
            "\n\nNo markdown fences. No commentary. One JSON object, two distinct strings in variants.",
            '\n\nExample shape only: {"variants": ["First complete question?", "Second complete question?"]}',
        )
        for i in range(max_attempts):
            if i > 0 and retry_sleep_s > 0:
                time.sleep(retry_sleep_s)
            p = base + suffixes[min(i, len(suffixes) - 1)]
            try:
                msg = self._llm_variants.invoke([HumanMessage(content=p)])
                raw = str(msg.content or "")
                pair = _variants_from_model_output(raw)
                if pair:
                    return pair
            except Exception:
                continue
        return None


def require_enhanced_row(strategy_id: str, gt_row: dict, target_words: int) -> dict:
    """
    Load enhanced question from JSONL only (no LLM). For --prefetch-write after materialize_llm_inputs.
    """
    out_path = ENHANCED_DIR / f"{strategy_id}.jsonl"
    by_id = _enhanced_table(out_path)
    qid = str(gt_row.get("id") or "")
    if qid not in by_id:
        raise SystemExit(
            f"Missing enhanced question for id={qid!r} strategy={strategy_id!r} in {out_path}.\n"
            "Run the LLM materialization phase first, e.g.:\n"
            "  python -m Scripts.eval.top10.materialize_llm_inputs --eval eval3 "
            "(or --eval both)"
        )
    rec = by_id[qid]
    if int(rec.get("target_words") or 0) != int(target_words):
        tqdm.write(
            f"Warning: cached target_words for {qid} ({rec.get('target_words')}) "
            f"!= current {target_words} (chunk_stats changed?). Using cached text."
        )
    return rec


def require_variants_row(strategy_id: str, gt_row: dict, target_words: int) -> dict:
    """Load multi-query record from JSONL only (no LLM). Requires enhanced + multi files."""
    require_enhanced_row(strategy_id, gt_row, target_words)
    out_path = MULTI_DIR / f"{strategy_id}.jsonl"
    by_id = _multi_table(out_path)
    qid = str(gt_row.get("id") or "")
    if qid not in by_id:
        raise SystemExit(
            f"Missing multi-query record for id={qid!r} strategy={strategy_id!r} in {out_path}.\n"
            "Run: python -m Scripts.eval.top10.materialize_llm_inputs --eval eval4 (or --eval both)"
        )
    return by_id[qid]


def ensure_enhanced_row(
    strategy_id: str,
    gt_row: dict,
    target_words: int,
    enhancer: QuestionEnhancer,
    *,
    verbose: bool = False,
    enhance_max_attempts: int = 8,
    enhance_retry_sleep_s: float = 0.0,
) -> dict:
    """Return record for gt_row id; append to JSONL if newly generated."""
    ENHANCED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ENHANCED_DIR / f"{strategy_id}.jsonl"
    ck_path = _default_checkpoint(out_path)
    qid = str(gt_row.get("id") or "")
    question = str(gt_row.get("question") or "")
    by_id = _enhanced_table(out_path)
    if qid in by_id:
        return by_id[qid]

    if verbose:
        tqdm.write(f"Enhance [{strategy_id}] {qid} …")
    enhanced = enhancer.enhance(
        question,
        target_words,
        max_attempts=enhance_max_attempts,
        retry_sleep_s=enhance_retry_sleep_s,
    )
    if not enhanced:
        raise RuntimeError(
            f"LLM failed to enhance question for {qid} ({strategy_id}). "
            "Re-run to resume; try --llm-max-attempts / --retry-sleep-seconds."
        )

    rec = {
        "id": qid,
        "original_question": question,
        "enhanced_question": enhanced,
        "target_words": target_words,
        "chunk_strategy": strategy_id,
    }
    append_jsonl_record(out_path, rec)
    by_id[qid] = rec
    _save_enhance_checkpoint(out_path, ck_path, strategy_id, sorted(by_id.keys()))
    return rec


def ensure_variants_row(
    strategy_id: str,
    gt_row: dict,
    target_words: int,
    enhancer: QuestionEnhancer,
    *,
    verbose: bool = False,
    variant_max_attempts: int = 10,
    variant_retry_sleep_s: float = 1.5,
    heuristic_fallback: bool = False,
) -> dict:
    """Requires enhanced row to exist (or creates it). Writes multi_query JSONL."""
    enhanced_rec = ensure_enhanced_row(
        strategy_id,
        gt_row,
        target_words,
        enhancer,
        verbose=verbose,
        enhance_max_attempts=variant_max_attempts,
        enhance_retry_sleep_s=variant_retry_sleep_s,
    )
    MULTI_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MULTI_DIR / f"{strategy_id}.jsonl"
    ck_path = _default_checkpoint(out_path)
    qid = str(gt_row.get("id") or "")
    by_id = _multi_table(out_path)
    if qid in by_id:
        return by_id[qid]

    eq = enhanced_rec["enhanced_question"]
    if verbose:
        tqdm.write(f"Variants [{strategy_id}] {qid} …")
    pair = enhancer.variants(
        eq,
        target_words,
        max_attempts=variant_max_attempts,
        retry_sleep_s=variant_retry_sleep_s,
    )
    if not pair and heuristic_fallback:
        pair = _heuristic_variant_pair(eq)
        if pair:
            tqdm.write(
                f"WARNING: heuristic variants for {qid} ({strategy_id}) "
                "(LLM did not return valid JSON after all attempts)."
            )
    if not pair:
        raise RuntimeError(
            f"LLM failed variants for {qid} ({strategy_id}). "
            "Re-run this script (resume skips completed ids). "
            "Try: increase --llm-max-attempts / --retry-sleep-seconds, restart Ollama, "
            "or pass --heuristic-fallback for last-resort paraphrases."
        )
    v1, v2 = pair
    rec = {
        "id": qid,
        "chunk_strategy": strategy_id,
        "original_question": enhanced_rec["original_question"],
        "enhanced_question": eq,
        "variants": [v1, v2],
        "target_words": target_words,
    }
    append_jsonl_record(out_path, rec)
    by_id[qid] = rec
    _save_enhance_checkpoint(out_path, ck_path, strategy_id, sorted(by_id.keys()), kind="multi")
    return rec


def _save_enhance_checkpoint(
    out_path: Path,
    ck_path: Path,
    strategy_id: str,
    completed_ids: list[str],
    *,
    kind: str = "enhance",
) -> None:
    atomic_write_json(
        ck_path,
        {
            "version": 1,
            "kind": kind,
            "strategy_id": strategy_id,
            "jsonl": str(out_path.resolve()),
            "n_completed": len(completed_ids),
            "completed_ids": completed_ids,
        },
    )
