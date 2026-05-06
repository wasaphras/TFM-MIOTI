"""
Generate ground_truth_dedup_top10_100.jsonl for dedup corpus eval.

Uses Data/train_dedup.jsonl + Data/eval_corpus_manifest_dedup.json.
Snippets must appear only in middle chunks (not first/last) for each
distinct top-10 chunk strategy. Question+answer JSON from LLM + validation pass.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import pickle
import random
import re
import signal
from pathlib import Path
from typing import Any

import pandas as pd
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from tqdm import tqdm

from .. import config
from ..chunking_strategies import chunk_one_document, row_meta
from ..preprocess import preprocess_for_rag
from .metrics import normalize_text
from .top10.dedup_paths import DEDUP_GROUND_TRUTH, DEDUP_MANIFEST, DEDUP_TRAIN_JSONL
from .top10.pairs import SELECTED_PAIRS_EVAL1, distinct_chunk_strategies

_SNIPPET_HARD_CAP = 480

CHECKPOINT_VERSION = 3

TOP10_STRATEGY_IDS: tuple[str, ...] = distinct_chunk_strategies(SELECTED_PAIRS_EVAL1)

_FORBIDDEN_Q_SUBSTRINGS = (
    "excerpt",
    "passage",
    "above",
    "below",
    "provided text",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def _default_checkpoint_path(out_path: Path) -> Path:
    return out_path.with_name(out_path.name + ".checkpoint.json")


def _count_jsonl_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _existing_references_from_jsonl(path: Path) -> set[str]:
    refs: set[str] = set()
    if not path.is_file():
        return refs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            r = str(rec.get("reference") or "").strip()
            if r:
                refs.add(r)
    return refs


def _append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _serialize_rng(rng: random.Random) -> str:
    return base64.b64encode(pickle.dumps(rng.getstate(), protocol=4)).decode("ascii")


def _deserialize_rng(data_b64: str, rng: random.Random) -> None:
    rng.setstate(pickle.loads(base64.b64decode(data_b64.encode("ascii"))))


def _snap_sentence_start(text: str, start: int, max_back: int = 120) -> int:
    if start <= 0:
        return 0
    lo = max(0, start - max_back)
    window = text[lo:start]
    best = -1
    for sep in (".\n", ". ", "\n\n"):
        j = window.rfind(sep)
        if j != -1:
            best = max(best, lo + j + len(sep))
    return start if best < 0 else min(best, len(text))


def _snap_sentence_end(text: str, end: int, n: int, max_fwd: int = 200) -> int:
    hi = min(n, end + max_fwd)
    window = text[end:hi]
    p = window.find(". ")
    if p != -1:
        return min(n, end + p + 1)
    p2 = window.find(".\n")
    if p2 != -1:
        return min(n, end + p2 + 1)
    return end


def _pick_snippet(
    text: str,
    rng: random.Random,
    snippet_min_chars: int,
    snippet_max_chars: int,
) -> str | None:
    cap = min(snippet_max_chars, _SNIPPET_HARD_CAP)
    low = max(60, min(snippet_min_chars, cap))
    if low > cap:
        return None
    n = len(text)
    if n < 500:
        return None
    lo = max(150, n // 6)
    hi = min(n - cap // 2, 5 * n // 6)
    if lo >= hi:
        return None
    target_len = rng.randint(low, cap)
    start = rng.randint(lo, max(lo, hi - 20))
    end = min(n, start + target_len)
    start = _snap_sentence_start(text, start)
    end = _snap_sentence_end(text, end, n)
    if end <= start:
        return None
    raw = text[start:end].strip()
    raw = re.sub(r"^[\s\W]+", "", raw)
    raw = re.sub(r"[\s\W]+$", "", raw)
    if len(raw) < low:
        return None
    if len(raw) > cap:
        cut = raw.rfind(". ", 0, cap)
        if cut > low:
            raw = raw[: cut + 1].strip()
        else:
            raw = raw[:cap].rstrip()
    if len(raw) < low:
        return None
    return raw


def _snippet_valid_for_middle_chunks_only(
    text: str,
    snippet: str,
    row_meta: dict,
    strategy_ids: tuple[str, ...],
) -> dict[str, int] | None:
    """Return chunk index per strategy in middle only, or None if invalid."""
    g = normalize_text(snippet)
    if not g:
        return None
    positions: dict[str, int] = {}
    for sid in strategy_ids:
        docs = chunk_one_document(text, dict(row_meta), sid)
        if len(docs) < 3:
            return None
        first = normalize_text(docs[0].page_content)
        last = normalize_text(docs[-1].page_content)
        if g in first or g in last:
            return None
        middle_hits = [
            i
            for i, d in enumerate(docs[1:-1], start=1)
            if g in normalize_text(d.page_content)
        ]
        if not middle_hits:
            return None
        positions[sid] = middle_hits[0]
    return positions


def _question_passes_validation(question: str, snippet: str) -> bool:
    q = question.strip()
    if len(q) < 15 or q.count("?") != 1:
        return False
    words = q.split()
    if len(words) < 12:
        return False
    ql = q.lower()
    for bad in _FORBIDDEN_Q_SUBSTRINGS:
        if bad in ql:
            return False
    n_snip = normalize_text(snippet)
    n_q = normalize_text(q.replace("?", ""))
    sq = set(re.findall(r"\w+", n_q))
    ss = set(re.findall(r"\w+", n_snip))
    if not sq:
        return False
    overlap = len(sq & ss) / len(sq)
    if overlap > 0.55:
        return False
    return True


def _answer_passes_heuristics(answer: str, snippet: str) -> bool:
    a = (answer or "").strip()
    if len(a) < 10:
        return False
    if len(a) > 500:
        return False
    aw = a.split()
    if len(aw) < 3:
        return False
    n_snip = normalize_text(snippet)
    n_a = normalize_text(a)
    sa = set(re.findall(r"\w+", n_a))
    ss = set(re.findall(r"\w+", n_snip))
    if not sa or not ss:
        return False
    if not (sa & ss):
        return False
    return True


def _strip_json_fence(content: str) -> str:
    c = content.strip()
    if c.startswith("```"):
        c = re.sub(r"^```\w*\n?", "", c)
        c = re.sub(r"\n?```$", "", c)
    return c.strip()


def _generate_question_and_answer(
    llm: ChatOllama,
    snippet: str,
    max_attempts: int = 3,
) -> tuple[str, str] | None:
    prompt = (
        "You are given an excerpt from an EU legal instrument (English).\n"
        "Write ONE specific factual question that can be answered ONLY by reading "
        "the ENTIRE excerpt together (not from a single name or phrase at the start).\n"
        "Also write a short factual answer (2–6 sentences) supported ONLY by the excerpt.\n"
        "Do not use the words: excerpt, passage, above, below, provided text.\n"
        "Reply with a single JSON object and nothing else, format: "
        '{"question": "...", "answer": "..."}\n\nEXCERPT:\n'
        + snippet[:2000]
    )
    for attempt_i in range(max_attempts):
        p = prompt
        if attempt_i > 0:
            p = (
                prompt
                + "\n\nThe previous JSON was rejected. Output one new JSON object "
                "meeting all instructions."
            )
        try:
            msg = llm.invoke([HumanMessage(content=p)])
            content = _strip_json_fence((msg.content or "").strip())
            data = json.loads(content)
            q = data.get("question")
            a = data.get("answer")
            if (
                isinstance(q, str)
                and isinstance(a, str)
                and _question_passes_validation(q, snippet)
                and _answer_passes_heuristics(a, snippet)
            ):
                return q.strip(), a.strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None


def _llm_validate_question_answer(
    llm: ChatOllama,
    snippet: str,
    question: str,
    answer: str,
    max_attempts: int = 2,
) -> dict[str, Any] | None:
    prompt = (
        "Given the EXCERPT, QUESTION, and PROPOSED_ANSWER, decide whether:\n"
        "- the question is specific and answerable from the excerpt alone;\n"
        "- the proposed answer is directly supported by the excerpt (no outside facts).\n"
        "Reply with ONLY one JSON object: "
        '{"supported": true or false, "reason": "short string"}\n\n'
        f"EXCERPT:\n{snippet[:2500]}\n\nQUESTION:\n{question}\n\nPROPOSED_ANSWER:\n{answer}\n"
    )
    for _ in range(max_attempts):
        try:
            msg = llm.invoke([HumanMessage(content=prompt)])
            content = _strip_json_fence((msg.content or "").strip())
            data = json.loads(content)
            sup = data.get("supported")
            reason = data.get("reason", "")
            if isinstance(sup, bool) and isinstance(reason, str):
                return {"supported": sup, "reason": reason}
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None


def generate_ground_truth_dedup(
    out_path: Path,
    n_target: int,
    seed: int,
    min_doc_chars: int,
    manifest_path: Path,
    train_path: Path,
    snippet_min_chars: int,
    snippet_max_chars: int,
    strategy_ids: tuple[str, ...],
    *,
    resume: bool = True,
    no_resume: bool = False,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
    append: bool = False,
    append_exclude_references: bool = False,
) -> None:
    out_path = Path(out_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    train_path = Path(train_path).resolve()
    ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path(out_path)

    if append and no_resume:
        raise SystemExit(
            "--append cannot be combined with --no-resume (that would delete the JSONL you are extending)."
        )

    if no_resume:
        resume = False
        if ckpt_path.is_file():
            ckpt_path.unlink()
            print(f"Removed checkpoint (--no-resume): {ckpt_path}")
        if out_path.is_file():
            out_path.unlink()
            print(f"Removed partial output (--no-resume): {out_path}")

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Run first:\n"
            "  python -m Scripts.eval.build_chunk_indices_dedup --top10"
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    allowed = {str(c) for c in manifest.get("celex_ids", []) if c}
    if not allowed:
        raise ValueError(f"{manifest_path} has no celex_ids; rebuild dedup indices.")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    manifest_sha256 = _sha256_file(manifest_path)
    train_sha256 = _sha256_file(train_path)
    train_st = train_path.stat()

    df = pd.read_json(train_path, lines=True)
    df["celex_id"] = df["celex_id"].apply(lambda x: str(x or ""))
    df = df[df["celex_id"].isin(allowed)]
    df = df[df["text"].astype(str).str.len() >= min_doc_chars]
    if len(df) == 0:
        raise ValueError(
            "No in-manifest documents meet min_doc_chars. Lower --min-doc-chars or rebuild manifest."
        )

    df = preprocess_for_rag(df)

    n_existing_on_disk = 0
    if append:
        if not out_path.is_file():
            raise FileNotFoundError(f"--append requires an existing JSONL at {out_path}")
        n_existing_on_disk = _count_jsonl_lines(out_path)
        if n_existing_on_disk >= n_target:
            print(
                f"Already have {n_existing_on_disk} rows (target {n_target}); nothing to append."
            )
            return
        existing_refs = _existing_references_from_jsonl(out_path)
        if append_exclude_references and existing_refs:
            before = len(df)
            df = df[~df["celex_id"].isin(existing_refs)]
            n_ex = len(existing_refs)
            print(
                f"Append mode: excluding {n_ex} CELEX id(s) already used as `reference` in "
                f"{out_path.name} ({before} -> {len(df)} candidate rows). "
                "Note: if this pass is interrupted, resume may fail until you remove the "
                "checkpoint; omit --exclude-used-references for easier resume."
            )
        if len(df) == 0:
            raise ValueError(
                "No candidate documents left after filters. Try lowering --min-doc-chars "
                "or omit --exclude-used-references."
            )

    df = df.reset_index(drop=True)
    n_df = len(df)

    run_meta: dict[str, Any] = {
        "out_path": str(out_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "train_path": str(train_path),
        "train_sha256": train_sha256,
        "train_stat": {"size": train_st.st_size, "mtime": int(train_st.st_mtime)},
        "n_df_rows": n_df,
        "seed": seed,
        "n_target": n_target,
        "min_doc_chars": min_doc_chars,
        "snippet_min_chars": snippet_min_chars,
        "snippet_max_chars": snippet_max_chars,
        "strategy_ids": list(strategy_ids),
        "first_last_chunk_exclusion": True,
        "append_mode": append,
        "append_exclude_references": append_exclude_references,
    }

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    order: list[int]
    pos: int
    attempt: int
    n_accepted: int

    loaded: dict[str, Any] | None = None
    if resume and ckpt_path.is_file():
        try:
            with open(ckpt_path, encoding="utf-8") as f:
                loaded = json.load(f)
        except (json.JSONDecodeError, OSError):
            loaded = None

    if loaded:
        disk_ver = int(loaded.get("version", 0))
        if disk_ver != CHECKPOINT_VERSION:
            if append and ckpt_path.is_file():
                ckpt_path.unlink()
                print(
                    f"Removed checkpoint (version {disk_ver} != {CHECKPOINT_VERSION}); "
                    "starting new --append pass."
                )
                loaded = None
            else:
                raise SystemExit(
                    f"Unsupported checkpoint version {loaded.get('version')!r} in {ckpt_path}. "
                    f"Delete that file or pass --append with adjusted flags, or use --no-resume."
                )

    if loaded and append:
        if loaded.get("append_mode") is not True or int(loaded.get("min_doc_chars", -1)) != int(
            min_doc_chars
        ):
            if ckpt_path.is_file():
                ckpt_path.unlink()
                print(
                    "Removed checkpoint (not an in-progress --append with the same "
                    "--min-doc-chars); starting a new append pass."
                )
            loaded = None

    if loaded:
        for k, v in run_meta.items():
            if loaded.get(k) != v:
                if append and ckpt_path.is_file():
                    ckpt_path.unlink()
                    print(
                        f"Removed checkpoint (mismatch on {k!r}); starting new --append pass."
                    )
                    loaded = None
                    break
                else:
                    raise SystemExit(
                        f"Checkpoint at {ckpt_path} does not match this run "
                        f"(field {k!r}: disk={loaded.get(k)!r} current={v!r}).\n"
                        "Use --no-resume to discard checkpoint and output, or restore matching inputs."
                    )

    if loaded:
        order = [int(x) for x in loaded["order"]]
        pos = int(loaded["next_pos"])
        attempt = int(loaded["attempt"])
        n_accepted = int(loaded["n_accepted"])
        _deserialize_rng(str(loaded["rng_state"]), rng)
        lines_on_disk = _count_jsonl_lines(out_path)
        if lines_on_disk != n_accepted:
            raise SystemExit(
                f"Checkpoint says n_accepted={n_accepted} but {out_path} has "
                f"{lines_on_disk} non-empty lines.\n"
                "Files are out of sync; use --no-resume to restart, or repair manually."
            )
        if pos > len(order):
            raise SystemExit(f"Invalid checkpoint next_pos={pos} (order length {len(order)}).")
        print(
            f"Resuming: {lines_on_disk} rows in {out_path.name}, "
            f"next shuffled-doc index {pos}/{len(order)}, attempt {attempt}"
        )
    else:
        if out_path.is_file() and out_path.stat().st_size > 0 and not append:
            raise SystemExit(
                f"{out_path} already has data but no usable checkpoint at {ckpt_path}.\n"
                "Use --no-resume to delete both and restart, or move the files aside."
            )
        if append:
            lines_now = _count_jsonl_lines(out_path)
            if lines_now != n_existing_on_disk:
                raise SystemExit(
                    f"--append: line count changed unexpectedly ({n_existing_on_disk} -> {lines_now}). "
                    "Stabilize the file or pick a different --out."
                )
            if ckpt_path.is_file():
                ckpt_path.unlink()
                print(f"Append mode: removed stale checkpoint: {ckpt_path}")
            n_accepted = n_existing_on_disk
            order = list(range(n_df))
            rng.shuffle(order)
            pos = 0
            attempt = 0
            payload = {
                "version": CHECKPOINT_VERSION,
                **run_meta,
                "order": order,
                "next_pos": pos,
                "attempt": attempt,
                "n_accepted": n_accepted,
                "rng_state": _serialize_rng(rng),
            }
            _atomic_write_json(ckpt_path, payload)
            print(
                f"Append mode: {n_existing_on_disk} existing rows; "
                f"adding up to {n_target - n_existing_on_disk} more (total target {n_target}). "
                f"Checkpoint: {ckpt_path}"
            )
        else:
            n_accepted = 0
            order = list(range(n_df))
            rng.shuffle(order)
            pos = 0
            attempt = 0
            payload = {
                "version": CHECKPOINT_VERSION,
                **run_meta,
                "order": order,
                "next_pos": pos,
                "attempt": attempt,
                "n_accepted": n_accepted,
                "rng_state": _serialize_rng(rng),
            }
            _atomic_write_json(ckpt_path, payload)
            print(f"Started fresh; checkpoint: {ckpt_path}")

    rem = max(1, n_target - n_accepted)
    max_attempts = max(rem * 150, 12000)
    stop_requested = False

    def _handle_stop(*_args: Any) -> None:
        nonlocal stop_requested
        stop_requested = True

    prev_sigint = signal.signal(signal.SIGINT, _handle_stop)

    def _persist_progress(pos_after: int) -> None:
        payload = {
            "version": CHECKPOINT_VERSION,
            **run_meta,
            "order": order,
            "next_pos": pos_after,
            "attempt": attempt,
            "n_accepted": n_accepted,
            "rng_state": _serialize_rng(rng),
        }
        _atomic_write_json(ckpt_path, payload)

    pbar = tqdm(
        total=len(order),
        initial=min(pos, len(order)),
        desc="Ground truth dedup candidates",
        unit="doc",
    )

    try:
        while pos < len(order):
            if stop_requested:
                _persist_progress(pos)
                print(
                    f"\nStopped. Checkpoint saved ({n_accepted} rows in {out_path}). "
                    "Re-run the same command to resume."
                )
                return
            if n_accepted >= n_target:
                break
            if attempt >= max_attempts:
                break

            attempt += 1
            idx = order[pos]
            row = df.iloc[idx]
            text = str(row["text"])
            celex = str(row.get("celex_id", "") or "")
            pbar.set_postfix(
                accepted=n_accepted,
                attempt=attempt,
                phase="scan",
                refresh=True,
            )

            if not celex:
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue

            meta = row_meta(row)

            snippet = _pick_snippet(text, rng, snippet_min_chars, snippet_max_chars)
            if not snippet or snippet not in text:
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue

            chunk_positions = _snippet_valid_for_middle_chunks_only(
                text, snippet, meta, strategy_ids
            )
            if not chunk_positions:
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue

            pbar.set_postfix(
                accepted=n_accepted,
                attempt=attempt,
                phase="llm",
                celex=celex[:12],
                refresh=True,
            )
            if verbose:
                tqdm.write(
                    f"LLM ({celex}): middle-chunk snippet ok; generating Q+A "
                    f"(snippet chars={len(snippet)})…"
                )

            qa = _generate_question_and_answer(llm, snippet)
            if not qa:
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue
            question, answer = qa

            val = _llm_validate_question_answer(llm, snippet, question, answer)
            if not val or not val.get("supported"):
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue

            rec = {
                "id": f"gt_dedup_{n_accepted:05d}",
                "question": question,
                "answer": answer,
                "reference": celex,
                "gold_snippet": snippet,
                "source_len_chars": len(text),
                "source_chunk_positions": chunk_positions,
                "question_validation": val,
            }
            _append_jsonl_record(out_path, rec)
            n_accepted += 1
            tqdm.write(f"Accepted {n_accepted}/{n_target}: {rec['id']} {celex}")

            pos += 1
            _persist_progress(pos)
            pbar.update(1)
    finally:
        pbar.close()
        signal.signal(signal.SIGINT, prev_sigint)

    if n_accepted >= n_target and ckpt_path.is_file():
        ckpt_path.unlink()
        print(f"Target reached; removed checkpoint: {ckpt_path}")

    if n_accepted < n_target:
        print(
            f"Warning: only {n_accepted} accepted (target {n_target}). "
            "Try lowering --min-doc-chars, increasing in-manifest docs, or more attempts. "
            "To extend an existing file without deleting it, use: "
            "`python -m Scripts.eval.ground_truth_generate_dedup --append --n 100 --min-doc-chars 800` "
            "(add `--exclude-used-references` to avoid reusing CELEX ids already in the file)."
        )
    print(f"Ground truth file: {out_path} ({n_accepted} records)")


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate dedup ground-truth JSONL for top-10 eval. "
            "Requires eval_corpus_manifest_dedup.json from build_chunk_indices_dedup."
        )
    )
    p.add_argument("--n", type=int, default=100, help="Target number of accepted rows")
    p.add_argument(
        "--out",
        type=Path,
        default=DEDUP_GROUND_TRUTH,
        help="Output JSONL path",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-doc-chars", type=int, default=1000)
    p.add_argument("--snippet-min-chars", type=int, default=280)
    p.add_argument("--snippet-max-chars", type=int, default=450)
    p.add_argument(
        "--train",
        type=Path,
        default=DEDUP_TRAIN_JSONL,
        help="Dedup train JSONL (default: Data/train_dedup.jsonl)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=DEDUP_MANIFEST,
        help="Dedup corpus manifest",
    )
    p.add_argument(
        "--append",
        action="store_true",
        help=(
            "Extend existing JSONL toward total --n rows (e.g. after a partial run). "
            "Drops an incompatible checkpoint, keeps existing lines, then continues with "
            "the same or lower --min-doc-chars. By default still considers all CELEX ids; "
            "pass --exclude-used-references to sample only documents not yet used as `reference`."
        ),
    )
    p.add_argument(
        "--exclude-used-references",
        action="store_true",
        help=(
            "With --append, exclude any CELEX id that already appears as `reference` in the "
            "JSONL (shrinks the candidate pool; resuming mid-run may require deleting the "
            "checkpoint if n_df no longer matches)."
        ),
    )
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()
    if args.snippet_min_chars > args.snippet_max_chars:
        raise SystemExit("--snippet-min-chars must be <= --snippet-max-chars")
    if args.snippet_max_chars > _SNIPPET_HARD_CAP:
        raise SystemExit(
            f"--snippet-max-chars cannot exceed {_SNIPPET_HARD_CAP} "
            "(must fit inside smallest eval chunk strategy)."
        )
    generate_ground_truth_dedup(
        out_path=args.out,
        n_target=args.n,
        seed=args.seed,
        min_doc_chars=args.min_doc_chars,
        manifest_path=args.manifest,
        train_path=args.train,
        snippet_min_chars=args.snippet_min_chars,
        snippet_max_chars=args.snippet_max_chars,
        strategy_ids=TOP10_STRATEGY_IDS,
        resume=not args.no_resume,
        no_resume=args.no_resume,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
        append=args.append,
        append_exclude_references=args.exclude_used_references,
    )


if __name__ == "__main__":
    main()
