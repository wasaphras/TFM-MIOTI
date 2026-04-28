"""
Generate ground_truth.jsonl: question, reference (celex_id), gold_snippet.

Requires eval_corpus_manifest.json from build_chunk_indices. Validates each
snippet against all 10 chunk strategies on the source document.
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
from ..chunking_strategies import CHUNK_STRATEGY_IDS, chunk_one_document, row_meta
from ..preprocess import preprocess_for_rag
from .metrics import normalize_text

# Smallest eval chunk target (~len_500_o50); snippet must fit in one chunk.
_SNIPPET_HARD_CAP = 480

CHECKPOINT_VERSION = 1


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


def _snippet_valid_for_all_strategies(
    text: str,
    snippet: str,
    row_meta: dict,
) -> bool:
    g = normalize_text(snippet)
    if not g:
        return False
    for sid in CHUNK_STRATEGY_IDS:
        docs = chunk_one_document(text, dict(row_meta), sid)
        ok = any(g in normalize_text(d.page_content) for d in docs)
        if not ok:
            return False
    return True


def _question_passes_validation(question: str, snippet: str) -> bool:
    q = question.strip()
    if len(q) < 15 or q.count("?") != 1:
        return False
    words = q.split()
    if len(words) < 12:
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


def _generate_question(
    llm: ChatOllama,
    snippet: str,
    max_attempts: int = 3,
) -> str | None:
    prompt = (
        "You are given an excerpt from an EU legal instrument (English).\n"
        "Write ONE specific factual question that can be answered ONLY by reading "
        "the ENTIRE excerpt together (not from a single name or phrase at the start).\n"
        "Use clear referents where helpful (e.g. this Regulation, the body named here, "
        "the obligation described above). Do not paste long quotes from the excerpt.\n"
        "Reply with a single JSON object and nothing else, format: "
        '{"question": "..."}\n\nEXCERPT:\n'
        + snippet[:2000]
    )
    for attempt_i in range(max_attempts):
        p = prompt
        if attempt_i > 0:
            p = (
                prompt
                + "\n\nThe previous JSON was rejected (too short, wrong format, or too similar to the excerpt). "
                "Output one new question meeting all instructions."
            )
        try:
            msg = llm.invoke([HumanMessage(content=p)])
            content_i = (msg.content or "").strip()
            content = content_i
            if content.startswith("```"):
                content = re.sub(r"^```\w*\n?", "", content)
                content = re.sub(r"\n?```$", "", content)
            data = json.loads(content)
            q = data.get("question")
            if isinstance(q, str) and _question_passes_validation(q, snippet):
                return q.strip()
        except (json.JSONDecodeError, TypeError, AttributeError):
            continue
    return None


def generate_ground_truth(
    out_path: Path,
    n_target: int,
    seed: int,
    min_doc_chars: int,
    manifest_path: Path,
    snippet_min_chars: int,
    snippet_max_chars: int,
    *,
    resume: bool = True,
    no_resume: bool = False,
    checkpoint_path: Path | None = None,
    verbose: bool = False,
) -> None:
    """Sample snippets only from documents listed in the eval corpus manifest (same scope as indices).

    Each accepted row is appended to ``out_path`` immediately (fsync). A JSON checkpoint stores the
    shuffled document order, RNG state, and the next index in that order after **every** document
    tried, so stopping and re-running the same command resumes without redoing accepted rows.

    Set ``no_resume=True`` to delete partial ``out_path`` and its checkpoint and start over.
    """
    out_path = Path(out_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    ckpt_path = Path(checkpoint_path) if checkpoint_path else _default_checkpoint_path(out_path)
    train_path = Path(config.TRAIN_JSONL).resolve()

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
            "  python -m Scripts.eval.build_chunk_indices --all [--limit N]\n"
            "That writes the manifest of documents used for chunking and Chroma."
        )

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    allowed = {str(c) for c in manifest.get("celex_ids", []) if c}
    if not allowed:
        raise ValueError(f"{manifest_path} has no celex_ids; rebuild indices.")

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    manifest_sha256 = _sha256_file(manifest_path)
    train_st = train_path.stat()

    df = pd.read_json(train_path, lines=True)
    df["celex_id"] = df["celex_id"].apply(lambda x: str(x or ""))
    df = df[df["celex_id"].isin(allowed)]
    df = df[df["text"].astype(str).str.len() >= min_doc_chars]
    if len(df) == 0:
        raise ValueError(
            "No in-manifest documents meet min_doc_chars. Lower --min-doc-chars or rebuild with more docs."
        )

    df = preprocess_for_rag(df)
    n_df = len(df)

    run_meta = {
        "out_path": str(out_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "train_path": str(train_path),
        "train_stat": {"size": train_st.st_size, "mtime": int(train_st.st_mtime)},
        "n_df_rows": n_df,
        "seed": seed,
        "n_target": n_target,
        "min_doc_chars": min_doc_chars,
        "snippet_min_chars": snippet_min_chars,
        "snippet_max_chars": snippet_max_chars,
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
        if int(loaded.get("version", 0)) != CHECKPOINT_VERSION:
            raise SystemExit(
                f"Unsupported checkpoint version {loaded.get('version')!r} in {ckpt_path}. "
                "Use --no-resume to start over."
            )
        for k, v in run_meta.items():
            if loaded.get(k) != v:
                raise SystemExit(
                    f"Checkpoint at {ckpt_path} does not match this run "
                    f"(field {k!r}: disk={loaded.get(k)!r} current={v!r}).\n"
                    "Use --no-resume to discard checkpoint and output, or restore matching inputs."
                )
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
        if out_path.is_file() and out_path.stat().st_size > 0:
            raise SystemExit(
                f"{out_path} already has data but no usable checkpoint at {ckpt_path}.\n"
                "Use --no-resume to delete both and restart, or move the files aside."
            )
        order = list(range(n_df))
        rng.shuffle(order)
        pos = 0
        attempt = 0
        n_accepted = 0
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

    max_attempts = max(n_target * 80, 5000)
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
        desc="Ground truth candidates",
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
            if not _snippet_valid_for_all_strategies(text, snippet, meta):
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
                    f"LLM ({celex}): validating snippet in all chunk strategies passed; "
                    f"generating question (snippet chars={len(snippet)})…"
                )

            question = _generate_question(llm, snippet)
            if not question:
                pos += 1
                _persist_progress(pos)
                pbar.update(1)
                continue

            rec = {
                "id": f"gt_{n_accepted:05d}",
                "question": question,
                "reference": celex,
                "gold_snippet": snippet,
                "source_len_chars": len(text),
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
            "Try lowering --min-doc-chars, increasing in-manifest docs (rebuild without --limit), or more attempts."
        )
    print(f"Ground truth file: {out_path} ({n_accepted} records)")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Generate ground_truth.jsonl for RAG eval. "
            "Requires Data/eval_corpus_manifest.json from build_chunk_indices "
            "(same document scope as Chroma + chunks_*.jsonl). "
            "Appends each accepted row immediately; use the same command to resume after stop/crash."
        )
    )
    p.add_argument("--n", type=int, default=120, help="Target number of accepted rows")
    p.add_argument(
        "--out",
        type=Path,
        default=config.GROUND_TRUTH_JSONL,
        help="Output JSONL path",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-doc-chars", type=int, default=1500)
    p.add_argument(
        "--snippet-min-chars",
        type=int,
        default=280,
        help="Target minimum gold snippet length (capped so snippets fit len_500 chunks)",
    )
    p.add_argument(
        "--snippet-max-chars",
        type=int,
        default=450,
        help="Target maximum gold snippet length (hard cap 480 for smallest chunk strategy)",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=config.EVAL_CORPUS_MANIFEST,
        help="Corpus manifest from build_chunk_indices (default: Data/eval_corpus_manifest.json)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete checkpoint and output file and start from scratch",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint JSON path (default: <out>.checkpoint.json next to the JSONL)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Log when entering LLM question generation (shows work during slow steps)",
    )
    args = p.parse_args()
    if args.snippet_min_chars > args.snippet_max_chars:
        raise SystemExit("--snippet-min-chars must be <= --snippet-max-chars")
    if args.snippet_max_chars > _SNIPPET_HARD_CAP:
        raise SystemExit(
            f"--snippet-max-chars cannot exceed {_SNIPPET_HARD_CAP} "
            "(must fit inside smallest eval chunk strategy)."
        )
    generate_ground_truth(
        out_path=args.out,
        n_target=args.n,
        seed=args.seed,
        min_doc_chars=args.min_doc_chars,
        manifest_path=args.manifest,
        snippet_min_chars=args.snippet_min_chars,
        snippet_max_chars=args.snippet_max_chars,
        resume=not args.no_resume,
        no_resume=args.no_resume,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
