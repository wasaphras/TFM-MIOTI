"""
Generate ground_truth.jsonl: question, reference (celex_id), gold_snippet.

Requires eval_corpus_manifest.json from build_chunk_indices. Validates each
snippet against all 10 chunk strategies on the source document.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

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
) -> None:
    """Sample snippets only from documents listed in the eval corpus manifest (same scope as indices)."""
    rng = random.Random(seed)
    manifest_path = Path(manifest_path)
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

    train_path = config.TRAIN_JSONL
    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")

    df = pd.read_json(train_path, lines=True)
    df["celex_id"] = df["celex_id"].apply(lambda x: str(x or ""))
    df = df[df["celex_id"].isin(allowed)]
    df = df[df["text"].astype(str).str.len() >= min_doc_chars]
    if len(df) == 0:
        raise ValueError(
            "No in-manifest documents meet min_doc_chars. Lower --min-doc-chars or rebuild with more docs."
        )

    df = preprocess_for_rag(df)

    llm = ChatOllama(model=config.LLM_MODEL, temperature=0.1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    accepted: list[dict] = []
    order = list(range(len(df)))
    rng.shuffle(order)
    attempt = 0
    max_attempts = max(n_target * 80, 5000)

    pbar = tqdm(order, desc="Ground truth candidates", unit="doc")
    for idx in pbar:
        if len(accepted) >= n_target:
            break
        if attempt >= max_attempts:
            break
        attempt += 1
        pbar.set_postfix(accepted=len(accepted), refresh=False)
        row = df.iloc[idx]
        text = str(row["text"])
        celex = str(row.get("celex_id", "") or "")
        if not celex:
            continue

        meta = row_meta(row)

        snippet = _pick_snippet(text, rng, snippet_min_chars, snippet_max_chars)
        if not snippet or snippet not in text:
            continue
        if not _snippet_valid_for_all_strategies(text, snippet, meta):
            continue

        question = _generate_question(llm, snippet)
        if not question:
            continue

        accepted.append(
            {
                "id": f"gt_{len(accepted):05d}",
                "question": question,
                "reference": celex,
                "gold_snippet": snippet,
                "source_len_chars": len(text),
            }
        )
        tqdm.write(f"Accepted {len(accepted)}/{n_target}: {accepted[-1]['id']} {celex}")

    pbar.close()
    if len(accepted) < n_target:
        print(
            f"Warning: only {len(accepted)} accepted (target {n_target}). "
            "Try lowering --min-doc-chars, increasing in-manifest docs (rebuild without --limit), or more attempts."
        )

    with open(out_path, "w", encoding="utf-8") as f:
        for row in tqdm(accepted, desc="Writing ground_truth.jsonl", unit="row"):
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(accepted)} records to {out_path}")


def main():
    p = argparse.ArgumentParser(
        description=(
            "Generate ground_truth.jsonl for RAG eval. "
            "Requires Data/eval_corpus_manifest.json from build_chunk_indices "
            "(same document scope as Chroma + chunks_*.jsonl)."
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
    )


if __name__ == "__main__":
    main()
