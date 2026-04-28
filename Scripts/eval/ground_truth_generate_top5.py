"""
Generate ground truth disjoint from an existing GT file (e.g. Data/ground_truth.jsonl).

Writes a filtered manifest (original corpus minus excluded CELEX ids) and calls
``ground_truth_generate.generate_ground_truth`` — same sampling/LLM logic as the main script.
Does not modify the original manifest or the original ground truth file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .. import config
from .ground_truth_generate import _SNIPPET_HARD_CAP, generate_ground_truth

DEFAULT_EXCLUDE_FROM = config.GROUND_TRUTH_JSONL
DEFAULT_FILTERED_MANIFEST = config.DATA_DIR / "eval_corpus_manifest_top5.json"
DEFAULT_OUT = config.DATA_DIR / "ground_truth_top5_1000.jsonl"
DEFAULT_SEED = 2026


def _load_excluded_celex_ids(path: Path) -> set[str]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing exclude-from file {path}. Pass --exclude-from to a valid JSONL "
            "or generate the original ground truth first."
        )
    out: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            ref = rec.get("reference")
            if ref:
                out.add(str(ref))
    return out


def _write_filtered_manifest(
    source_manifest: Path,
    excluded_celex: set[str],
    out_path: Path,
) -> None:
    source_manifest = Path(source_manifest)
    if not source_manifest.is_file():
        raise FileNotFoundError(
            f"Missing {source_manifest}. Run build_chunk_indices first."
        )
    with open(source_manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    celex_ids = manifest.get("celex_ids", [])
    filtered = [str(c) for c in celex_ids if c and str(c) not in excluded_celex]
    if not filtered:
        raise SystemExit(
            "Filtered manifest would be empty (all CELEX ids excluded). "
            "Check --exclude-from or use a smaller exclusion set."
        )
    new_manifest = dict(manifest)
    new_manifest["celex_ids"] = filtered
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(new_manifest, f, indent=2)
        f.write("\n")
    print(
        f"Wrote filtered manifest: {out_path} "
        f"({len(filtered)} celex_ids, excluded {len(excluded_celex)} from prior GT)"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Generate ground_truth JSONL disjoint from references in --exclude-from. "
            "Uses a filtered copy of the eval corpus manifest (does not overwrite the original)."
        )
    )
    p.add_argument("--n", type=int, default=1000, help="Target number of accepted rows")
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSONL path",
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--min-doc-chars", type=int, default=1500)
    p.add_argument(
        "--snippet-min-chars",
        type=int,
        default=280,
    )
    p.add_argument(
        "--snippet-max-chars",
        type=int,
        default=450,
    )
    p.add_argument(
        "--exclude-from",
        type=Path,
        default=DEFAULT_EXCLUDE_FROM,
        help="JSONL whose `reference` CELEX ids are removed from the sampling manifest",
    )
    p.add_argument(
        "--source-manifest",
        type=Path,
        default=config.EVAL_CORPUS_MANIFEST,
        help="Full eval corpus manifest (default: Data/eval_corpus_manifest.json)",
    )
    p.add_argument(
        "--filtered-manifest-out",
        type=Path,
        default=DEFAULT_FILTERED_MANIFEST,
        help="Where to write the CELEX-filtered manifest used for sampling",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="Delete ground-truth JSONL + checkpoint and start from scratch (keeps filtered manifest)",
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
        help="Log when calling the LLM for question generation (useful if progress looks stuck)",
    )
    args = p.parse_args()

    if args.snippet_min_chars > args.snippet_max_chars:
        raise SystemExit("--snippet-min-chars must be <= --snippet-max-chars")
    if args.snippet_max_chars > _SNIPPET_HARD_CAP:
        raise SystemExit(
            f"--snippet-max-chars cannot exceed {_SNIPPET_HARD_CAP} "
            "(must fit inside smallest eval chunk strategy)."
        )

    excluded = _load_excluded_celex_ids(args.exclude_from)
    _write_filtered_manifest(
        args.source_manifest,
        excluded,
        args.filtered_manifest_out,
    )

    generate_ground_truth(
        out_path=args.out,
        n_target=args.n,
        seed=args.seed,
        min_doc_chars=args.min_doc_chars,
        manifest_path=args.filtered_manifest_out,
        snippet_min_chars=args.snippet_min_chars,
        snippet_max_chars=args.snippet_max_chars,
        resume=not args.no_resume,
        no_resume=args.no_resume,
        checkpoint_path=args.checkpoint,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
