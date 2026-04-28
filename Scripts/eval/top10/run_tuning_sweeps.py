"""
Run focused tuning sweeps via subprocess (conda env recommended).

Examples::

    conda run -n Data python -m Scripts.eval.top10.run_tuning_sweeps candidate
    conda run -n Data python -m Scripts.eval.top10.run_tuning_sweeps candidate --execute --limit-queries 5
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from ... import config

PAIR = "len_1000_o100:hyb_rrf_k60_ce_r50"
SWEEP_ROOT = config.DATA_DIR / "eval_top10" / "sweeps"


def _run_module(mod: str, extra: list[str], execute: bool) -> int:
    cmd = [sys.executable, "-m", mod, *extra]
    print(" ", " ".join(cmd))
    if not execute:
        return 0
    r = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT))
    return int(r.returncode)


def _read_summary_row(out_dir: Path, chunk: str, retriever: str) -> dict[str, str] | None:
    p = out_dir / "results_summary.csv"
    if not p.is_file():
        return None
    with open(p, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("chunk_strategy") == chunk and row.get("retriever") == retriever:
                return dict(row)
    return None


def cmd_candidate(args: argparse.Namespace) -> int:
    chunk, retriever = PAIR.split(":", 1)
    ks = [int(x) for x in args.candidate_ks.split(",") if x.strip()]
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for k in ks:
        for mode in ("eval1_baseline", "eval2_neighbors"):
            mod = f"Scripts.eval.top10.run_{mode}"
            out = SWEEP_ROOT / f"{mode}_cand{k}"
            extra = [
                "--pairs",
                PAIR,
                "--candidate-k",
                str(k),
                "--out",
                str(out),
                "--no-resume",
            ]
            if args.limit_queries:
                extra += ["--limit-queries", str(args.limit_queries)]
            if args.ground_truth:
                extra += ["--ground-truth", str(args.ground_truth)]
            if args.manifest:
                extra += ["--manifest", str(args.manifest)]
            rc = _run_module(mod, extra, args.execute)
            if rc != 0:
                return rc
            row = _read_summary_row(out, chunk, retriever)
            if row:
                row["sweep_mode"] = mode
                row["candidate_k"] = k
                rows.append(row)

    if rows and args.execute:
        out_csv = SWEEP_ROOT / "candidate_k_sweep_summary.csv"
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
    return 0


def _parse_seed_tops(spec: str) -> list[int | None]:
    out: list[int | None] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.lower() == "all":
            out.append(None)
        else:
            out.append(int(tok))
    return out


def cmd_neighbor(args: argparse.Namespace) -> int:
    chunk, retriever = PAIR.split(":", 1)
    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    offset_specs = [s.strip() for s in args.offset_sets.split("|") if s.strip()]
    seed_tops = _parse_seed_tops(args.seed_tops)

    for off in offset_specs:
        for st in seed_tops:
            st_label = "all" if st is None else str(st)
            label = f"off_{off.replace(',', '_')}_seed{st_label}"
            out = SWEEP_ROOT / f"eval2_neighbors_{label}"
            extra = [
                "--pairs",
                PAIR,
                "--neighbor-offsets",
                off,
                "--candidate-k",
                str(args.candidate_k),
                "--out",
                str(out),
                "--no-resume",
            ]
            if st is not None:
                extra += ["--neighbor-seed-top", str(st)]
            if args.limit_queries:
                extra += ["--limit-queries", str(args.limit_queries)]
            if args.ground_truth:
                extra += ["--ground-truth", str(args.ground_truth)]
            if args.manifest:
                extra += ["--manifest", str(args.manifest)]
            rc = _run_module("Scripts.eval.top10.run_eval2_neighbors", extra, args.execute)
            if rc != 0:
                return rc
            row = _read_summary_row(out, chunk, retriever)
            if row:
                row["neighbor_offsets"] = off
                row["neighbor_seed_top"] = "" if st is None else st
                rows.append(row)

    if rows and args.execute:
        out_csv = SWEEP_ROOT / "neighbor_param_sweep_summary.csv"
        keys = list(rows[0].keys())
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute", action="store_true", help="Run evals (default is dry-run print)")
    p.add_argument("--limit-queries", type=int, default=None)
    p.add_argument("--ground-truth", type=Path, default=None)
    p.add_argument("--manifest", type=Path, default=None)

    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("candidate", help="Sweep candidate_k on baseline vs neighbors")
    pc.add_argument(
        "--candidate-ks",
        type=str,
        default="50,100,150,200,300",
        help="Comma-separated candidate_k values",
    )
    pc.set_defaults(func=cmd_candidate)

    pn = sub.add_parser("neighbor", help="Sweep neighbor offsets x seed-top (eval2 only)")
    pn.add_argument("--candidate-k", type=int, default=100)
    pn.add_argument(
        "--offset-sets",
        type=str,
        default="-1,1|-2,-1,1,2|-3,-2,-1,1,2,3",
        help="Pipe-separated offset specs, e.g. -1,1|-2,-1,1,2",
    )
    pn.add_argument(
        "--seed-tops",
        type=str,
        default="20,50,100,all",
        help="Comma-separated neighbor-seed-top values; use 'all' for no limit (legacy expansion).",
    )
    pn.set_defaults(func=cmd_neighbor)

    args = p.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
