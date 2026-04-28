"""Run eval1 -> eval4 sequentially; forwards CLI args to each module."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    modules = [
        "Scripts.eval.top10.run_eval1_baseline",
        "Scripts.eval.top10.run_eval2_neighbors",
        "Scripts.eval.top10.run_eval3_enhanced",
        "Scripts.eval.top10.run_eval4_multiquery",
    ]
    extra = sys.argv[1:]
    for mod in modules:
        cmd = [sys.executable, "-m", mod, *extra]
        print(f"\n>>> {' '.join(cmd)}\n")
        r = subprocess.run(cmd)
        if r.returncode != 0:
            sys.exit(r.returncode)
    print("\nAll four evals finished successfully.")


if __name__ == "__main__":
    main()
