"""Paired comparison of two final runs on the test split (S06 p22-24): McNemar and paired bootstrap.

    python scripts/evaluate.py --a lora_lr5e-4_seed0 --b bert_lr3e-5_seed0

Writes reports/comparisons/<a>__vs__<b>.json and prints a one-line summary. Both runs must have predicted the
same test clauses, in the same order.
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from ledgar import runs, stats
from ledgar.config import ROOT


def compare_runs(a: str, b: str, n_boot: int = 1000, seed: int = 0, results: str = "results") -> dict:
    pa = runs.load_predictions(ROOT / results / a, "test")
    pb = runs.load_predictions(ROOT / results / b, "test")
    if not (np.array_equal(pa["index"], pb["index"]) and np.array_equal(pa["y_true"], pb["y_true"])):
        raise SystemExit("the two runs did not predict the same test clauses")
    return {"a": a, "b": b, **stats.compare(pa["y_true"], pa["y_pred"], pb["y_pred"], n_boot, seed)}


def summary(r: dict) -> str:
    m, bs = r["mcnemar"], r["bootstrap"]
    return (
        f"{r['a']} vs {r['b']}: macro-F1 {r['macro_f1'][0]:.4f} vs {r['macro_f1'][1]:.4f} "
        f"(diff {bs['diff']:+.4f}, 95 % CI [{bs['low']:+.4f}, {bs['high']:+.4f}]) | "
        f"accuracy {r['accuracy'][0]:.4f} vs {r['accuracy'][1]:.4f}, b = {r['contingency']['only_a']}, "
        f"c = {r['contingency']['only_b']}, McNemar p = {m['p']:.4g}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = compare_runs(args.a, args.b, args.n_boot, args.seed)
    out = ROOT / "reports" / "comparisons" / f"{args.a}__vs__{args.b}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(summary(result))


if __name__ == "__main__":
    main()
