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
from ledgar.config import ROOT, Config


def _seed_siblings(name: str, base_dir) -> list[str]:
    if not name.endswith("_seed0"):
        return [name]
    prefix = name[: -len("_seed0")]
    siblings = [f"{prefix}_seed{s}" for s in (0, 1, 2)]
    return siblings if all((base_dir / s).exists() for s in siblings) else [name]


def compare_runs(a: str, b: str, n_boot: int = 1000, seed: int = 0, results: str = "results") -> dict:
    base_dir = ROOT / results
    pa = runs.load_predictions(base_dir / a, "test")
    pb = runs.load_predictions(base_dir / b, "test")
    if not (np.array_equal(pa["index"], pb["index"]) and np.array_equal(pa["y_true"], pb["y_true"])):
        raise SystemExit("the two runs did not predict the same test clauses")
    out = {"a": a, "b": b, **stats.compare(pa["y_true"], pa["y_pred"], pb["y_pred"], n_boot, seed)}
    seeds_a, seeds_b = _seed_siblings(a, base_dir), _seed_siblings(b, base_dir)
    if len(seeds_a) > 1 or len(seeds_b) > 1:
        preds_a = [runs.load_predictions(base_dir / s, "test")["y_pred"] for s in seeds_a]
        preds_b = [runs.load_predictions(base_dir / s, "test")["y_pred"] for s in seeds_b]
        out["multi_seed"] = {
            "runs_a": seeds_a,
            "runs_b": seeds_b,
            **stats.compare_seeds(pa["y_true"], preds_a, preds_b, n_boot, seed),
        }
    return out


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
    cfg, args = Config.from_args(argv, parser)
    result = compare_runs(args.a, args.b, cfg.n_boot, cfg.seed, cfg.out_dir)
    out = cfg.path(cfg.reports_dir, "comparisons", f"{args.a}__vs__{args.b}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(summary(result))


if __name__ == "__main__":
    main()
