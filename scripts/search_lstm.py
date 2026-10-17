"""m3: hyperparameter search for the LSTM, one factor at a time, on validation macro-F1 (seed 0).

Each step changes one factor of the current best configuration; the change is kept only if validation
macro-F1 improves. Runs are written to results/search/<name>/ (validation only, the test split is never
read) and the table to results/search/lstm_search.csv.

    python scripts/search_lstm.py              # about 3 hours on the Mac (MPS)
    FAST_RUN=1 python scripts/search_lstm.py   # a few minutes
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import pandas as pd

from ledgar.config import ROOT, Config

STEPS = [
    ("init", "glove", {"glove": True, "emb_dim": 50}),
    ("direction", "bilstm", {"bidirectional": True}),
    ("hidden size", "h256", {"hidden": 256}),
    ("learning rate", "lr1e-3", {"lstm_lr": 1e-3}),
    ("class weights", "sqrt_weights", {"lstm_class_weight": "sqrt"}),
    ("dropout", "dropout0.5", {"dropout": 0.5}),
]


def flags(overrides: dict) -> list[str]:
    out = []
    for key, value in overrides.items():
        opt = "--" + key.replace("_", "-")
        out += [opt] if value is True else ["--no-" + opt[2:]] if value is False else [opt, str(value)]
    return out


def run(name: str, overrides: dict, extra: list[str]) -> dict:
    cmd = [sys.executable, str(ROOT / "scripts" / "train.py"), "--model", "lstm", "--search", "--name", name]
    subprocess.run([*cmd, *flags(overrides), *extra], check=True, cwd=ROOT)
    cfg = Config.from_env()
    out_dir = (
        cfg.path(cfg.out_dir) if "--out-dir" not in extra else ROOT / extra[extra.index("--out-dir") + 1]
    )
    return json.loads((out_dir / "search" / name / "metrics.json").read_text())


def main() -> None:
    extra = sys.argv[1:]  # passed through to train.py, e.g. --device cpu
    best, rows = {}, []
    start = time.perf_counter()
    m = run("lstm_base", best, extra)
    best_f1 = m["val"]["macro_f1"]
    rows.append({"step": "start", "run": "lstm_base", "change": "-", **_row(m), "kept": True})
    for step, name, change in STEPS:
        candidate = {**best, **change}
        m = run(f"lstm_{name}", candidate, extra)
        kept = m["val"]["macro_f1"] > best_f1
        rows.append(
            {"step": step, "run": f"lstm_{name}", "change": json.dumps(change), **_row(m), "kept": kept}
        )
        if kept:
            best, best_f1 = candidate, m["val"]["macro_f1"]
        print(pd.DataFrame(rows).to_string(), flush=True)
    cfg = Config.from_env()
    table = pd.DataFrame(rows)
    table.to_csv(cfg.path(cfg.out_dir, "search", "lstm_search.csv"), index=False)
    print(f"\nbest configuration: {json.dumps(best)} -> validation macro-F1 {best_f1:.4f}")
    print(f"search time: {(time.perf_counter() - start) / 60:.0f} min")


def _row(m: dict) -> dict:
    return {
        "val_macro_f1": round(m["val"]["macro_f1"], 4),
        "val_accuracy": round(m["val"]["accuracy"], 4),
        "best_epoch": m["best_epoch"],
        "trainable_params": m["trainable_params"],
        "glove_coverage": round(m.get("glove_coverage", float("nan")), 3),
    }


if __name__ == "__main__":
    main()
