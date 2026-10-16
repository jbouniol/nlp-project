"""Run bookkeeping: one directory per run under results/, readable by people and by scripts."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ledgar.config import ROOT, Config

logger = logging.getLogger(__name__)

REQUIRED_FILES = (
    "config.json",
    "metrics.json",
    "pred_val.npz",
    "pred_test.npz",
    "log_history.json",
    "manifest.json",
)


def save_run(
    run_dir: str | Path,
    cfg: Config,
    metrics: dict[str, Any],
    *,
    log_history: list[dict[str, Any]] | None = None,
    duration_s: float | None = None,
) -> Path:
    """Write config.json, metrics.json, log_history.json and manifest.json.

    `metrics` is merged into an existing metrics.json, so the test metrics can be added by the
    final "predict & save" cell without rewriting the validation ones.

    Args:
        run_dir: Directory of the run, created if needed.
        cfg: The Config used by the run.
        metrics: For instance {"val": {"accuracy": .., "macro_f1": ..}, "trainable_params": ..}.
        log_history: Per-step or per-epoch training log (Trainer `log_history` or our own).
        duration_s: Training time in seconds.

    Returns:
        The run directory.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "config.json", cfg.to_dict())
    old = _read_json(run_dir / "metrics.json") if (run_dir / "metrics.json").exists() else {}
    _write_json(run_dir / "metrics.json", _deep_merge(old, metrics))
    if log_history is not None or not (run_dir / "log_history.json").exists():
        _write_json(run_dir / "log_history.json", log_history or [])
    _write_json(run_dir / "manifest.json", make_manifest(cfg, duration_s, previous=run_dir / "manifest.json"))
    logger.info("run saved to %s", run_dir)
    return run_dir


def save_predictions(
    run_dir: str | Path,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    logits: np.ndarray | None = None,
    index: np.ndarray | None = None,
    label_names: list[str] | None = None,
) -> Path:
    """Write pred_<split>.npz (index, y_true, y_pred, logits) and a readable pred_<split>.csv.

    `index` is the position of each row in the official split, so that the bias audit can join the
    predictions back to the raw clause text.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    index = np.arange(len(y_true)) if index is None else np.asarray(index)
    arrays = {"index": index, "y_true": y_true, "y_pred": y_pred}
    if logits is not None:
        arrays["logits"] = np.asarray(logits, dtype=np.float32)
    path = run_dir / f"pred_{split}.npz"
    np.savez_compressed(path, **arrays)
    table = pd.DataFrame({"index": index, "y_true": y_true, "y_pred": y_pred})
    if label_names is not None:
        table["true_label"] = [label_names[i] for i in y_true]
        table["pred_label"] = [label_names[i] for i in y_pred]
    if logits is not None:
        z = arrays["logits"] - arrays["logits"].max(axis=1, keepdims=True)
        table["prob_max"] = (np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)).max(axis=1).round(4)
    table.to_csv(run_dir / f"pred_{split}.csv", index=False)
    return path


def load_predictions(run_dir: str | Path, split: str) -> dict[str, np.ndarray]:
    """Read pred_<split>.npz back as a dictionary of arrays."""
    with np.load(Path(run_dir) / f"pred_{split}.npz") as data:
        return {k: data[k] for k in data.files}


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Read config, metrics and manifest of a run."""
    run_dir = Path(run_dir)
    return {name: _read_json(run_dir / f"{name}.json") for name in ("config", "metrics", "manifest")}


def missing_files(run_dir: str | Path) -> list[str]:
    """Names of the required artefacts that a run directory does not have yet."""
    return [name for name in REQUIRED_FILES if not (Path(run_dir) / name).exists()]


def results_table(results_dir: str | Path | None = None, *, strict: bool = True) -> pd.DataFrame:
    """One row per run found under results_dir, built only from the files of each run.

    Args:
        results_dir: Defaults to `<repo>/results`.
        strict: Raise if a run misses one of REQUIRED_FILES (traceability gate), else skip it.

    Returns:
        DataFrame with run, model, seed, trainable params, time, val / test macro-F1 and accuracy.
    """
    results_dir = Path(results_dir) if results_dir is not None else ROOT / "results"
    rows = []
    for run_dir in sorted(p.parent for p in results_dir.glob("*/metrics.json")):
        missing = missing_files(run_dir)
        if missing:
            if strict:
                raise FileNotFoundError(f"{run_dir.name} misses {missing}")
            logger.warning("skipping %s: missing %s", run_dir.name, missing)
            continue
        run = load_run(run_dir)
        m = run["metrics"]
        rows.append(
            {
                "run": run_dir.name,
                "model": m.get("model", ""),
                "seed": run["config"].get("seed"),
                "trainable_params": m.get("trainable_params"),
                "train_time_s": run["manifest"].get("duration_s"),
                "val_macro_f1": m.get("val", {}).get("macro_f1"),
                "test_macro_f1": m.get("test", {}).get("macro_f1"),
                "test_accuracy": m.get("test", {}).get("accuracy"),
            }
        )
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """A GitHub Markdown table (no extra dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, (int, np.integer)) and not isinstance(v, bool):
            return f"{v:,}"
        return "" if v is None else str(v)

    head = "| " + " | ".join(table.columns) + " |"
    rule = "|" + "|".join("---" for _ in table.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in table.itertuples(index=False)]
    return "\n".join([head, rule, *body]) + "\n"


def make_manifest(
    cfg: Config, duration_s: float | None = None, previous: Path | None = None
) -> dict[str, Any]:
    """git SHA, date, machine, library versions and duration of a run."""
    import sklearn
    import torch
    import transformers

    manifest = _read_json(previous) if previous is not None and previous.exists() else {}
    manifest.update(
        {
            "git_sha": _git("rev-parse", "HEAD"),
            "git_dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "machine": platform.platform(),
            "python": platform.python_version(),
            "device": cfg.device,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
            },
        }
    )
    if duration_s is not None:
        manifest["duration_s"] = round(float(duration_s), 1)
    return manifest


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _deep_merge(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = dict(old)
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _to_builtin(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=_to_builtin) + "\n")


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())
