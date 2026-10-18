"""One frozen Config for every notebook, script and run (the constants block of the TPs)."""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

# What FAST_RUN=1 changes: a per-class subset, one epoch, short sequences, few resamples.
FAST_OVERRIDES: dict[str, Any] = {
    "n_train_per_class": 30,
    "n_eval_per_class": 5,
    "epochs": 1,
    "lstm_epochs": 1,
    "max_length": 128,
    "n_boot": 200,
    "out_dir": ".fast/results",  # never overwrite the committed results and figures
    "reports_dir": ".fast/reports",
    "checkpoint_dir": ".fast/checkpoints",
    "models_dir": ".fast/models",
}


@dataclass(frozen=True)
class Config:
    """Hyperparameters and paths. Defaults are the starting values of the design plan.

    Build it with `Config.from_env()` (notebooks) or `Config.from_args()` (scripts), never by
    editing a value in place: the same object is saved to `results/<run>/config.json`.
    """

    # Reproducibility
    seed: int = 0
    fast_run: bool = False
    device: str = "auto"  # "auto" -> cuda, then mps, then cpu

    # Data: the official LexGLUE splits, pinned to one revision of the dataset repository
    dataset_name: str = "coastalcph/lex_glue"
    dataset_config: str = "ledgar"
    dataset_revision: str = "c23fdff1a6bf74e0e1a71cb86f1e781d37da888c"  # Hub commit of 2024-01-04
    n_classes: int = 100
    n_train_per_class: int | None = None  # None = the full split; set by FAST_RUN
    n_eval_per_class: int | None = None

    # m1: TF-IDF + logistic regression
    tfidf_ngram_max: int = 2
    tfidf_min_df: int = 2
    logreg_c: float = 30.0  # chosen on validation (02_baseline_tfidf, docs/DECISIONS.md)
    class_weight: str | None = None
    logreg_solver: str = "lbfgs"  # saga is faster without weights but does not converge with balanced ones
    logreg_max_iter: int = 1000

    # m3: LSTM
    min_freq: int = 2
    max_words: int = 476  # p99 of training clause length in words (01_explore_ledgar)
    # m3 defaults = the configuration chosen on validation (scripts/search_lstm.py, docs/DECISIONS.md)
    emb_dim: int = 50  # GloVe 50-d
    glove: bool = True  # initialise the embedding from GloVe (S04 p15); `make glove` downloads it
    lstm_class_weight: str | None = "sqrt"  # None, "balanced" or "sqrt" (data.class_weights)
    hidden: int = 256
    bidirectional: bool = True
    dropout: float = 0.3
    lstm_lr: float = 2e-3
    lstm_batch_size: int = 64
    lstm_epochs: int = 15  # every 8-epoch run of the search peaked at its last epoch
    glove_path: str = "data/glove-wiki-gigaword-50.gz"

    # m4: DistilBERT, full fine-tuning and LoRA
    model_name: str = "distilbert-base-uncased"
    max_length: int = 512  # 98.2 % of training clauses whole (01_explore_ledgar, docs/DECISIONS.md)
    group_by_length: bool = True  # Trainer sampler: batches of similar lengths, little padding
    # Set per device by `device_profile` (from_env) unless given explicitly; batch_size * grad_accum = 32
    precision: str = "fp32"  # "fp32", "fp16" or "bf16" (mixed precision, CUDA)
    batch_size: int = 32
    grad_accum: int = 1
    eval_batch_size: int = 64
    epochs: int = 3
    lr_full: float = 3e-5
    lr_lora: float = 5e-4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    lora_r: int = 8
    lora_alpha: int = 16
    lora_targets: tuple[str, ...] = ("q_lin", "v_lin")
    lora_dropout: float = 0.05

    # Evaluation
    n_boot: int = 1000

    # Paths (relative to the repository root)
    data_dir: str = "data"
    out_dir: str = "results"
    reports_dir: str = "reports"
    checkpoint_dir: str = "checkpoints"  # Trainer checkpoints, not versioned (on Colab: a Drive folder)
    models_dir: str = "models"  # full fine-tuned weights, not versioned (on Colab: a Drive folder)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides: Any) -> Config:
        """Defaults, then the device profile, then FAST_RUN values if FAST_RUN=1, then SEED / DEVICE,
        then overrides: an explicit precision or batch size wins over the device profile."""
        values: dict[str, Any] = {"fast_run": os.environ.get("FAST_RUN", "0") == "1"}
        if "SEED" in os.environ:
            values["seed"] = int(os.environ["SEED"])
        if "DEVICE" in os.environ:
            values["device"] = os.environ["DEVICE"]
        values.update(overrides)
        if values.get("fast_run"):
            values = {**FAST_OVERRIDES, **values}
        device = resolve_device(values.get("device", "auto"))
        return cls(**{**device_profile(device), **values, "device": device})

    @classmethod
    def from_args(cls, argv: list[str] | None = None) -> tuple[Config, argparse.Namespace]:
        """One `--field-name` option per field (e.g. `--lr-full 2e-5 --seed 1`), plus `--model`, `--name`."""
        parser = argparse.ArgumentParser(description="Train or evaluate one LEDGAR classifier run.")
        parser.add_argument("--model", choices=["lazy", "tfidf", "lstm", "bert", "lora"], default=None)
        parser.add_argument("--name", default=None, help="run directory name under out_dir")
        parser.add_argument("--search", action="store_true", help="validation only, under out_dir/search/")
        for f in fields(cls):
            if f.name == "extra":
                continue
            opt = "--" + f.name.replace("_", "-")
            default = f.default
            if isinstance(default, bool):
                parser.add_argument(opt, action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS)
            elif isinstance(default, tuple):
                parser.add_argument(opt, nargs="+", default=argparse.SUPPRESS)
            else:
                parser.add_argument(opt, type=_parse_scalar, default=argparse.SUPPRESS)
        args = parser.parse_args(argv)
        passed = vars(args)
        overrides = {
            f.name: tuple(passed[f.name]) if isinstance(passed[f.name], list) else passed[f.name]
            for f in fields(cls)
            if f.name in passed
        }
        return cls.from_env(**overrides), args

    def to_dict(self) -> dict[str, Any]:
        """JSON-friendly view (tuples become lists)."""
        return {k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(self).items()}

    def path(self, *parts: str) -> Path:
        """Absolute path under the repository root, e.g. `cfg.path(cfg.out_dir, "tfidf")`."""
        return ROOT.joinpath(*parts)

    def __str__(self) -> str:
        shown = ["fast_run", "seed", "device", "precision", "batch_size", "grad_accum", "model_name"]
        shown += ["max_length", "max_words", "epochs"]
        return " | ".join(f"{k}={getattr(self, k)}" for k in shown)


def _parse_scalar(text: str) -> Any:
    """Parse a CLI value: int, then float, then None, else the string itself."""
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return None if text.lower() == "none" else text


def resolve_device(device: str = "auto") -> str:
    """Return "cuda", "mps" or "cpu"; "auto" picks the first available in that order."""
    if device != "auto":
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def device_profile(device: str) -> dict[str, Any]:
    """Precision and batch sizes of the m4 runs on `device`, with batch_size * grad_accum = 32 everywhere.

    The m4 learning rates are searched at an effective batch of 32: keeping it fixed keeps the runs
    comparable whatever the GPU. CUDA: bf16 (with TF32, see `train_bert.make_args`) where the GPU has
    native bf16 (A100, L4, H100), fp16 otherwise (T4, V100); 32 x 1 from 14 GB on (a T4 has 16 GB),
    16 x 2 below. MPS and CPU: fp32, 8 x 4. Evaluation keeps no activations for the backward pass, so its
    batch only follows memory. The thresholds are not measured.
    """
    if device == "cuda":
        import torch

        precision = "bf16" if torch.cuda.is_bf16_supported(including_emulation=False) else "fp16"
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        if memory_gb >= 40:
            return {"precision": precision, "batch_size": 32, "grad_accum": 1, "eval_batch_size": 256}
        if memory_gb >= 14:
            return {"precision": precision, "batch_size": 32, "grad_accum": 1, "eval_batch_size": 64}
        return {"precision": precision, "batch_size": 16, "grad_accum": 2, "eval_batch_size": 64}
    return {"precision": "fp32", "batch_size": 8, "grad_accum": 4, "eval_batch_size": 32}


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch (CPU, CUDA, MPS). Call it at the top and before each model is built."""
    import torch

    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (the global RNG the TPs rely on)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("seed set to %d", seed)
