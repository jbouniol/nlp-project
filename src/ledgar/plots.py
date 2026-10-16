"""Figures in the style of the TPs; each function returns the matplotlib Figure (and saves it if asked)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"train": "#0f766e", "validation": "#d97706", "test": "#6b7280"}


def _save(fig: plt.Figure, path: str | Path | None) -> plt.Figure:
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
    return fig


def plot_class_distribution(
    counts: pd.DataFrame, label_names: list[str], path: str | Path | None = None
) -> plt.Figure:
    """Clauses per class, sorted by training count, log scale, one series per split."""
    order = np.argsort(-counts["train"].to_numpy())
    fig, ax = plt.subplots(figsize=(14, 4.5))
    x = np.arange(len(order))
    for split in counts.columns:
        values = counts[split].to_numpy()[order]
        if split == "train":
            ax.bar(x, values, color=COLORS[split], label=split)
        else:
            ax.plot(x, values, "o", ms=3, color=COLORS.get(split), label=split)
    ax.set_yscale("log")
    ax.set_xticks(x[::3], [label_names[i] for i in order[::3]], rotation=90, fontsize=7)
    ax.set_ylabel("clauses (log scale)")
    ax.set_title("LEDGAR: clauses per type, sorted by training count")
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)


def plot_length_hist(
    lengths: np.ndarray,
    thresholds: Sequence[int] = (128, 256, 512),
    unit: str = "tokens",
    path: str | Path | None = None,
) -> plt.Figure:
    """Histogram of text lengths with a vertical line and the share above each threshold."""
    lengths = np.asarray(lengths)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.hist(np.clip(lengths, 0, np.percentile(lengths, 99.5)), bins=80, color=COLORS["train"])
    for t in thresholds:
        ax.axvline(t, color="#b91c1c", ls="--", lw=1)
        ax.text(t, ax.get_ylim()[1] * 0.9, f" {t}: {100 * np.mean(lengths > t):.1f} % longer", fontsize=8)
    ax.set_xlabel(f"{unit} per clause (clipped at p99.5)")
    ax.set_ylabel("clauses")
    ax.set_title(f"Training clauses: length in {unit}")
    fig.tight_layout()
    return _save(fig, path)


def plot_confusion(
    C: np.ndarray, labels: Sequence[str], title: str = "", path: str | Path | None = None
) -> plt.Figure:
    """Confusion (sub-)matrix, rows = true class, columns = predicted class, values annotated."""
    n = len(labels)
    fig, ax = plt.subplots(figsize=(0.45 * n + 3, 0.45 * n + 2.5))
    ax.imshow(C, cmap="Greens")
    ax.set_xticks(range(n), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(n), labels, fontsize=8)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    for i in range(n):
        for j in range(n):
            if C[i, j]:
                color = "white" if C[i, j] > C.max() / 2 else "black"
                ax.text(j, i, int(C[i, j]), ha="center", va="center", fontsize=7, color=color)
    ax.set_title(title)
    fig.tight_layout()
    return _save(fig, path)


def plot_loss_curves(
    histories: dict[str, Sequence[float]], ylabel: str = "training loss", path: str | Path | None = None
) -> plt.Figure:
    """One curve per run, e.g. {"full": [...], "LoRA r=8": [...]}."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for name, values in histories.items():
        ax.plot(values, label=name)
    ax.set_xlabel("step")
    ax.set_ylabel(ylabel)
    ax.legend()
    fig.tight_layout()
    return _save(fig, path)
