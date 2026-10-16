"""LEDGAR loading, per-class subsets and the descriptive statistics of the exploration step."""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from ledgar.config import Config
from ledgar.text import tokenize

logger = logging.getLogger(__name__)

SPLITS = ("train", "validation", "test")


def load_ledgar(cfg: Config) -> DatasetDict:
    """The three official splits at the pinned revision, each with an `index` column.

    `index` is the position of the clause in its official split: it travels with every subset and
    every saved prediction, so results can always be joined back to the raw text. Under FAST_RUN
    each split is reduced to a per-class subset (see `stratified_subset`).
    """
    ds = load_dataset(cfg.dataset_name, cfg.dataset_config, revision=cfg.dataset_revision)
    ds = DatasetDict({s: ds[s].add_column("index", np.arange(ds[s].num_rows)) for s in SPLITS})
    if cfg.n_train_per_class is not None:
        ds["train"] = stratified_subset(ds["train"], cfg.n_train_per_class, cfg.seed)
    if cfg.n_eval_per_class is not None:
        for s in ("validation", "test"):
            ds[s] = stratified_subset(ds[s], cfg.n_eval_per_class, cfg.seed)
    logger.info("LEDGAR loaded: %s", {s: ds[s].num_rows for s in SPLITS})
    return ds


def label_names(ds: DatasetDict) -> list[str]:
    """The 100 clause-type names, in label-id order."""
    return list(ds["train"].features["label"].names)


def stratified_subset(split: Dataset, per_class: int, seed: int = 0) -> Dataset:
    """Up to `per_class` random rows of every class, in their original order.

    A uniform random sample would lose most rare classes (Books: 23 of 60 000 training clauses)
    and break per-class F1 and the paired tests; this keeps all 100 classes.
    """
    labels = np.asarray(split["label"])
    rng = np.random.default_rng(seed)
    keep = [rng.permutation(np.flatnonzero(labels == k))[:per_class] for k in np.unique(labels)]
    return split.select(np.sort(np.concatenate(keep)))


def class_counts(ds: DatasetDict, n_classes: int) -> pd.DataFrame:
    """Number of clauses per class (rows) and split (columns)."""
    return pd.DataFrame({s: np.bincount(ds[s]["label"], minlength=n_classes) for s in ds})


def class_stats(counts: pd.DataFrame, thresholds: Sequence[int] = (100, 300)) -> dict[str, float]:
    """Imbalance summary of the training split, and how many classes are thin in each split."""
    train = counts["train"].to_numpy()
    stats: dict[str, float] = {
        "n_classes": len(train),
        "largest": int(train.max()),
        "smallest": int(train.min()),
        "imbalance_ratio": float(train.max() / train.min()),
        "mean_per_class": float(train.mean()),
        "classes_below_1pct": int(np.sum(train / train.sum() < 0.01)),
    }
    for t in thresholds:
        stats[f"train_classes_below_{t}"] = int(np.sum(train < t))
    for s in counts.columns:
        stats[f"{s}_classes_below_10"] = int(np.sum(counts[s] < 10))
        stats[f"{s}_classes_missing"] = int(np.sum(counts[s] == 0))
    return stats


def frequency_band(train_counts: np.ndarray, edges: tuple[int, int] = (300, 1000)) -> np.ndarray:
    """'tail' (< edges[0] training clauses), 'middle', or 'head' (>= edges[1]) for each class."""
    counts = np.asarray(train_counts)
    return np.where(counts < edges[0], "tail", np.where(counts >= edges[1], "head", "middle"))


def word_lengths(texts: Sequence[str]) -> np.ndarray:
    """Number of words of each text with the regex tokenizer of TP04."""
    return np.array([len(tokenize(t)) for t in texts])


def wordpiece_lengths(texts: Sequence[str], tokenizer, batch_size: int = 1000) -> np.ndarray:
    """Number of model tokens of each text, [CLS] and [SEP] included, **without** truncation (TP06 Ex 4)."""
    lengths = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(list(texts[start : start + batch_size]), truncation=False, verbose=False)
        lengths.extend(len(ids) for ids in enc["input_ids"])
    return np.array(lengths)


def length_stats(lengths: np.ndarray, thresholds: Sequence[int] = (128, 256, 512)) -> dict[str, float]:
    """Median, p90 / p95 / p99, max, and the share of texts longer than each threshold."""
    lengths = np.asarray(lengths)
    stats = {"median": float(np.median(lengths))}
    stats.update({f"p{q}": float(np.percentile(lengths, q)) for q in (90, 95, 99)})
    stats["max"] = int(lengths.max())
    stats.update({f"share_over_{t}": float(np.mean(lengths > t)) for t in thresholds})
    return stats


def padded_tokens(
    lengths: np.ndarray,
    max_length: int,
    batch_size: int = 32,
    seed: int = 0,
    group_by_length: bool = False,
    mega_batch_mult: int = 50,
) -> int:
    """Tokens the model processes in one epoch when each batch is padded to its longest (truncated) text.

    Random batches pay for the longest clause of each batch. `group_by_length` mimics the sampler of
    `TrainingArguments(train_sampling_strategy="group_by_length")`: shuffle, cut into mega-batches of
    `mega_batch_mult * batch_size` clauses, sort each mega-batch by length, then cut into batches.
    """
    clipped = np.minimum(np.asarray(lengths), max_length)
    order = np.random.default_rng(seed).permutation(len(clipped))
    if group_by_length:
        size = mega_batch_mult * batch_size
        chunks = [order[i : i + size] for i in range(0, len(order), size)]
        order = np.concatenate([c[np.argsort(-clipped[c], kind="stable")] for c in chunks])
    batches = [clipped[order[i : i + batch_size]] for i in range(0, len(order), batch_size)]
    return int(sum(len(b) * b.max() for b in batches))


def normalise(text: str) -> str:
    """Lower case, single spaces: the key used to find exact duplicates."""
    return " ".join(text.split()).lower()


def exact_duplicates(ds: DatasetDict) -> pd.DataFrame:
    """Exact duplicate clauses (after whitespace and case normalisation) inside and across splits."""
    keys = {s: pd.Series([normalise(t) for t in ds[s]["text"]]) for s in ds}
    rows = [{"pair": f"{s} (within)", "duplicates": int(keys[s].duplicated().sum())} for s in ds]
    names = list(ds)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            shared = keys[b].isin(set(keys[a]))
            rows.append({"pair": f"{a} / {b}", "duplicates": int(shared.sum())})
    return pd.DataFrame(rows)


def seen_in_train(ds: DatasetDict, split: str) -> np.ndarray:
    """True for each clause of `split` whose normalised text also appears in the training split."""
    train_keys = {normalise(t) for t in ds["train"]["text"]}
    return np.array([normalise(t) in train_keys for t in ds[split]["text"]])


def label_conflicts(ds: DatasetDict, label_names: list[str]) -> pd.DataFrame:
    """Texts that appear with more than one label anywhere in the dataset (ambiguous or noisy labels)."""
    rows = [
        {"text": normalise(t), "split": s, "label": label_names[y]}
        for s in ds
        for t, y in zip(ds[s]["text"], ds[s]["label"], strict=True)
    ]
    df = pd.DataFrame(rows)
    n_labels = df.groupby("text")["label"].transform("nunique")
    conflicts = (
        df[n_labels > 1]
        .groupby("text", sort=False)
        .agg(
            labels=("label", lambda x: sorted(set(x))),
            splits=("split", lambda x: sorted(set(x))),
            copies=("label", "size"),
        )
    )
    return conflicts.reset_index().assign(text=lambda d: d["text"].str.slice(0, 90) + "...")


def lazy_baseline(train_labels: np.ndarray, n: int) -> np.ndarray:
    """Predictions of the lazy classifier: always the most frequent training class (TP01 Ex 11)."""
    return np.full(n, np.bincount(np.asarray(train_labels)).argmax())


def class_weights(
    train_counts: np.ndarray, scheme: str = "sqrt", clip: tuple[float, float] = (0.5, 5.0)
) -> np.ndarray:
    """Candidate loss weights per class.

    "balanced": N / (K * count), scikit-learn's class_weight="balanced" (Books, 23 clauses: about 26).
    "sqrt": 1 / sqrt(count) normalised to mean 1, then clipped to `clip`: a softer version.
    """
    counts = np.asarray(train_counts, dtype=float)
    if scheme == "balanced":
        return counts.sum() / (len(counts) * counts)
    if scheme == "sqrt":
        w = 1 / np.sqrt(counts)
        return np.clip(w / w.mean(), *clip)
    raise ValueError(f"unknown scheme {scheme!r}")
