"""LEDGAR loading, per-class subsets and the descriptive statistics of the exploration step."""

from __future__ import annotations

import logging
import re
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


def length_grouped_order(
    lengths: np.ndarray, batch_size: int, rng: np.random.Generator, mega_batch_mult: int = 50
) -> np.ndarray:
    """Shuffle indices, cut into mega-batches, and sort each mega-batch by decreasing length."""
    lengths = np.asarray(lengths)
    order = rng.permutation(len(lengths))
    size = mega_batch_mult * batch_size
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    return np.concatenate([c[np.argsort(-lengths[c], kind="stable")] for c in chunks])


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
    rng = np.random.default_rng(seed)
    order = (
        length_grouped_order(clipped, batch_size, rng, mega_batch_mult)
        if group_by_length
        else rng.permutation(len(clipped))
    )
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


# --- text-derived slices for the bias audit across contracts, parties, jurisdictions and formatting ---

CONTRACT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credit / loan",
        re.compile(
            r"\b(?:Credit Agreement|Loan Agreement|Facility Agreement|Revolving Credit|Term Loan"
            r"|Lenders?|Borrowers?|Administrative Agent)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "employment / equity",
        re.compile(
            r"\b(?:Employment Agreement|Executive|Employee|Employer|Incentive Plan|Stock Option"
            r"|Award Agreement|Participant|Base Salary|Severance)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "indenture / securities",
        re.compile(
            r"\b(?:Indenture|Note(?:s)?|Debenture|Bondholder|Noteholder|Trustee|Warrant"
            r"|Registration Rights|Securities Act)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "M&A / purchase",
        re.compile(
            r"\b(?:Merger Agreement|Purchase Agreement|Stock Purchase|Asset Purchase|Acquisition"
            r"|Buyer|Seller|Purchaser|Target|Closing Date)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "lease / license",
        re.compile(
            r"\b(?:Lease|Landlord|Tenant|Lessor|Lessee|Premises|License Agreement"
            r"|Licensor|Licensee|Royalty)\b",
            re.IGNORECASE,
        ),
    ),
)

PARTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "borrower / lender",
        re.compile(
            r"\b(?:Borrower|Lender|Administrative Agent|Guarantor|Credit Parties|Loan Parties)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "employee / participant",
        re.compile(r"\b(?:Employee|Executive|Employer|Participant|Grantee|Optionee)\b", re.IGNORECASE),
    ),
    (
        "holder / trustee / investor",
        re.compile(
            r"\b(?:Stockholder|Shareholder|Holder|Noteholder|Trustee|Underwriter|Investor)\b", re.IGNORECASE
        ),
    ),
    (
        "buyer / seller",
        re.compile(r"\b(?:Buyer|Seller|Purchaser|Vendor|Acquiror|Merging)\b", re.IGNORECASE),
    ),
    (
        "landlord / tenant / licensor",
        re.compile(r"\b(?:Landlord|Tenant|Lessor|Lessee|Licensor|Licensee)\b", re.IGNORECASE),
    ),
)

JURISDICTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("New York", re.compile(r"\bNew York\b", re.IGNORECASE)),
    ("Delaware", re.compile(r"\bDelaware\b", re.IGNORECASE)),
    ("California", re.compile(r"\bCalifornia\b", re.IGNORECASE)),
    ("Texas / Nevada / Florida", re.compile(r"\b(?:Texas|Nevada|Florida)\b", re.IGNORECASE)),
    (
        "non-US / international",
        re.compile(
            r"\b(?:England|Wales|United Kingdom|London|Ontario|Canada|Cayman|Bermuda|Hong Kong"
            r"|Singapore|Switzerland|Germany|France|PRC|China|Japan|Ireland|Luxembourg)\b",
            re.IGNORECASE,
        ),
    ),
)


def _match_first(
    texts: Sequence[str], patterns: Sequence[tuple[str, re.Pattern[str]]], default: str
) -> np.ndarray:
    out = []
    for text in texts:
        label = default
        for name, pattern in patterns:
            if pattern.search(text):
                label = name
                break
        out.append(label)
    return np.array(out)


def contract_kind(texts: Sequence[str]) -> np.ndarray:
    """Inferred contract family of each clause from the agreement and party terms it names."""
    return _match_first(texts, CONTRACT_PATTERNS, "general / unspecified")


def party_role(texts: Sequence[str]) -> np.ndarray:
    """Primary party role named in each clause (borrower/lender, employee, holder/trustee, etc.)."""
    return _match_first(texts, PARTY_PATTERNS, "company / general")


def governing_jurisdiction(texts: Sequence[str]) -> np.ndarray:
    """Jurisdiction named in each clause (New York, Delaware, California, non-US, or none named)."""
    return _match_first(texts, JURISDICTION_PATTERNS, "none named")


def uppercase_formatting(texts: Sequence[str], threshold: float = 0.5) -> np.ndarray:
    """'all-caps (>50% upper)' for conspicuous boilerplate clauses, else 'standard case'."""
    out = []
    for text in texts:
        alpha = [c for c in text if c.isalpha()]
        share = sum(c.isupper() for c in alpha) / len(alpha) if alpha else 0.0
        out.append("all-caps (>50% upper)" if share > threshold else "standard case")
    return np.array(out)


def bias_slices(texts: Sequence[str]) -> dict[str, np.ndarray]:
    """Text-derived slices for the bias audit across contracts, parties, jurisdictions and formatting."""
    return {
        "contract kind": contract_kind(texts),
        "party role": party_role(texts),
        "jurisdiction named": governing_jurisdiction(texts),
        "all-caps formatting": uppercase_formatting(texts),
    }
