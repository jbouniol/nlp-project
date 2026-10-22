"""Classification metrics written from the confusion matrix (S01 p13-14), checked against scikit-learn.

Convention: macro-F1 is the mean of F1_k over the classes that have at least one true example in the
evaluated set (support > 0), i.e. `f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")`.
It equals scikit-learn's default whenever every predicted class also appears in y_true (always the case
on the LEDGAR test split, where the 100 classes are present). It matters on validation: Books has no
validation clause, so validation macro-F1 is always a mean over the same 99 classes, whatever a model
predicts, which keeps validation scores comparable across models. Divisions by zero give 0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """C[i, j] = number of examples of true class i predicted as j."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    flat = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes)
    return flat.reshape(n_classes, n_classes)


def precision_recall(C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class precision C_kk / column_k and recall C_kk / row_k, 0 when the denominator is 0."""
    tp = np.diag(C).astype(float)
    col, row = C.sum(axis=0), C.sum(axis=1)
    precision = np.divide(tp, col, out=np.zeros_like(tp), where=col > 0)
    recall = np.divide(tp, row, out=np.zeros_like(tp), where=row > 0)
    return precision, recall


def f1_scores(C: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Per-class F1 = 2 C_kk / (row_k + col_k), macro-F1 (mean over classes with support), micro-F1."""
    tp = np.diag(C).astype(float)
    denom = C.sum(axis=0) + C.sum(axis=1)
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    present = C.sum(axis=1) > 0  # a class absent from y_true has no F1 to average (Books on validation)
    macro = float(f1[present].mean()) if present.any() else 0.0
    micro = float(tp.sum() / C.sum()) if C.sum() else 0.0
    return f1, macro, micro


def macro_f1(C: np.ndarray) -> float:
    """Macro-F1 from a confusion matrix (TP04 Ex 13)."""
    return f1_scores(C)[1]


def macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    """Macro-F1 from labels, over the classes present in y_true (see the module docstring)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    k = n_classes or int(max(y_true.max(), y_pred.max())) + 1
    return macro_f1(confusion(y_true, y_pred, k))


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of correct predictions (= micro-F1 for single-label classification)."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def scores(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None, prefix: str = ""
) -> dict[str, float]:
    """{"accuracy": .., "macro_f1": ..} (with optional key prefix such as "val_")."""
    return {
        f"{prefix}accuracy": accuracy(y_true, y_pred),
        f"{prefix}macro_f1": macro_f1_score(y_true, y_pred, n_classes),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax exp(z - max(z)) / sum(exp(z - max(z))) (S01 p5)."""
    z = np.asarray(logits, dtype=np.float32)
    if z.size == 0:
        return z
    exp_z = np.exp(z - z.max(axis=-1, keepdims=True))
    return exp_z / exp_z.sum(axis=-1, keepdims=True)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    """Trainer hook (TP06 Ex 9): `(logits, labels)` -> {"accuracy", "macro_f1"}."""
    logits, labels = eval_pred
    logits = logits[0] if isinstance(logits, tuple) else logits
    preds = np.asarray(logits).argmax(axis=-1)
    k = max(np.asarray(logits).shape[-1], int(np.max(labels)) + 1)
    return scores(labels, preds, k)


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, label_names: list[str], train_counts: np.ndarray | None = None
) -> pd.DataFrame:
    """One row per class: support, precision, recall, F1 (and train count if given), sorted by F1."""
    C = confusion(y_true, y_pred, len(label_names))
    precision, recall = precision_recall(C)
    f1, _, _ = f1_scores(C)
    report = pd.DataFrame(
        {
            "label": label_names,
            "support": C.sum(axis=1),
            "predicted": C.sum(axis=0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )
    if train_counts is not None:
        report.insert(1, "train_count", np.asarray(train_counts))
    return report.sort_values(["f1", "support"]).reset_index(drop=True)


def most_confused(C: np.ndarray, k: int = 10, label_names: list[str] | None = None) -> pd.DataFrame:
    """The k pairs of classes confused most often, in either direction (TP04 Ex 13).

    off = C + C^T - 2 diag(C): entry (i, j) counts i predicted as j plus j predicted as i.
    """
    off = C + C.T - 2 * np.diag(np.diag(C))
    i, j = np.triu_indices_from(off, k=1)
    order = np.argsort(-off[i, j], kind="stable")[:k]
    rows = [(i[o], j[o], int(C[i[o], j[o]]), int(C[j[o], i[o]]), int(off[i[o], j[o]])) for o in order]
    pairs = pd.DataFrame(rows, columns=["class_a", "class_b", "a_as_b", "b_as_a", "total"])
    if label_names is not None:
        pairs.insert(2, "label_a", [label_names[a] for a in pairs["class_a"]])
        pairs.insert(3, "label_b", [label_names[b] for b in pairs["class_b"]])
    return pairs[pairs["total"] > 0].reset_index(drop=True)


def f1_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    n_classes: int,
    n_boot: int = 0,
    seed: int = 0,
) -> pd.DataFrame:
    """Macro-F1 and accuracy inside each group of examples (with optional 95 % bootstrap CIs)."""
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    rows = []
    for g in pd.unique(groups):
        mask = groups == g
        yt, yp = y_true[mask], y_pred[mask]
        n_g = int(mask.sum())
        row: dict[str, object] = {"group": g, "n": n_g, **scores(yt, yp, n_classes)}
        if n_boot > 0 and n_g > 0:
            rng = np.random.default_rng(seed)
            f1s = np.empty(n_boot, dtype=float)
            accs = np.empty(n_boot, dtype=float)
            for b in range(n_boot):
                idx = rng.integers(0, n_g, n_g)
                f1s[b] = macro_f1_score(yt[idx], yp[idx], n_classes)
                accs[b] = accuracy(yt[idx], yp[idx])
            row["f1_low"], row["f1_high"] = map(float, np.percentile(f1s, [2.5, 97.5]))
            row["acc_low"], row["acc_high"] = map(float, np.percentile(accs, [2.5, 97.5]))
        rows.append(row)
    return pd.DataFrame(rows)


def reliability_bins(
    y_true: np.ndarray, y_pred: np.ndarray, prob_max: np.ndarray, n_bins: int = 10
) -> pd.DataFrame:
    """Equal-width confidence bins on [0, 1] with count, mean confidence, accuracy and |acc - conf|."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(float)
    conf = np.asarray(prob_max, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for b in range(n_bins):
        lo, hi = float(edges[b]), float(edges[b + 1])
        mask = (conf >= lo) & (conf <= hi) if b == 0 else (conf > lo) & (conf <= hi)
        n_b = int(mask.sum())
        c_b = float(conf[mask].mean()) if n_b else float("nan")
        a_b = float(correct[mask].mean()) if n_b else float("nan")
        rows.append(
            {
                "bin_low": lo,
                "bin_high": hi,
                "n": n_b,
                "conf": c_b,
                "acc": a_b,
                "gap": abs(a_b - c_b) if n_b else 0.0,
            }
        )
    return pd.DataFrame(rows)


def ece(y_true: np.ndarray, y_pred: np.ndarray, prob_max: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error: sum_b (n_b / N) |acc(b) - conf(b)| over `n_bins` equal-width bins."""
    bins = reliability_bins(y_true, y_pred, prob_max, n_bins)
    total = int(bins["n"].sum())
    if total == 0:
        return 0.0
    return float((bins["n"] * bins["gap"]).sum() / total)


def scale_confidence(prob_max: np.ndarray, temperature: float) -> np.ndarray:
    """Monotonic logit temperature scaling sigma(logit(p) / T) of top-1 confidence `prob_max`."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    p = np.clip(np.asarray(prob_max, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    logit = np.log(p / (1.0 - p)) / float(temperature)
    return (1.0 / (1.0 + np.exp(-logit))).astype(np.float64)


def fit_temperature(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prob_max: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Fit post-hoc temperature T > 0 on validation to minimize binary correctness NLL."""
    correct = (np.asarray(y_true) == np.asarray(y_pred)).astype(np.float64)
    candidates = np.linspace(0.5, 5.0, 451) if grid is None else np.asarray(grid, dtype=np.float64)
    best_t, best_nll = 1.0, float("inf")
    for t in candidates:
        p = np.clip(scale_confidence(prob_max, float(t)), 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(correct * np.log(p) + (1.0 - correct) * np.log(1.0 - p)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return round(best_t, 4)
