"""Paired comparison of two classifiers on the same test set (S06 p22-24, TP06 Part D)."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ledgar.metrics import accuracy, macro_f1_score


def contingency(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> tuple[int, int, int, int]:
    """(a, b, c, d) = (both correct, only A correct, only B correct, both wrong)."""
    ok_a = np.asarray(pred_a) == np.asarray(y_true)
    ok_b = np.asarray(pred_b) == np.asarray(y_true)
    return (
        int(np.sum(ok_a & ok_b)),
        int(np.sum(ok_a & ~ok_b)),
        int(np.sum(~ok_a & ok_b)),
        int(np.sum(~ok_a & ~ok_b)),
    )


def mcnemar(b: int, c: int, exact: bool | None = None) -> tuple[float, float]:
    """McNemar's test on the disagreements b and c; returns (chi2, p).

    chi2 = (|b - c| - 1)^2 / (b + c) (continuity correction), p = P(chi2_1 > chi2) = erfc(sqrt(chi2/2)).
    When b + c < 25 (or `exact=True`) p comes from the exact binomial test, b ~ Binomial(b + c, 1/2).
    """
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = max(abs(b - c) - 1, 0) ** 2 / n
    if exact is None:
        exact = n < 25
    if exact:
        tail = sum(math.comb(n, k) for k in range(min(b, c) + 1)) / 2**n
        return chi2, min(1.0, 2 * tail)
    return chi2, math.erfc(math.sqrt(chi2 / 2))


def bootstrap_interval(
    n: int, diff_fn: Callable[[np.ndarray], float], n_boot: int = 1000, seed: int = 0
) -> dict[str, Any]:
    """Paired bootstrap of a difference `diff_fn(indices)` over `n_boot` resamples of 0..n-1."""
    observed = float(diff_fn(np.arange(n)))
    rng = np.random.default_rng(seed)
    diffs = np.array([diff_fn(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=float)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {
        "diff": observed,
        "low": float(low),
        "high": float(high),
        "n_boot": n_boot,
        "significant": bool(not low <= 0 <= high),
    }


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    n_classes: int | None = None,
) -> tuple[float, float, float]:
    """Paired bootstrap of macro-F1(A) - macro-F1(B); returns (observed, low 2.5 %, high 97.5 %).

    The same resampled indices are used for both models at each round: 0 outside [low, high] means
    the difference is significant at 5 %.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    k = n_classes or int(max(y_true.max(), pred_a.max(), pred_b.max())) + 1
    res = bootstrap_interval(
        len(y_true),
        lambda idx: macro_f1_score(y_true[idx], pred_a[idx], k) - macro_f1_score(y_true[idx], pred_b[idx], k),
        n_boot,
        seed,
    )
    return res["diff"], res["low"], res["high"]


def compare(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
    n_classes: int = 100,
) -> dict:
    """The paired comparison of S06 p22-24 for two models on the same examples.

    Returns the scores of both models, the contingency table, McNemar's test on per-example correctness and
    the paired bootstrap of the macro-F1 difference (A - B), with the 5 % verdict of each test.
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a, b, c, d = contingency(y_true, pred_a, pred_b)
    chi2, p = mcnemar(b, c)
    bootstrap = bootstrap_interval(
        len(y_true),
        lambda idx: (
            macro_f1_score(y_true[idx], pred_a[idx], n_classes)
            - macro_f1_score(y_true[idx], pred_b[idx], n_classes)
        ),
        n_boot,
        seed,
    )
    return {
        "n": len(y_true),
        "accuracy": [accuracy(y_true, pred_a), accuracy(y_true, pred_b)],
        "macro_f1": [macro_f1_score(y_true, pred_a, n_classes), macro_f1_score(y_true, pred_b, n_classes)],
        "contingency": {"both_correct": a, "only_a": b, "only_b": c, "both_wrong": d},
        "mcnemar": {"chi2": chi2, "p": p, "exact": b + c < 25, "significant": p < 0.05},
        "bootstrap": bootstrap,
    }


def compare_seeds(
    y_true: np.ndarray,
    preds_a: Sequence[np.ndarray],
    preds_b: Sequence[np.ndarray],
    n_boot: int = 1000,
    seed: int = 0,
    n_classes: int = 100,
) -> dict[str, Any]:
    """Paired multi-seed comparison across S seeds (a 1-seed deterministic baseline is broadcast to S)."""
    y_true = np.asarray(y_true)
    list_a = [np.asarray(p) for p in preds_a]
    list_b = [np.asarray(p) for p in preds_b]
    if len(list_a) == 1 and len(list_b) > 1:
        list_a = list_a * len(list_b)
    elif len(list_b) == 1 and len(list_a) > 1:
        list_b = list_b * len(list_a)
    if len(list_a) != len(list_b) or not list_a:
        raise ValueError("preds_a and preds_b must have the same number of seeds (or 1 broadcast seed)")
    per_seed = []
    for pa, pb in zip(list_a, list_b, strict=True):
        a, b, c, d = contingency(y_true, pa, pb)
        chi2, p = mcnemar(b, c)
        per_seed.append(
            {
                "accuracy": [accuracy(y_true, pa), accuracy(y_true, pb)],
                "macro_f1": [macro_f1_score(y_true, pa, n_classes), macro_f1_score(y_true, pb, n_classes)],
                "contingency": {"both_correct": a, "only_a": b, "only_b": c, "both_wrong": d},
                "mcnemar": {"chi2": chi2, "p": p, "exact": b + c < 25, "significant": p < 0.05},
            }
        )
    f1_a = [s["macro_f1"][0] for s in per_seed]
    f1_b = [s["macro_f1"][1] for s in per_seed]
    acc_a = [s["accuracy"][0] for s in per_seed]
    acc_b = [s["accuracy"][1] for s in per_seed]
    bootstrap = bootstrap_interval(
        len(y_true),
        lambda idx: float(
            np.mean(
                [
                    macro_f1_score(y_true[idx], pa[idx], n_classes)
                    - macro_f1_score(y_true[idx], pb[idx], n_classes)
                    for pa, pb in zip(list_a, list_b, strict=True)
                ]
            )
        ),
        n_boot,
        seed,
    )
    return {
        "n": len(y_true),
        "n_seeds": len(list_a),
        "accuracy_mean": [float(np.mean(acc_a)), float(np.mean(acc_b))],
        "accuracy_std": [
            float(np.std(acc_a, ddof=1)) if len(acc_a) > 1 else 0.0,
            float(np.std(acc_b, ddof=1)) if len(acc_b) > 1 else 0.0,
        ],
        "macro_f1_mean": [float(np.mean(f1_a)), float(np.mean(f1_b))],
        "macro_f1_std": [
            float(np.std(f1_a, ddof=1)) if len(f1_a) > 1 else 0.0,
            float(np.std(f1_b, ddof=1)) if len(f1_b) > 1 else 0.0,
        ],
        "bootstrap": bootstrap,
        "per_seed": per_seed,
    }
