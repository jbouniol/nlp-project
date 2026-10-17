"""Paired comparison of two classifiers on the same test set (S06 p22-24, TP06 Part D)."""

from __future__ import annotations

import math

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
    observed = macro_f1_score(y_true, pred_a, k) - macro_f1_score(y_true, pred_b, k)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    diffs = np.empty(n_boot)
    for r in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[r] = macro_f1_score(y_true[idx], pred_a[idx], k) - macro_f1_score(y_true[idx], pred_b[idx], k)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(observed), float(low), float(high)


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
    a, b, c, d = contingency(y_true, pred_a, pred_b)
    chi2, p = mcnemar(b, c)
    observed, low, high = paired_bootstrap(y_true, pred_a, pred_b, n_boot, seed, n_classes)
    return {
        "n": len(y_true),
        "accuracy": [accuracy(y_true, pred_a), accuracy(y_true, pred_b)],
        "macro_f1": [macro_f1_score(y_true, pred_a, n_classes), macro_f1_score(y_true, pred_b, n_classes)],
        "contingency": {"both_correct": a, "only_a": b, "only_b": c, "both_wrong": d},
        "mcnemar": {"chi2": chi2, "p": p, "exact": b + c < 25, "significant": p < 0.05},
        "bootstrap": {
            "diff": observed,
            "low": low,
            "high": high,
            "n_boot": n_boot,
            "significant": not low <= 0 <= high,
        },
    }
