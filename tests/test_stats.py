"""Paired tests against the course values (S06 p24, TP06 Ex 19-22)."""

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from ledgar.stats import compare, compare_seeds, contingency, mcnemar, paired_bootstrap


def test_contingency_course_case():
    assert contingency([0, 1, 2, 3, 0], [0, 1, 0, 0, 0], [0, 2, 2, 0, 1]) == (1, 2, 1, 1)


@pytest.mark.parametrize(
    ("b", "c", "chi2", "p"),
    [(60, 40, 3.61, 0.0574), (25, 5, 12.03, 0.00052), (0, 0, 0.0, 1.0)],
)
def test_mcnemar_course_values(b, c, chi2, p):
    got_chi2, got_p = mcnemar(b, c)
    assert got_chi2 == pytest.approx(chi2, abs=5e-3) and got_p == pytest.approx(p, abs=5e-5)


def test_mcnemar_exact_below_25():
    chi2, p = mcnemar(15, 4)
    assert chi2 == pytest.approx((11 - 1) ** 2 / 19)
    assert p == pytest.approx(binomtest(4, 19, 0.5).pvalue), "b + c < 25: exact binomial test"
    assert mcnemar(15, 4, exact=False)[1] == pytest.approx(math.erfc(math.sqrt(chi2 / 2)))


def test_bootstrap_same_model_is_zero():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 500)
    pred = np.where(rng.random(500) < 0.7, y, rng.integers(0, 10, 500))
    assert paired_bootstrap(y, pred, pred, n_boot=100) == (0.0, 0.0, 0.0)


def test_bootstrap_detects_a_clear_gap():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 10, 2000)
    good = np.where(rng.random(2000) < 0.9, y, rng.integers(0, 10, 2000))
    bad = np.where(rng.random(2000) < 0.6, y, rng.integers(0, 10, 2000))
    observed, low, high = paired_bootstrap(y, good, bad, n_boot=200)
    assert low <= observed <= high and low > 0, "0 is outside the interval: significant"


def test_compare_bundles_the_paired_tests():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 5, 400)
    a = np.where(rng.random(400) < 0.9, y, rng.integers(0, 5, 400))
    b = np.where(rng.random(400) < 0.6, y, rng.integers(0, 5, 400))
    r = compare(y, a, b, n_boot=100, n_classes=5)
    t = r["contingency"]
    assert t["both_correct"] + t["only_a"] == round(r["accuracy"][0] * 400), "a + b = correct answers of A"
    assert r["mcnemar"]["significant"] and r["bootstrap"]["significant"]
    same = compare(y, a, a, n_boot=50, n_classes=5)
    assert not same["mcnemar"]["significant"] and same["bootstrap"]["diff"] == 0.0


def test_compare_seeds_multi_seed():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 4, 300)
    preds_a = [np.where(rng.random(300) < 0.85, y, rng.integers(0, 4, 300)) for _ in range(3)]
    pred_b = [np.where(rng.random(300) < 0.60, y, rng.integers(0, 4, 300))]
    res = compare_seeds(y, preds_a, pred_b, n_boot=100, n_classes=4)
    assert res["n_seeds"] == 3 and len(res["per_seed"]) == 3
    assert res["bootstrap"]["significant"] and res["macro_f1_mean"][0] > res["macro_f1_mean"][1]
