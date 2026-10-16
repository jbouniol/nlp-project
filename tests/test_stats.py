"""Paired tests against the course values (S06 p24, TP06 Ex 19-22)."""

import math

import numpy as np
import pytest
from scipy.stats import binomtest

from ledgar.stats import contingency, mcnemar, paired_bootstrap


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
