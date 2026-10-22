"""Metrics against the course values (S01 p15, TP01, TP04 Ex 13, TP06 Ex 9) and against scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from ledgar import metrics
from ledgar.data import (
    bias_slices,
    class_weights,
    contract_kind,
    frequency_band,
    governing_jurisdiction,
    lazy_baseline,
    length_stats,
    padded_tokens,
    party_role,
    uppercase_formatting,
)

COURSE_C = np.array([[44, 4, 2], [2, 6, 2], [1, 1, 8]])


def labels_from_confusion(C):
    y_true = np.repeat(np.arange(len(C)), C.sum(axis=1))
    y_pred = np.concatenate([np.repeat(np.arange(len(C)), row) for row in C])
    return y_true, y_pred


def test_course_matrix():
    P, R = metrics.precision_recall(COURSE_C)
    F1, macro, micro = metrics.f1_scores(COURSE_C)
    assert np.allclose(P, [0.936, 0.545, 0.667], atol=1e-3)
    assert np.allclose(R, [0.88, 0.60, 0.80], atol=1e-3)
    assert np.allclose(F1, [0.907, 0.571, 0.727], atol=1e-3)
    assert round(macro, 3) == 0.735 and round(micro, 3) == 0.829


def test_confusion_roundtrip_and_sklearn():
    y_true, y_pred = labels_from_confusion(COURSE_C)
    C = metrics.confusion(y_true, y_pred, 3)
    assert (C == COURSE_C).all()
    P, R = metrics.precision_recall(C)
    assert np.allclose(P, precision_score(y_true, y_pred, average=None))
    assert np.allclose(R, recall_score(y_true, y_pred, average=None))
    assert metrics.macro_f1(C) == pytest.approx(f1_score(y_true, y_pred, average="macro"))
    assert metrics.f1_scores(C)[2] == pytest.approx(accuracy_score(y_true, y_pred))


def test_toy_macro_f1():
    assert metrics.macro_f1(np.array([[5, 1, 0], [2, 3, 1], [0, 0, 4]])) == pytest.approx(0.7527, abs=1e-4)


def test_compute_metrics_course_case():
    logits = np.array([[2.0, 0, 0], [0, 1, 0], [0, 3, 0], [0, 0, 1]])
    m = metrics.compute_metrics((logits, np.array([0, 0, 1, 2])))
    assert m["accuracy"] == pytest.approx(0.75) and m["macro_f1"] == pytest.approx(7 / 9)


def test_scores_and_softmax():
    y_true, y_pred = np.array([0, 1, 1]), np.array([0, 1, 0])
    s = metrics.scores(y_true, y_pred, n_classes=2, prefix="val_")
    assert s["val_accuracy"] == pytest.approx(2 / 3) and s["val_macro_f1"] == pytest.approx(2 / 3)
    probs = metrics.softmax(np.array([[0.0, 0.0], [1000.0, 999.0]]))
    assert np.allclose(probs.sum(axis=1), 1.0) and probs[0, 0] == pytest.approx(0.5)


def test_lazy_classifier():
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    y_pred = lazy_baseline(y_true, len(y_true))
    assert (y_pred == 0).all()
    assert metrics.accuracy(y_true, y_pred) == pytest.approx(0.90)
    assert metrics.macro_f1_score(y_true, y_pred, 3) == pytest.approx(0.316, abs=1e-3)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_labels_equal_sklearn(seed):
    rng = np.random.default_rng(seed)
    y_true, y_pred = rng.integers(0, 100, 2000), rng.integers(0, 100, 2000)
    y_pred[:20] = 99  # some classes predicted but rare in y_true
    expected = f1_score(y_true, y_pred, average="macro")
    assert metrics.macro_f1_score(y_true, y_pred, 100) == pytest.approx(expected)


def test_macro_f1_over_classes_with_support():
    y_true, y_pred = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    assert metrics.macro_f1_score(y_true, y_pred, n_classes=5) == pytest.approx(
        f1_score(y_true, y_pred, average="macro")
    ), "classes never seen nor predicted do not count"
    y_pred = np.array([0, 4, 1, 1])  # class 4 is predicted but has no true example (Books on validation)
    expected = f1_score(y_true, y_pred, labels=np.unique(y_true), average="macro")
    assert metrics.macro_f1_score(y_true, y_pred, n_classes=5) == pytest.approx(expected)


def test_most_confused():
    C = np.array([[10, 3, 0], [4, 8, 1], [0, 2, 9]])
    pairs = metrics.most_confused(C, k=2, label_names=["a", "b", "c"])
    assert (pairs.loc[0, "label_a"], pairs.loc[0, "label_b"], pairs.loc[0, "total"]) == ("a", "b", 7)
    assert pairs.loc[1, "total"] == 3


def test_per_class_report_sorted_by_f1():
    y_true, y_pred = labels_from_confusion(COURSE_C)
    report = metrics.per_class_report(y_true, y_pred, ["termination", "payment", "confidentiality"])
    assert list(report["label"]) == ["payment", "confidentiality", "termination"]
    assert report["support"].sum() == COURSE_C.sum()


def test_length_stats_and_bands():
    stats = length_stats(np.arange(1, 1001), thresholds=(500,))
    assert stats["median"] == 500.5 and stats["max"] == 1000 and stats["share_over_500"] == 0.5
    assert list(frequency_band(np.array([23, 600, 3167]))) == ["tail", "middle", "head"]


def test_padded_tokens():
    lengths = np.array([10, 100, 10, 100])
    assert padded_tokens(lengths, max_length=50, batch_size=2, group_by_length=True) == 2 * 10 + 2 * 50
    assert padded_tokens(lengths, max_length=512, batch_size=4) == 4 * 100


def test_class_weights():
    counts = np.array([23, 600, 3167])
    balanced = class_weights(counts, "balanced")
    assert balanced[0] == pytest.approx(counts.sum() / (3 * 23)), "scikit-learn's balanced formula"
    soft = class_weights(counts, "sqrt")
    assert soft.min() >= 0.5 and soft.max() <= 5 and soft[0] > soft[1] >= soft[2]


def test_bias_audit_text_slices():
    clauses = [
        "The Borrower shall repay the Lenders under this Credit Agreement in New York.",
        "The Employee shall receive a Base Salary under this Employment Agreement in California.",
        "THE LANDLORD HEREBY LEASES THE PREMISES TO THE TENANT IN DELAWARE.",
        "Each party agrees to keep confidential all information disclosed hereunder.",
    ]
    assert contract_kind(clauses).tolist() == [
        "credit / loan",
        "employment / equity",
        "lease / license",
        "general / unspecified",
    ]
    assert party_role(clauses).tolist() == [
        "borrower / lender",
        "employee / participant",
        "landlord / tenant / licensor",
        "company / general",
    ]
    assert governing_jurisdiction(clauses).tolist() == [
        "New York",
        "California",
        "Delaware",
        "none named",
    ]
    assert uppercase_formatting(clauses).tolist() == [
        "standard case",
        "standard case",
        "all-caps (>50% upper)",
        "standard case",
    ]
    slices = bias_slices(clauses)
    assert set(slices) == {"contract kind", "party role", "jurisdiction named", "all-caps formatting"}


def test_f1_by_group_bootstrap_ci():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, 120)
    y_pred = np.where(rng.random(120) < 0.8, y_true, rng.integers(0, 3, 120))
    groups = np.array(["a"] * 60 + ["b"] * 60)
    df = metrics.f1_by_group(y_true, y_pred, groups, n_classes=3, n_boot=100, seed=0)
    assert {"f1_low", "f1_high", "acc_low", "acc_high"} <= set(df.columns)
    assert (df["f1_low"] <= df["macro_f1"]).all() and (df["macro_f1"] <= df["f1_high"]).all()


def test_ece_and_post_hoc_temperature_scaling():
    rng = np.random.default_rng(7)
    y_true = rng.integers(0, 4, 400)
    y_pred = np.where(rng.random(400) < 0.75, y_true, (y_true + 1) % 4)
    prob_max = np.full(400, 0.98)  # severely overconfident
    raw_ece = metrics.ece(y_true, y_pred, prob_max)
    assert raw_ece == pytest.approx(0.98 - metrics.accuracy(y_true, y_pred), abs=1e-6)
    temp = metrics.fit_temperature(y_true, y_pred, prob_max)
    assert temp > 1.5, "overconfident predictions require T > 1"
    cal_prob = metrics.scale_confidence(prob_max, temp)
    assert metrics.ece(y_true, y_pred, cal_prob) < 0.05
    with pytest.raises(ValueError):
        metrics.scale_confidence(prob_max, 0.0)
