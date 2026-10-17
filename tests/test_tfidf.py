"""m1 baseline on a toy corpus: fitted on train only, parameter count, grid on validation."""

import numpy as np
import pandas as pd
import pytest

from ledgar import train_tfidf
from ledgar.config import ROOT, Config

TRAIN = [
    "the rent is due monthly",
    "rent payment is due on the first day",
    "this agreement is governed by the laws of delaware",
    "governed by the laws of new york",
    "either party may terminate this agreement upon notice",
    "the company may terminate upon written notice",
]
Y_TRAIN = np.array([0, 0, 1, 1, 2, 2])
VAL = ["monthly rent payment", "laws of delaware govern", "terminate with notice", "arbitration clause"]
Y_VAL = np.array([0, 1, 2, 2])


def toy_config(**kw):
    return Config.from_env(device="cpu", n_classes=3, tfidf_min_df=1, **kw)


def test_pipeline_fits_on_train_only():
    model = train_tfidf.make_pipeline(toy_config()).fit(TRAIN, Y_TRAIN)
    vocabulary = model.named_steps["tfidf"].vocabulary_
    assert "arbitration" not in vocabulary, "a word seen only in validation must not be in the vocabulary"
    assert (model.predict(TRAIN) == Y_TRAIN).all()
    assert train_tfidf.count_parameters(model) == 3 * len(vocabulary) + 3, "V weights and one bias per class"


def test_grid_search_sorted_by_validation_macro_f1():
    grid = {"ngram_max": [1, 2], "c": [0.1, 10.0], "class_weight": [None, "balanced"]}
    table = train_tfidf.grid_search(TRAIN, Y_TRAIN, VAL, Y_VAL, toy_config(), grid, n_jobs=2)
    assert len(table) == 8 and table["val_macro_f1"].is_monotonic_decreasing
    assert set(table["class_weight"]) == {"none", "balanced"}
    vocabulary = table.groupby("ngram_max")["vocabulary"].first()
    assert vocabulary[2] > vocabulary[1], "bigrams add features"


def test_config_holds_the_configuration_chosen_on_validation():
    """`make m1` must reproduce the model of 02_baseline_tfidf: Config defaults = best row of its grid."""
    path = ROOT / "results" / "tfidf" / "grid_val.csv"
    if not path.exists():
        pytest.skip("02_baseline_tfidf has not been run yet")
    best = pd.read_csv(path).iloc[0]
    cfg = Config()
    weight = None if best["class_weight"] == "none" else best["class_weight"]
    assert (cfg.tfidf_ngram_max, cfg.logreg_c, cfg.class_weight) == (best["ngram_max"], best["c"], weight)
