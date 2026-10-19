"""m1: TF-IDF on bags of words + logistic regression, the project baseline (S01 p21)."""

from __future__ import annotations

import itertools
import logging
import time
import warnings
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ledgar import metrics
from ledgar.config import Config

logger = logging.getLogger(__name__)


def make_vectorizer(cfg: Config, ngram_max: int | None = None) -> TfidfVectorizer:
    """Lower-cased words, n-grams up to `ngram_max`, seen in >= min_df training clauses, sublinear tf."""
    return TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, ngram_max or cfg.tfidf_ngram_max),
        min_df=cfg.tfidf_min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )


def make_logreg(
    cfg: Config, c: float | None = None, class_weight: str | None = "config"
) -> LogisticRegression:
    """Multinomial logistic regression = one softmax over the 100 types (S01 p5), L2 penalty 1/C."""
    return LogisticRegression(
        C=cfg.logreg_c if c is None else c,
        class_weight=cfg.class_weight if class_weight == "config" else class_weight,
        solver=cfg.logreg_solver,
        max_iter=cfg.logreg_max_iter,
        random_state=cfg.seed,
    )


def make_pipeline(cfg: Config) -> Pipeline:
    """The baseline as one object: text in, label out. The vectorizer is fitted on the training texts only."""
    return Pipeline([("tfidf", make_vectorizer(cfg)), ("logreg", make_logreg(cfg))])


def count_parameters(model: Pipeline | LogisticRegression) -> int:
    """Weights of the logistic regression: one row of V weights and one bias per class."""
    logreg = model.named_steps["logreg"] if isinstance(model, Pipeline) else model
    return int(logreg.coef_.size + logreg.intercept_.size)


def fit_timed(model: Any, X: Any, y: np.ndarray) -> tuple[Any, float, bool]:
    """Fit and return (model, seconds, converged)."""
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return model, time.perf_counter() - start, converged


def grid_search(
    train_texts: Sequence[str],
    train_labels: np.ndarray,
    val_texts: Sequence[str],
    val_labels: np.ndarray,
    cfg: Config,
    grid: dict[str, Sequence[Any]],
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Fit every combination of `grid` on train and score it on validation, best macro-F1 first.

    `grid` has the keys "ngram_max", "c" and "class_weight". The vectorizer does not depend on C or on
    the class weights, so it is fitted once per n-gram setting, on the training texts only. `n_jobs`
    fits run in parallel: an lbfgs fit on bigrams holds about 3 GB, so keep it small on a laptop.
    """
    rows = []
    for ngram_max in grid["ngram_max"]:
        vectorizer = make_vectorizer(cfg, ngram_max)
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        rows += Parallel(n_jobs=n_jobs)(
            delayed(_score_one)(cfg, c, class_weight, X_train, train_labels, X_val, val_labels)
            for c, class_weight in itertools.product(grid["c"], grid["class_weight"])
        )
        for row in rows[-len(grid["c"]) * len(grid["class_weight"]) :]:
            row["ngram_max"] = ngram_max
            logger.info("tfidf grid %s", row)
    columns = [
        "ngram_max",
        "c",
        "class_weight",
        "vocabulary",
        "val_accuracy",
        "val_macro_f1",
        "fit_s",
        "converged",
    ]
    table = pd.DataFrame(rows)[columns]
    return table.sort_values("val_macro_f1", ascending=False).reset_index(drop=True)


def _score_one(
    cfg: Config,
    c: float,
    class_weight: str | None,
    X_train: Any,
    y_train: np.ndarray,
    X_val: Any,
    y_val: np.ndarray,
) -> dict[str, Any]:
    logreg, seconds, converged = fit_timed(make_logreg(cfg, c, class_weight), X_train, y_train)
    pred = logreg.predict(X_val)
    return {
        "c": c,
        "class_weight": class_weight or "none",
        "vocabulary": X_train.shape[1],
        **metrics.scores(y_val, pred, cfg.n_classes, prefix="val_"),
        "fit_s": round(seconds, 1),
        "converged": converged,
    }
