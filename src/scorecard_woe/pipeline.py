"""German Credit scorecard pipeline: split, WOE encoding, C tuning, scorecard fit and evaluation."""

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_core import RANDOM_STATE
from credit_core.datasets import load_german
from credit_core.metrics import discrimination
from credit_core.models import make_encoder, tune_scorecard_C
from credit_core.scorecard import Scorecard

NUMERIC = ["Age", "Credit amount", "Duration"]
CATEGORICAL = ["Job", "Housing", "Saving accounts", "Checking account", "Purpose"]
# Sex is protected. It is kept out of the model and used only in the fairness audit (CR03).
PROTECTED = ["Sex"]
FEATURES = NUMERIC + CATEGORICAL


@dataclass
class GermanSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_split(test_size: float = 0.3) -> GermanSplit:
    X, y = load_german()
    X = X[FEATURES + PROTECTED]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    return GermanSplit(X_tr, X_te, y_tr, y_te)


def fit_scorecard(split: GermanSplit, C: float | None = None) -> tuple[Scorecard, float]:
    X_tr = split.X_train[FEATURES]
    if C is None:
        C = tune_scorecard_C(NUMERIC, CATEGORICAL, X_tr, split.y_train)
    enc = make_encoder(NUMERIC, CATEGORICAL, n_bins=5, min_count=20, iv_min=0.02, iv_max=0.5)
    sc = Scorecard(enc, C=C).fit(X_tr, split.y_train)
    return sc, C


def evaluate(sc: Scorecard, split: GermanSplit) -> dict:
    p_test = sc.predict_good(split.X_test[FEATURES])
    p_train = sc.predict_good(split.X_train[FEATURES])
    return {
        "train": discrimination(split.y_train, p_train),
        "test": discrimination(split.y_test, p_test),
    }
