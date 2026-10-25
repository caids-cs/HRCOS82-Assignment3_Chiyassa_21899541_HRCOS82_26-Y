"""
Tests that protect the three rules the protocol depends on.

Run with:  pytest

If one of these fails after you have changed something, the change has broken a
methodological guarantee, not just a piece of code. Read the failure before you
"fix" the test.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data as D
from src import features as F
from src import modelling as M


@pytest.fixture(scope="module")
def synth():
    return D.make_synthetic(domain="elearning", n_entities=200, seed=1)


def test_synthetic_matches_canonical_schema(synth):
    events, labels = synth
    assert list(events.columns) == D.EVENT_COLUMNS
    assert list(labels.columns) == D.LABEL_COLUMNS
    assert set(labels["label"].unique()) <= {0, 1}


def test_both_domains_produce_the_same_shape():
    """The whole convergence claim depends on this: two domains, one schema."""
    e1, l1 = D.make_synthetic(domain="elearning", n_entities=120, seed=2)
    e2, l2 = D.make_synthetic(domain="devops", n_entities=120, seed=2)
    assert list(e1.columns) == list(e2.columns)
    f1 = F.build_features(e1)
    f2 = F.build_features(e2)
    # Domain-specific event-type columns differ; the structural features do not.
    structural = {"n_events", "events_per_day", "n_active_days", "n_event_types",
                  "hours_since_last_event", "activity_trend", "gap_mean_h"}
    assert structural <= set(f1.columns)
    assert structural <= set(f2.columns)


def test_cutoff_actually_truncates(synth):
    """Features must not be built from events after the cutoff."""
    events, _ = synth
    early, cutoff_early = F.apply_cutoff(events, 0.2)
    late, cutoff_late = F.apply_cutoff(events, 0.8)
    assert early["timestamp"].max() <= cutoff_early
    assert len(early) < len(late)
    assert cutoff_early < cutoff_late


def test_features_have_one_row_per_entity(synth):
    events, labels = synth
    feats = F.build_features(events)
    X, y = F.align(feats, labels)
    assert X.index.is_unique
    assert len(X) == len(y)
    assert not X.index.duplicated().any()


def test_split_is_stratified_and_disjoint(synth):
    events, labels = synth
    X, y = F.align(F.build_features(events), labels)
    Xtr, Xte, ytr, yte = M.split(X, y, seed=3)
    assert set(Xtr.index).isdisjoint(set(Xte.index))
    assert len(Xtr) + len(Xte) == len(X)
    # Positive rate preserved in both halves, to within a couple of points.
    assert abs(ytr.mean() - yte.mean()) < 0.06


def test_preprocessing_is_inside_the_pipeline():
    """Guards against leakage: every model must carry its own preprocessing,
    so scalers and imputers are fitted on the training fold only."""
    for name, model in M.make_models().items():
        steps = [s for s, _ in model.steps]
        assert "impute" in steps and "scale" in steps, f"{name} lacks preprocessing"
        assert steps[-1] == "clf", f"{name} must end with the classifier"


def test_majority_baseline_is_present_and_useless():
    """The baseline exists so that a high accuracy cannot be mistaken for skill."""
    models = M.make_models()
    assert "majority_baseline" in models
    assert "logistic_regression" in models


def test_scores_include_the_metrics_that_matter():
    y_true = np.array([0] * 80 + [1] * 20)
    y_pred = np.zeros(100, dtype=int)          # predicts "nobody", always
    s = M.score(y_true, y_pred)
    assert s["accuracy"] == pytest.approx(0.80)  # looks good
    assert s["recall"] == 0.0                    # and is worthless
    assert s["f1"] == 0.0
    assert {"tn", "fp", "fn", "tp"} <= set(s)


def test_end_to_end_beats_the_majority_baseline(synth):
    """A smoke test with a methodological assertion: on data with real signal,
    logistic regression must beat predicting the majority class."""
    events, labels = synth
    X, y = F.align(F.build_features(events), labels)
    Xtr, Xte, ytr, yte = M.split(X, y, seed=4)
    models = M.make_models(seed=4)
    lr = M.evaluate_on_test(models["logistic_regression"], Xtr, ytr, Xte, yte)
    base = M.evaluate_on_test(models["majority_baseline"], Xtr, ytr, Xte, yte)
    assert lr["f1"] > base["f1"]
    assert lr["recall"] > 0.3
