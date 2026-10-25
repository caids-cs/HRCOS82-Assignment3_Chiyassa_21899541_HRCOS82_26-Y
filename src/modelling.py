"""
Splitting, models and evaluation.

THREE RULES ENFORCED HERE
-------------------------
1. Split first. Nothing - no scaler, no imputer, no feature selection - is
   fitted before the data is divided. Every transformer lives inside a
   scikit-learn Pipeline, so it is fitted on the training fold only. This is
   not stylistic; fitting before the split inflates every number you report.

2. Baseline first. A majority-class classifier and a logistic regression are
   always run. If your sophisticated model cannot beat logistic regression,
   you have learned something real and you must report it.

3. Never trust accuracy on imbalanced data. If 18% of students drop out, a
   model that predicts "nobody drops out" is 82% accurate and useless. The
   evaluation below always reports recall, F1 and average precision alongside
   accuracy, and always prints the majority-class accuracy next to your own so
   the comparison is impossible to avoid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


# --------------------------------------------------------------------------
# 1. Splitting
# --------------------------------------------------------------------------

def split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, seed: int = 42):
    """Stratified train/test split. Call this BEFORE fitting anything at all.

    Stratification keeps the positive rate the same in both halves, which
    matters when the positive class is small.
    """
    return train_test_split(X, y, test_size=test_size, stratify=y,
                            random_state=seed, shuffle=True)


# --------------------------------------------------------------------------
# 2. Models
# --------------------------------------------------------------------------

def make_models(seed: int = 42, balanced: bool = True) -> dict[str, Pipeline]:
    """The standard model set. Every one is a Pipeline, so preprocessing is
    fitted inside the cross-validation fold and never on the test set.

    `balanced=True` sets class_weight='balanced', which tells the model that
    the rare class matters as much as the common one. Report whether you used
    it - it changes the results substantially.
    """
    cw = "balanced" if balanced else None
    pre = [("impute", SimpleImputer(strategy="median")),
           ("scale", StandardScaler())]

    return {
        # Baseline 0: predicts the majority class, always. The number to beat.
        "majority_baseline": Pipeline(pre + [
            ("clf", DummyClassifier(strategy="most_frequent"))]),

        # Baseline 1: the real baseline. Interpretable, fast, often excellent.
        "logistic_regression": Pipeline(pre + [
            ("clf", LogisticRegression(max_iter=2000, class_weight=cw,
                                       random_state=seed))]),

        "decision_tree": Pipeline(pre + [
            ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                           class_weight=cw, random_state=seed))]),

        "random_forest": Pipeline(pre + [
            ("clf", RandomForestClassifier(n_estimators=400, max_depth=None,
                                           min_samples_leaf=5, n_jobs=-1,
                                           class_weight=cw, random_state=seed))]),
    }


# --------------------------------------------------------------------------
# 3. Evaluation
# --------------------------------------------------------------------------

def score(y_true, y_pred, y_proba=None) -> dict:
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_proba)
        out["avg_precision"] = average_precision_score(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})
    return out


def cross_validate_models(models: dict, X: pd.DataFrame, y: pd.Series,
                          n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Cross-validated performance on the TRAINING data only.

    Use this to choose a model. Do not look at the test set until you have
    chosen - once you tune on the test set it is no longer a test set.
    """
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        try:
            proba = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
        except (AttributeError, IndexError):
            proba = None
        rows.append({"model": name, **score(y, pred, proba)})
    return pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False)


def evaluate_on_test(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate once on test. Do this exactly once, at the end."""
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    try:
        proba = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        proba = None
    return score(y_test, pred, proba)


def stability_check(model, X: pd.DataFrame, y: pd.Series,
                    seeds=(1, 7, 13, 21, 42)) -> pd.DataFrame:
    """Re-run the split and fit under several random seeds.

    A single run is an anecdote. If the F1 score swings by 0.1 across seeds,
    report the spread rather than the best number.
    """
    rows = []
    for s in seeds:
        Xtr, Xte, ytr, yte = split(X, y, seed=s)
        rows.append({"seed": s, **evaluate_on_test(model, Xtr, ytr, Xte, yte)})
    return pd.DataFrame(rows).set_index("seed")


# --------------------------------------------------------------------------
# 4. Interpretation
# --------------------------------------------------------------------------

def importances(model, X_test: pd.DataFrame, y_test: pd.Series,
                n_repeats: int = 10, seed: int = 42) -> pd.DataFrame:
    """Permutation importance: how much worse does the model get when each
    feature is shuffled? Model-agnostic, and it measures what the model
    actually relies on rather than what the algorithm happens to expose.

    For SHAP values (finer-grained, per-prediction) install `shap` separately -
    it is deliberately not a required dependency here.
    """
    r = permutation_importance(model, X_test, y_test, n_repeats=n_repeats,
                               random_state=seed, scoring="f1")
    return (pd.DataFrame({"feature": X_test.columns,
                          "importance": r.importances_mean,
                          "std": r.importances_std})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True))
