"""
Model development and evaluation for the OULAD study.

Study comparison:
- Logistic regression
- Explainable Boosting Machine (EBM)
- XGBoost

A majority-class classifier provides an additional reference.

Preprocessing is fitted within training folds. Evaluation uses
learner-separated partitions and reports predictive performance,
variability across random seeds, and permutation importance.

Implementation status:
The functions below are being adapted from the supplied template.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from interpret.glassbox import ExplainableBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline




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
def make_preprocessor():
    """Prepare numeric and categorical OULAD predictors."""

    numeric_features = [
        "num_of_prev_attempts",
        "studied_credits",
        "total_clicks",
        "active_days",
    ]

    categorical_features = [
        "code_module",
        "code_presentation",
        "gender",
        "region",
        "highest_education",
        "imd_band",
        "age_band",
        "disability",
    ]

    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
        )),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ], remainder="drop")

def make_models(seed: int = 42) -> dict[str, Pipeline]:
    """Create the study models with separate preprocessing pipelines."""

    classifiers = {
        "majority_baseline": DummyClassifier(
            strategy="most_frequent"
        ),

        "logistic_regression": LogisticRegression(
            C=1.0,
            max_iter=5000,
            random_state=seed,
        ),

        "ebm": ExplainableBoostingClassifier(
            interactions=0,
            outer_bags=8,
            max_rounds=2000,
            random_state=seed,
            n_jobs=2,
        ),

        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=1.0,
            colsample_bytree=1.0,
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=2,
        ),
    }

    models = {}

    for name, classifier in classifiers.items():
        models[name] = Pipeline([
            ("preprocess", make_preprocessor()),
            ("clf", classifier),
        ])

    return models


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


def cross_validate_models(
    models,
    X_train,
    y_train,
    validation_folds,
    balanced=False,
):
    """Evaluate models using the saved training validation folds."""

    # Ensure predictors, labels and fold assignments line up.
    assert X_train.index.equals(y_train.index)
    assert X_train.index.equals(validation_folds.index)
    assert set(validation_folds.unique()) == set(range(5))

    rows = []

    for name, original_model in models.items():
        # The majority reference should remain unweighted.
        use_weights = balanced and name != "majority_baseline"

        for fold in range(5):
            fit_mask = validation_folds.ne(fold)
            validation_mask = validation_folds.eq(fold)

            X_fit = X_train.loc[fit_mask]
            y_fit = y_train.loc[fit_mask]

            X_validation = X_train.loc[validation_mask]
            y_validation = y_train.loc[validation_mask]

            # Start each fold with a fresh, unfitted pipeline.
            model = clone(original_model)

            fit_options = {}

            if use_weights:
                fit_options["clf__sample_weight"] = (
                    compute_sample_weight(
                        class_weight="balanced",
                        y=y_fit,
                    )
                )

            model.fit(X_fit, y_fit, **fit_options)

            predictions = model.predict(X_validation)
            probabilities = model.predict_proba(
                X_validation
            )[:, 1]

            rows.append({
                "model": name,
                "balanced": use_weights,
                "fold": fold,
                **score(
                    y_validation,
                    predictions,
                    probabilities,
                ),
            })

            print(
                f"{name}: fold {fold + 1}/5 complete "
                f"(balanced={use_weights})",
                flush=True,
            )

    return pd.DataFrame(rows)


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