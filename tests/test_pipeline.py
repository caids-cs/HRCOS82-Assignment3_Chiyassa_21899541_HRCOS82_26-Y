"""Small software checks
"""

import numpy as np
import pandas as pd
import pytest

from sklearn.pipeline import Pipeline

from src.modelling import make_models, make_preprocessor, score


@pytest.fixture
def example_features():
    """Four artificial records matching the study's predictor columns."""
    return pd.DataFrame({
        "num_of_prev_attempts": [0, 1, 0, 2],
        "studied_credits": [60, 120, 60, 120],
        "total_clicks": [10, 20, 30, 40],
        "active_days": [1, 2, 3, 4],
        "code_module": ["AAA", "AAA", "BBB", "BBB"],
        "code_presentation": ["2013J"] * 4,
        "gender": ["F", "M", "F", "M"],
        "region": ["Region A"] * 4,
        "highest_education": ["A Level or Equivalent"] * 4,
        "imd_band": ["20-30%"] * 4,
        "age_band": ["0-35"] * 4,
        "disability": ["N"] * 4,
    })


def test_models_have_separate_preprocessing():
    models = make_models()

    assert set(models) == {
        "majority_baseline",
        "logistic_regression",
        "ebm",
        "xgboost",
    }

    preprocessors = []
    for model in models.values():
        assert isinstance(model, Pipeline)
        assert "preprocess" in model.named_steps
        assert model.steps[-1][0] == "clf"
        preprocessors.append(model.named_steps["preprocess"])

    assert len({id(item) for item in preprocessors}) == len(models)


def test_transform_handles_missing_values_and_unseen_categories(
    example_features,
):
    preprocessor = make_preprocessor()
    training = preprocessor.fit_transform(example_features)

    held_out = example_features.iloc[[0]].copy()
    held_out["total_clicks"] = np.nan
    held_out["region"] = "Previously unseen region"

    transformed = preprocessor.transform(held_out)

    assert transformed.shape == (1, training.shape[1])
    assert np.isfinite(transformed).all()


def test_transform_does_not_refit_training_statistics(example_features):
    preprocessor = make_preprocessor()
    preprocessor.fit(example_features)

    scaler = (
        preprocessor.named_transformers_["numeric"]
        .named_steps["scale"]
    )
    original_mean = scaler.mean_.copy()

    # Training means for the four numeric predictors.
    np.testing.assert_allclose(original_mean, [0.75, 90, 25, 2.5])

    held_out = example_features.iloc[[0]].copy()
    held_out["total_clicks"] = 1_000_000
    preprocessor.transform(held_out)

    np.testing.assert_array_equal(scaler.mean_, original_mean)


def test_score_uses_positive_class_and_correct_confusion_counts():
    actual = np.array([0, 0, 0, 1, 1])
    predicted = np.array([0, 1, 0, 0, 1])

    result = score(actual, predicted)

    assert result["accuracy"] == pytest.approx(0.6)
    assert result["precision"] == pytest.approx(0.5)
    assert result["recall"] == pytest.approx(0.5)
    assert result["f1"] == pytest.approx(0.5)
    assert [result[key] for key in ["tn", "fp", "fn", "tp"]] == [
        2, 1, 1, 1
    ]

def test_majority_predictions_have_zero_positive_recall():
    actual = np.array([0] * 8 + [1] * 2)
    predicted = np.zeros(10, dtype=int)
    probabilities = np.full(10, 0.2)

    result = score(actual, predicted, probabilities)

    assert result["accuracy"] == pytest.approx(0.8)
    assert result["recall"] == 0.0
    assert result["f1"] == 0.0
    assert result["roc_auc"] == pytest.approx(0.5)
    assert result["avg_precision"] == pytest.approx(0.2)
