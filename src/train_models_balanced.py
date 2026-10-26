"Train models"
from pathlib import Path
import pandas as pd

from modelling import make_models, cross_validate_models

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "oulad" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

KEY = ["code_module", "code_presentation", "id_student"]

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

splits = pd.read_csv(
    PROCESSED / "enrolment_splits_seed42.csv"
)

combined = data.merge(
    splits,
    on=KEY,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert combined["_merge"].eq("both").all()
assert combined["split"].isin(["train", "test"]).all()

# Only training enrolments enter cross-validation.
train = combined.loc[combined["split"].eq("train")].copy()

test_learners = set(
    combined.loc[combined["split"].eq("test"), "id_student"]
)

assert set(train["id_student"]).isdisjoint(test_learners)
assert (
    train.groupby("id_student")["validation_fold"]
    .nunique()
    .eq(1)
    .all()
)

feature_columns = [
    "num_of_prev_attempts",
    "studied_credits",
    "total_clicks",
    "active_days",
    "code_module",
    "code_presentation",
    "gender",
    "region",
    "highest_education",
    "imd_band",
    "age_band",
    "disability",
]

X_train = train[feature_columns]
y_train = train["final_result_binary"]
folds = train["validation_fold"]

models = make_models(seed=42)

print("Starting balanced training cross-validation...", flush=True)

fold_results = cross_validate_models(
    models=models,
    X_train=X_train,
    y_train=y_train,
    validation_folds=folds,
    balanced=True,
)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "avg_precision",
]

summary = (
    fold_results
    .groupby("model")[metrics]
    .agg(["mean", "std"])
)

fold_results.to_csv(
    TABLES / "cv_balanced_folds.csv",
    index=False,
)

summary.to_csv(
    TABLES / "cv_balanced_summary.csv"
)

print("\nBALANCED CROSS-VALIDATION RESULTS")
print(summary.round(4).to_string())

print("\nSaved tables to:", TABLES)
print("The reserved test set has not been evaluated.")