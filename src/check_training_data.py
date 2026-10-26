"""
Check the training data for any issues before training the model.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
KEY = ["code_module", "code_presentation", "id_student"]

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

splits = pd.read_csv(
    PROCESSED / "enrolment_splits_seed42.csv"
)

# Match enrolments by their identifiers, not their row positions.
combined = data.merge(
    splits,
    on=KEY,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert combined["_merge"].eq("both").all(), (
    "The dataset and split file contain different enrolments."
)

assert combined["split"].isin(["train", "test"]).all()

train = combined.loc[combined["split"].eq("train")].copy()
test = combined.loc[combined["split"].eq("test")].copy()

assert set(train["id_student"]).isdisjoint(test["id_student"])
assert train["validation_fold"].isin(range(5)).all()
assert test["validation_fold"].eq(-1).all()

# Every training learner must belong to only one validation fold.
assert (
    train.groupby("id_student")["validation_fold"]
    .nunique()
    .eq(1)
    .all()
)

print("Training enrolments:", len(train))
print("Test enrolments:", len(test))
print("Learner overlap: 0")

print("\nTraining enrolments per validation fold:")
print(train["validation_fold"].value_counts().sort_index())

print("\nAll checks passed.")