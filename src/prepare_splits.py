from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

keys = ["code_module", "code_presentation", "id_student"]
target = data["final_result_binary"]
groups = data["id_student"]

# Reserve approximately 20% for testing.
# Each learner's enrolments stay together.
outer = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

train_idx, test_idx = next(
    outer.split(data, target, groups=groups)
)

train = data.iloc[train_idx].copy()
test = data.iloc[test_idx].copy()

overlap = set(train["id_student"]) & set(test["id_student"])
assert not overlap, "A learner appears in both partitions!"

# Save enrolment assignments rather than fitting any models yet.
manifest = data[keys].copy()
manifest["split"] = "train"
manifest.loc[test_idx, "split"] = "test"
manifest["validation_fold"] = -1

# Prepare five validation folds using training learners only.
inner = StratifiedGroupKFold(
    n_splits=5, shuffle=True, random_state=42
)

for fold, (fit_idx, validation_idx) in enumerate(
    inner.split(
        train,
        train["final_result_binary"],
        groups=train["id_student"],
    )
):
    fit_learners = set(train.iloc[fit_idx]["id_student"])
    validation_learners = set(
        train.iloc[validation_idx]["id_student"]
    )
    assert fit_learners.isdisjoint(validation_learners)

    original_rows = train.index[validation_idx]
    manifest.loc[original_rows, "validation_fold"] = fold

assert manifest.loc[
    manifest["split"].eq("train"), "validation_fold"
].between(0, 4).all()

destination = PROCESSED / "enrolment_splits_seed42.csv"
manifest.to_csv(destination, index=False)

for name, partition in [("Training", train), ("Test", test)]:
    print(f"\n{name}")
    print("Enrolments:", len(partition))
    print("Unique learners:", partition["id_student"].nunique())
    print(
        "Fail/Withdrawn percentage:",
        round(partition["final_result_binary"].mean() * 100, 2),
    )

print("\nLearner overlap:", len(overlap))
print("Training validation folds: 5")
print("Saved:", destination)