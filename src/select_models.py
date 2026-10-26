from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "oulad"
TABLES = RESULTS / "tables"

unweighted = pd.read_csv(
    TABLES / "cv_unweighted_folds.csv"
)

balanced = pd.read_csv(
    TABLES / "cv_balanced_folds.csv"
)

# The majority reference is identical in both runs.
# Keep it only once.
balanced = balanced.loc[
    balanced["model"].ne("majority_baseline")
]

combined = pd.concat(
    [unweighted, balanced],
    ignore_index=True,
)

# Verify that each model/weighting option has all five folds.
for _, group in combined.groupby(["model", "balanced"]):
    assert len(group) == 5
    assert set(group["fold"]) == set(range(5))

summary = (
    combined.groupby(["model", "balanced"], as_index=False)
    .agg(
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
    )
)

# Choose each model's weighting using validation results only.
# In an exact tie, prefer unweighted fitting.
selected = (
    summary.sort_values(
        ["model", "mean_f1", "balanced"],
        ascending=[True, False, True],
    )
    .drop_duplicates("model")
)

candidates = selected.loc[
    selected["model"].ne("majority_baseline")
]

best_model = candidates.loc[
    candidates["mean_f1"].idxmax(), "model"
]

settings = {
    "random_seed": 42,
    "selection_metric": "mean validation positive-class F1",
    "selected_model": best_model,
    "balanced_by_model": {
        row["model"]: bool(row["balanced"])
        for _, row in selected.iterrows()
    },
    "decision_threshold": "estimator default; not tuned",
}

summary.to_csv(
    TABLES / "weighting_comparison.csv",
    index=False,
)

selected.to_csv(
    TABLES / "selected_validation_settings.csv",
    index=False,
)

with open(
    RESULTS / "selected_settings.json",
    "w",
    encoding="utf-8",
) as file:
    json.dump(settings, file, indent=2)

print(selected.round(4).to_string(index=False))
print("\nSelected model:", best_model)
print("\nSettings saved. No test results were used.")