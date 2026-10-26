from pathlib import Path
import json
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from modelling import make_models, score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "oulad" / "tables"

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

primary_results = pd.read_csv(
    TABLES / "test_results.csv"
)

with open(
    ROOT / "results" / "oulad" / "selected_settings.json",
    encoding="utf-8",
) as file:
    settings = json.load(file)

assert settings["random_seed"] == 42

features = [
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

keys = ["code_module", "code_presentation", "id_student"]

assert not data.duplicated(keys).any()
assert data["id_student"].notna().all()

X = data[features]
y = data["final_result_binary"]
groups = data["id_student"]

seeds = [1, 7, 13, 21, 42]
rows = []
split_checks = []

for seed in seeds:
    if seed == 42:
        # Preserve and reuse the primary experiment.
        for row in primary_results.to_dict("records"):
            row["seed"] = seed
            rows.append(row)

        print("Seed 42: reused saved primary results.", flush=True)
        continue

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_learners = set(groups.iloc[train_idx])
    test_learners = set(groups.iloc[test_idx])

    assert train_learners.isdisjoint(test_learners)

    split_checks.append({
        "seed": seed,
        "training_enrolments": len(train_idx),
        "test_enrolments": len(test_idx),
        "training_learners": len(train_learners),
        "test_learners": len(test_learners),
        "learner_overlap": 0,
        "test_positive_rate": y.iloc[test_idx].mean(),
    })

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    models = make_models(seed=seed)

    for name, model in models.items():
        balanced = settings["balanced_by_model"][name]
        fit_options = {}

        if balanced:
            fit_options["clf__sample_weight"] = (
                compute_sample_weight(
                    class_weight="balanced",
                    y=y_train,
                )
            )

        print(
            f"Seed {seed}: fitting {name}...",
            flush=True,
        )

        model.fit(X_train, y_train, **fit_options)

        predicted = model.predict(X_test)
        probabilities = model.predict_proba(X_test)[:, 1]

        rows.append({
            "seed": seed,
            "model": name,
            "balanced": balanced,
            **score(y_test, predicted, probabilities),
        })

    print(f"Seed {seed} complete.\n", flush=True)

results = pd.DataFrame(rows)

# Confirm one result per model for every requested seed.
assert not results.duplicated(["model", "seed"]).any()

for name, group in results.groupby("model"):
    assert set(group["seed"]) == set(seeds)

metrics = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "avg_precision",
]

summary = (
    results.groupby("model")[metrics]
    .agg(["mean", "std"])
)

results.to_csv(
    TABLES / "stability_seed_results.csv",
    index=False,
)

summary.to_csv(
    TABLES / "stability_summary.csv"
)

pd.DataFrame(split_checks).to_csv(
    TABLES / "additional_seed_split_checks.csv",
    index=False,
)

print("\nFIVE-SEED STABILITY RESULTS")
print(summary.round(4).to_string())

print("\nSaved stability tables.")
print("Primary test results and fitted models remain unchanged.")