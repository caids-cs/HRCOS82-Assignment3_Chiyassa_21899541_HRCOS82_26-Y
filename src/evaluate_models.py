"Evaluat e the models on the test set and save the results to a CSV file."

from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.utils.class_weight import compute_sample_weight
from modelling import make_models, score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "oulad"
TABLES = RESULTS / "tables"

# Fitted models stay inside the git-ignored processed-data folder.
MODEL_FOLDER = PROCESSED / "fitted_models"
MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

output_file = TABLES / "test_results.csv"

if output_file.exists():
    raise FileExistsError(
        "test_results.csv already exists. "
        "Keep the existing results and check before rerunning."
    )

with open(
    RESULTS / "selected_settings.json",
    encoding="utf-8",
) as file:
    settings = json.load(file)

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

splits = pd.read_csv(
    PROCESSED / "enrolment_splits_seed42.csv"
)

keys = ["code_module", "code_presentation", "id_student"]

combined = data.merge(
    splits,
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert combined["_merge"].eq("both").all()
assert combined["split"].isin(["train", "test"]).all()

train = combined.loc[combined["split"].eq("train")].copy()
test = combined.loc[combined["split"].eq("test")].copy()

assert set(train["id_student"]).isdisjoint(test["id_student"])

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

X_train = train[features]
y_train = train["final_result_binary"]

X_test = test[features]
y_test = test["final_result_binary"]

models = make_models(seed=settings["random_seed"])
rows = []

# Save predictions privately for later graphs and paired comparisons.
predictions = test[keys].copy()
predictions["actual"] = y_test

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

    print(f"Fitting {name} on all training data...", flush=True)
    model.fit(X_train, y_train, **fit_options)

    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    rows.append({
        "model": name,
        "balanced": balanced,
        **score(y_test, predicted, probabilities),
    })

    predictions[f"{name}_prediction"] = predicted
    predictions[f"{name}_probability"] = probabilities

    joblib.dump(
        model,
        MODEL_FOLDER / f"{name}.joblib",
    )

test_results = pd.DataFrame(rows)
test_results.to_csv(output_file, index=False)

predictions.to_csv(
    PROCESSED / "test_predictions.csv",
    index=False,
)

print("\nFINAL HELD-OUT TEST RESULTS")
print(test_results.round(4).to_string(index=False))

print("\nSaved results:", output_file)
print("Saved fitted models:", MODEL_FOLDER)
print("Saved predictions privately in data/processed.")