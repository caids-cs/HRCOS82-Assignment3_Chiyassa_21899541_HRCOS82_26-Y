"Explain the models."
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from modelling import importances

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "oulad"
TABLES = RESULTS / "tables"
FIGURES = RESULTS / "figures"

TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

keys = ["code_module", "code_presentation", "id_student"]

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

saved_predictions = pd.read_csv(
    PROCESSED / "test_predictions.csv"
)

# Recover the exact test enrolments used for evaluation.
test = saved_predictions.merge(
    data,
    on=keys,
    how="left",
    validate="one_to_one",
    indicator=True,
)

assert test["_merge"].eq("both").all()
assert test["actual"].eq(test["final_result_binary"]).all()

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

X_test = test[features]
y_test = test["actual"]

models = {
    "logistic_regression": "Logistic regression",
    "ebm": "EBM",
    "xgboost": "XGBoost",
}

for name, label in models.items():
    model = joblib.load(
        PROCESSED / "fitted_models" / f"{name}.joblib"
    )

    # Confirm that the model and data reproduce saved predictions.
    np.testing.assert_array_equal(
        model.predict(X_test),
        test[f"{name}_prediction"].to_numpy(),
    )

    np.testing.assert_allclose(
        model.predict_proba(X_test)[:, 1],
        test[f"{name}_probability"].to_numpy(),
        rtol=1e-7,
        atol=1e-9,
    )

    print(f"Calculating importance for {label}...", flush=True)

    importance = importances(
        model,
        X_test,
        y_test,
        n_repeats=10,
        seed=42,
    )

    importance.to_csv(
        TABLES / f"permutation_importance_{name}.csv",
        index=False,
    )

    # Sort so the largest importance appears at the top.
    plot_data = importance.sort_values("importance")

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        plot_data["feature"],
        plot_data["importance"],
        xerr=plot_data["std"],
        color="#4472C4",
        capsize=3,
    )

    ax.axvline(0, color="grey", linewidth=1)
    ax.set_xlabel("Mean decrease in positive-class F1")
    ax.set_title(f"Permutation importance — {label}")

    fig.tight_layout()
    fig.savefig(
        FIGURES / f"permutation_importance_{name}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)

    print(importance.head(5).round(4).to_string(index=False))
    print()

print("Finished. Three tables and three graphs saved.")