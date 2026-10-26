"Compare bootstrap samples."
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "results" / "oulad" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

data = pd.read_csv(PROCESSED / "test_predictions.csv")

keys = ["code_module", "code_presentation", "id_student"]
assert not data.duplicated(keys).any()
assert data["id_student"].notna().all()

actual = data["actual"].to_numpy()

predictions = {
    name: data[f"{name}_prediction"].to_numpy()
    for name in ["logistic_regression", "ebm", "xgboost"]
}

assert np.isin(actual, [0, 1]).all()
for values in predictions.values():
    assert np.isin(values, [0, 1]).all()

# Store row positions belonging to each test learner.
learner_rows = {}

for position, learner in enumerate(data["id_student"]):
    learner_rows.setdefault(learner, []).append(position)

learner_ids = np.array(list(learner_rows))

rng = np.random.default_rng(42)
repetitions = 1000

differences = {
    "ebm": [],
    "xgboost": [],
}

def positive_f1(y, predicted):
    return f1_score(
        y,
        predicted,
        pos_label=1,
        zero_division=0,
    )

for repeat in range(repetitions):
    sampled_learners = rng.choice(
        learner_ids,
        size=len(learner_ids),
        replace=True,
    )

    # A repeatedly sampled learner contributes their rows repeatedly.
    positions = np.concatenate([
        learner_rows[learner]
        for learner in sampled_learners
    ])

    sampled_actual = actual[positions]

    logistic_f1 = positive_f1(
        sampled_actual,
        predictions["logistic_regression"][positions],
    )

    for name in differences:
        candidate_f1 = positive_f1(
            sampled_actual,
            predictions[name][positions],
        )

        differences[name].append(candidate_f1 - logistic_f1)

    if (repeat + 1) % 200 == 0:
        print(
            f"Completed {repeat + 1}/{repetitions} resamples",
            flush=True,
        )

original_logistic_f1 = positive_f1(
    actual,
    predictions["logistic_regression"],
)

rows = []

for name, values in differences.items():
    lower, upper = np.percentile(values, [2.5, 97.5])

    observed = (
        positive_f1(actual, predictions[name])
        - original_logistic_f1
    )

    rows.append({
        "comparison": f"{name} minus logistic_regression",
        "metric": "positive_class_F1",
        "observed_difference": observed,
        "CI_95_lower": lower,
        "CI_95_upper": upper,
        "includes_zero": bool(lower <= 0 <= upper),
        "bootstrap_repetitions": repetitions,
    })

results = pd.DataFrame(rows)

results.to_csv(
    TABLES / "paired_bootstrap_comparison.csv",
    index=False,
)

print("\nPAIRED LEARNER BOOTSTRAP")
print(results.round(5).to_string(index=False))