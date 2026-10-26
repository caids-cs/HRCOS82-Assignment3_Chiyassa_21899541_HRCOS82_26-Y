"Plot the test results of the evaluated models."
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "results" / "oulad" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

predictions = pd.read_csv(
    PROCESSED / "test_predictions.csv"
)

actual = predictions["actual"]

models = {
    "logistic_regression": "Logistic regression",
    "ebm": "EBM",
    "xgboost": "XGBoost",
}

# Confirm that all saved predictions are available.
assert actual.isin([0, 1]).all()

for name in models:
    assert predictions[f"{name}_prediction"].isin([0, 1]).all()
    assert predictions[f"{name}_probability"].between(0, 1).all()

# Figure 1: confusion matrices.
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, (name, label) in zip(axes, models.items()):
    matrix = confusion_matrix(
        actual,
        predictions[f"{name}_prediction"],
        labels=[0, 1],
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Pass/Distinction", "Fail/Withdrawn"],
    )

    display.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
        values_format="d",
    )

    ax.set_title(label)
    ax.tick_params(axis="x", labelrotation=15)

fig.tight_layout()
fig.savefig(
    FIGURES / "test_confusion_matrices.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

# Figure 2: ROC curves.
fig, ax = plt.subplots(figsize=(7, 6))

for name, label in models.items():
    probabilities = predictions[f"{name}_probability"]

    false_positive_rate, true_positive_rate, _ = roc_curve(
        actual,
        probabilities,
    )

    auc = roc_auc_score(actual, probabilities)

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{label} (AUC = {auc:.4f})",
    )

ax.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    color="grey",
    label="Chance reference",
)

ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC curves on the held-out test set")
ax.legend(loc="lower right")
ax.grid(alpha=0.2)

fig.tight_layout()
fig.savefig(
    FIGURES / "test_roc_curves.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print("Saved:")
print(FIGURES / "test_confusion_matrices.png")
print(FIGURES / "test_roc_curves.png")