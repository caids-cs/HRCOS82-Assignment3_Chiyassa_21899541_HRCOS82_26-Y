"""Figures and tables. Everything written here is regenerable from raw data."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay

FIG = Path("results/figures")
TAB = Path("results/tables")

# A restrained palette that survives greyscale printing, which examiners
# still sometimes use.
INK = "#1f2a37"
ACCENT = "#25506b"
MUTED = "#9aa4ae"


def _prep():
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "grid.color": "#e6e9ec", "axes.grid": True, "grid.linewidth": 0.6,
    })


def save_table(df: pd.DataFrame, name: str) -> Path:
    _prep()
    path = TAB / f"{name}.csv"
    df.to_csv(path)
    return path


def plot_model_comparison(results: pd.DataFrame, name: str = "model_comparison") -> Path:
    """Bar chart of F1 and recall by model, with the majority baseline marked."""
    _prep()
    d = results.copy()
    fig, ax = plt.subplots(figsize=(6.4, 0.55 * len(d) + 1.6))
    y = range(len(d))
    ax.barh([i + 0.19 for i in y], d["f1"], height=0.36, color=ACCENT, label="F1")
    ax.barh([i - 0.19 for i in y], d["recall"], height=0.36, color=MUTED, label="Recall")
    ax.set_yticks(list(y)); ax.set_yticklabels(d.index)
    ax.invert_yaxis(); ax.set_xlim(0, 1); ax.set_xlabel("score")
    ax.set_title("Model comparison (cross-validated on training data)")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="y", visible=False)
    path = FIG / f"{name}.png"
    fig.savefig(path); plt.close(fig)
    return path


def plot_importances(imp: pd.DataFrame, top: int = 15,
                     name: str = "feature_importance") -> Path:
    _prep()
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(d) + 1.4))
    ax.barh(d["feature"], d["importance"], xerr=d["std"],
            color=ACCENT, error_kw={"ecolor": MUTED, "lw": 0.8})
    ax.set_xlabel("drop in F1 when the feature is shuffled")
    ax.set_title("Permutation importance (test set)")
    ax.grid(axis="y", visible=False)
    path = FIG / f"{name}.png"
    fig.savefig(path); plt.close(fig)
    return path


def plot_curves(model, X_test, y_test, name: str = "curves") -> Path:
    """ROC and precision-recall side by side.

    Read the precision-recall curve first. On imbalanced data ROC AUC looks
    flattering; average precision tells you what the model is worth.
    """
    _prep()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    roc = RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[0])
    roc.line_.set_color(ACCENT)
    axes[0].plot([0, 1], [0, 1], ls="--", lw=0.8, color=MUTED)
    axes[0].set_title("ROC")
    pr = PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=axes[1])
    pr.line_.set_color(ACCENT)
    base = float(pd.Series(y_test).mean())
    axes[1].axhline(base, ls="--", lw=0.8, color=MUTED)
    axes[1].set_title(f"Precision-Recall (chance = {base:.2f})")
    for a in axes:
        a.legend(frameon=False, fontsize=8)
    path = FIG / f"{name}.png"
    fig.savefig(path); plt.close(fig)
    return path
