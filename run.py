#!/usr/bin/env python3
"""
CAIDS reference pipeline - the one command that runs everything.

    python run.py                        # synthetic e-learning data
    python run.py --domain devops        # synthetic DevOps data
    python run.py --events data/raw/events.csv --labels data/raw/labels.csv
    python run.py --cutoff 0.5           # predict from the first half of the window

Every figure and table in your report must come out of this command. If a
number in your write-up cannot be produced by running this, it does not go in
the write-up.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from src import data as D
from src import features as F
from src import modelling as M
from src import report as R


def main() -> int:
    ap = argparse.ArgumentParser(description="CAIDS reference pipeline")
    ap.add_argument("--domain", default="elearning", choices=["elearning", "devops"],
                    help="which synthetic generator to use when no data is supplied")
    ap.add_argument("--events", default=None, help="path to a real event CSV")
    ap.add_argument("--labels", default=None, help="path to a real label CSV")
    ap.add_argument("--cutoff", type=float, default=0.3,
                    help="fraction of the observation window used to build features")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-balance", action="store_true",
                    help="turn OFF class_weight='balanced' (report if you do)")
    args = ap.parse_args()

    balanced = not args.no_balance

    # ---- 1. Load -------------------------------------------------------
    if args.events and args.labels:
        print(f"Loading real data from {args.events} and {args.labels}")
        events = D.load_events(args.events)
        labels = D.load_labels(args.labels)
        source = "real"
    else:
        print(f"No data supplied - generating synthetic '{args.domain}' data.")
        print("  (Synthetic results are for learning and testing only.)")
        events, labels = D.make_synthetic(domain=args.domain, seed=args.seed)
        source = f"synthetic:{args.domain}"

    print(f"  {len(events):,} events across {events['entity_id'].nunique():,} entities")
    print(f"  positive rate: {labels['label'].mean():.1%}")

    # ---- 2. Features, built only from before the cutoff ----------------
    print(f"\nBuilding features from the first {args.cutoff:.0%} of the window...")
    feats = F.build_features(events, cutoff_fraction=args.cutoff)
    X, y = F.align(feats, labels)
    print(f"  {X.shape[0]:,} rows x {X.shape[1]} features")

    # ---- 3. Split FIRST ------------------------------------------------
    X_train, X_test, y_train, y_test = M.split(X, y, seed=args.seed)
    print(f"  train {len(X_train):,} / test {len(X_test):,}")

    # ---- 4. Choose a model using the TRAINING data only ----------------
    print("\nCross-validating on training data...")
    models = M.make_models(seed=args.seed, balanced=balanced)
    cv = M.cross_validate_models(models, X_train, y_train, seed=args.seed)
    print(cv[["accuracy", "precision", "recall", "f1"]].round(3).to_string())

    majority_acc = cv.loc["majority_baseline", "accuracy"]
    print(f"\n  Majority-class accuracy is {majority_acc:.3f}.")
    print("  Any accuracy near that number means the model has learned nothing.")

    best_name = cv.drop(index="majority_baseline")["f1"].idxmax()
    print(f"  Best by F1 on cross-validation: {best_name}")

    lr_f1 = cv.loc["logistic_regression", "f1"]
    best_f1 = cv.loc[best_name, "f1"]
    if best_name != "logistic_regression" and best_f1 - lr_f1 < 0.02:
        print(f"  NOTE: {best_name} beats logistic regression by only "
              f"{best_f1 - lr_f1:.3f}. Prefer the simpler model and say so.")

    # ---- 5. Test set, used exactly once --------------------------------
    print(f"\nEvaluating {best_name} on the held-out test set (once)...")
    best = M.make_models(seed=args.seed, balanced=balanced)[best_name]
    test_scores = M.evaluate_on_test(best, X_train, y_train, X_test, y_test)
    print("  " + "  ".join(f"{k}={v:.3f}" for k, v in test_scores.items()
                           if isinstance(v, float)))
    print(f"  confusion: tn={test_scores['tn']} fp={test_scores['fp']} "
          f"fn={test_scores['fn']} tp={test_scores['tp']}")

    # ---- 6. Stability across seeds -------------------------------------
    print("\nRe-running under several random seeds...")
    stab = M.stability_check(M.make_models(seed=args.seed, balanced=balanced)[best_name], X, y)
    print(f"  F1 mean {stab['f1'].mean():.3f}, sd {stab['f1'].std():.3f}, "
          f"range {stab['f1'].min():.3f}-{stab['f1'].max():.3f}")

    # ---- 7. Interpretation ---------------------------------------------
    print("\nComputing permutation importance...")
    imp = M.importances(best, X_test, y_test, seed=args.seed)
    print(imp.head(6).round(4).to_string(index=False))

    # ---- 8. Write everything out ---------------------------------------
    R.save_table(cv, "cv_results")
    R.save_table(pd.DataFrame([test_scores], index=[best_name]), "test_results")
    R.save_table(stab, "stability")
    R.save_table(imp, "feature_importance")
    R.plot_model_comparison(cv.drop(index="majority_baseline"))
    R.plot_importances(imp)
    R.plot_curves(best, X_test, y_test)

    Path("results").mkdir(exist_ok=True)
    Path("results/run_config.json").write_text(json.dumps({
        "source": source, "cutoff_fraction": args.cutoff, "seed": args.seed,
        "class_weight_balanced": balanced, "selected_model": best_name,
        "n_entities": int(X.shape[0]), "n_features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
    }, indent=2))

    print("\nWritten to results/ - figures/, tables/ and run_config.json")
    print("Commit results/ so your supervisor can see what you produced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
