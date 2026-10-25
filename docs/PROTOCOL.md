# The CAIDS Modelling Protocol

Every predictive-modelling project in this programme follows this protocol. It
exists so that results from different students are comparable, and so that the
cohort's work can be synthesised into a single study rather than nine unrelated ones.

Where your project departs from the protocol, say so explicitly in your methodology
chapter and justify it. Departing is allowed. Departing silently is not.

---

## 1. Problem framing

State, before you touch the data:

- **The entity.** What is one row? A student, a release, a module registration.
- **The label.** Exactly how is the positive class defined, and when is it observed?
- **The prediction point.** At what moment must the prediction be available for it
  to be useful? This determines the cutoff.
- **The cost asymmetry.** Which error is worse — a missed at-risk student, or an
  unnecessary intervention? Your answer determines which metric leads.

## 2. Data

- Raw data is read-only. Never edit it, never overwrite it.
- Every transformation is a script that reads from `data/raw/` and writes to
  `data/processed/`.
- Record the extraction date and the source system in `results/run_config.json`.
- No personal data enters the repository. Ever.

## 3. The observation window

Features are built **only** from events occurring before the cutoff.

The default is `--cutoff 0.3`: the first 30% of the observation period. Report
results at more than one cutoff (0.25, 0.5, 0.75) so the reader can see how
prediction quality trades against how early the prediction arrives. That trade-off
is usually the most interesting thing in the study.

## 4. Splitting

- Stratified train/test split, 80/20, **before anything is fitted**.
- Model selection uses cross-validation on the training set only.
- The test set is touched **once**, at the end, after the model is chosen.
- If you tune hyperparameters against the test set, it is no longer a test set and
  every number you report from it is optimistic.

## 5. Models

Run, at minimum:

1. `majority_baseline` — the number that means "learned nothing"
2. `logistic_regression` — the real baseline
3. At least one candidate of your own choosing

Report all of them. A candidate that does not beat logistic regression is a
legitimate and publishable result.

## 6. Class imbalance

State the positive rate. Use `class_weight='balanced'` or an explicit resampling
strategy, and say which. Report results both with and without if the difference is
material.

## 7. Metrics

Report **all** of: accuracy, precision, recall, F1, ROC AUC, average precision, and
the full confusion matrix.

Lead with the metric your cost asymmetry justifies — usually recall for at-risk
detection, because a missed at-risk student costs more than an unnecessary phone call.

Never report accuracy alone. Never report accuracy without the majority-class
accuracy beside it.

## 8. Stability

Re-run the split and fit under at least five random seeds. Report mean and standard
deviation, not the best run. A model whose F1 swings by 0.10 across seeds is not a
result you can defend in a viva.

## 9. Interpretation

Report permutation importance at minimum. SHAP values if your project calls for
per-prediction explanation.

Then do the part that matters: say what a practitioner should **do** with the top
features. A feature ranking that suggests no action has explained nothing.

## 10. Reproducibility

- Every figure and table in the write-up is produced by `python run.py`.
- `results/run_config.json` records the settings that produced them.
- If a number in your document cannot be traced to a line of code, remove it.

## 11. Ethics

- Ethical clearance reference on every document: **9396** (HRCOS82 class), or your
  own reference for postgraduate work.
- De-identified or aggregated data only.
- Address in your reflection: what happens to a student wrongly flagged, and what
  happens to one wrongly missed.
