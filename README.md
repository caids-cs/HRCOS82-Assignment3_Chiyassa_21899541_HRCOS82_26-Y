# An Interpretable Machine Learning Framework for Analysing Learner Engagement Patterns in E-Learning Environments

**Author:** Carol Mwaba Chiyassa
**Student number:** 21899541
**Module:** HRCOS82,BSc Honours Computing, UNISA

## Study overview

This repository compares logistic regression, Explainable Boosting Machines
(EBM), and XGBoost using the Open University Learning Analytics Dataset
(OULAD). A majority-class classifier provides a reference baseline.

Early online activity and learner characteristics are used to predict final
academic risk. Activity measures represent aspects of behavioural engagement;
the prediction target is an academic outcome, not a direct engagement label.

The supplied assignment template was used as a structural guide. The results
reported here come from the OULAD analysis, not the template's synthetic example.

## Data and prediction window

Features use activity recorded on days 0–30 inclusive. Prediction is therefore
made after day 30.

Enrolments with recorded withdrawal on or before day 30 are excluded.
Missing withdrawal dates are retained as no recorded withdrawal.
Eligible enrolments without activity in the window receive zero activity counts.

The final dataset contains 27,466 enrolments from 24,787 unique learners.

Target coding:

- 0: Pass or Distinction — 15,385 enrolments.
- 1: Fail or Withdrawn — 12,081 enrolments.

Numeric predictors:

- Previous attempts.
- Studied credits.
- Total clicks during the observation window.
- Number of active days during the observation window.

Categorical predictors:

- Module and presentation.
- Gender and region.
- Highest education and deprivation band.
- Age band and disability status.

Learner identifiers are used for grouping, not prediction. Assessment scores
are excluded because their availability at the prediction point was not verified.

## Evaluation design

The primary split uses random seed 42:

- Training: 21,973 enrolments from 19,831 learners.
- Test: 5,493 enrolments from 4,956 learners.
- Learner overlap between training and test: zero.

Five learner-separated validation folds are used within the training data.
Preprocessing is fitted within each training fold.

Unweighted and balanced training are compared using mean validation
positive-class F1. Balanced training is selected for all three candidate models.
Logistic regression is selected before test evaluation because it has the
highest mean validation F1.

The test set is used for final evaluation of the fixed model configurations.
Decision thresholds are not tuned.

## Primary held-out test results

Precision, recall, and F1 refer to the positive class: Fail or Withdrawn.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Average precision |
|---|---:|---:|---:|---:|---:|---:|
| Majority baseline | 0.5602 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.4398 |
| Logistic regression | 0.6785 | 0.6227 | 0.6825 | 0.6513 | 0.7384 | 0.6951 |
| EBM | 0.6829 | 0.6343 | 0.6589 | 0.6464 | 0.7420 | 0.7062 |
| XGBoost | 0.6836 | 0.6410 | 0.6378 | 0.6394 | 0.7413 | 0.7078 |

Logistic regression has the highest test F1 and recall. XGBoost has the
highest accuracy, precision, and average precision. EBM has the highest
test ROC-AUC. No model leads on every metric.

## Interpretation and uncertainty

Permutation importance measures the decrease in positive-class F1 after
shuffling a predictor. Active days, module, and highest education are the
three highest-ranked predictors for each candidate model.

These importance values describe model reliance. They do not establish
causality or the direction of an association.

A paired bootstrap with 1,000 learner-level resamples gives:

| F1 comparison | Observed difference | 95% percentile interval |
|---|---:|---|
| EBM minus logistic regression | -0.00490 | [-0.01192, 0.00161] |
| XGBoost minus logistic regression | -0.01184 | [-0.02012, -0.00337] |

The EBM comparison includes zero and does not establish a clear F1 difference.
The XGBoost comparison favours logistic regression under this procedure.
These exploratory intervals are conditional on the fitted models and are
not adjusted for multiple comparisons.

Across five seeds, mean F1 and sample standard deviation are:

- Logistic regression: 0.6522 ± 0.0026.
- EBM: 0.6495 ± 0.0022.
- XGBoost: 0.6428 ± 0.0036.

These runs assess sensitivity to splitting and fitting randomness with
the selected settings held fixed. They are not independent replications.

## Results and figures

Study outputs are stored in `results/oulad/`.

- `tables/`: validation, test, importance, bootstrap, and stability results.
- `figures/test_confusion_matrices.png`: test classification counts.
- `figures/test_roc_curves.png`: test ROC curves.
- `figures/permutation_importance_logistic_regression.png`.
- `figures/permutation_importance_ebm.png`.
- `figures/permutation_importance_xgboost.png`.
- `selected_settings.json`: settings selected using training validation.
- `experimental_settings.json`: recorded experiment configuration.
- `environment_freeze.txt`: recorded installed package versions.

Fitted models and individual predictions are stored locally under
`data/processed/`.

## Running the project

Use a Python virtual environment and install the project dependencies
from `requirements.txt`.

Place the original OULAD files below in `data/raw/`:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`

Check required file availability:

    python run.py --check

Run the complete workflow in a fresh output directory:

    python run.py

The runner stops when existing study outputs are present, protecting them
from an accidental rerun. The file check only checks file availability;
it does not validate the analysis.

The recorded environment uses Python 3.14.0 on Windows 11. Full package
versions are recorded in `results/oulad/environment_freeze.txt`.

## Limitations

The study uses one dataset and one observation window. Learners who register
late have shorter exposure within that window. Results do not establish
performance at other institutions or on future presentations.

Predictive stability does not demonstrate explanation stability.
Educator comprehension, intervention effectiveness, and causal effects
were not evaluated.
