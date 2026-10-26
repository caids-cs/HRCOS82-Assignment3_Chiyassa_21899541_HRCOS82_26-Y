# Start here: OULAD study

**Author:** Carol Chiyassa
**Student number:** 21899541
**Module:** HRCOS82
**Ethics clearance reference:** 9396 (HRCOS82 class)

## 1. Understand the project

This study compares logistic regression, Explainable Boosting Machines
(EBM), and XGBoost for predicting academic risk using OULAD.
A majority-class classifier provides a reference baseline.

Predictors include learner characteristics and activity recorded on
days 0–30 inclusive. Prediction is made after day 30.

The target is:

- 0: Pass or Distinction.
- 1: Fail or Withdrawn.

This is an academic-risk target, not a direct engagement label.

Read:

- `README.md` for the study overview and results.
- `data/README.md` for data preparation and handling.
- `docs/PROTOCOL.md` for the original CAIDS reference protocol.
- `docs/STUDY_ADAPTATIONS.md` for the study's documented adaptations.

## 2. Open the repository in VS Code

Select **File → Open Folder** and open the project folder.

Select **Terminal → New Terminal**. On Windows, use PowerShell.
Run commands from the project root, where `run.py` is located.

Commands belong in the terminal. Python code and documentation belong
in their respective files in the editor.

## 3. Set up Python

The recorded study environment uses Python 3.14.0 on Windows 11.
Miniconda is not required.

For a fresh installation, check Python:

```powershell
python --version
```

Create a virtual environment if `.venv` does not already exist:

```powershell
python -m venv .venv
```

Install the dependencies into that environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

In VS Code, open the Command Palette with **Ctrl + Shift + P**,
select **Python: Select Interpreter**, and choose:

```text
.venv\Scripts\python.exe
```

The commands below use the environment's Python directly, so activation
is optional.

The recorded package environment is available in
`results/oulad/environment_freeze.txt`.

## 4. Prepare the source data

Obtain the original Open University Learning Analytics Dataset (OULAD).
Place these files in `data/raw/`:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`

Keep the source files unchanged.

The preparation script creates enrolment-level features and excludes
enrolments with recorded withdrawal on or before day 30.
Eligible enrolments without activity receive zero activity counts.

Raw data, processed records, individual predictions, and fitted models
remain local and are excluded from version control.

## 5. Check the installation

Check installed dependency consistency:

```powershell
.\.venv\Scripts\python.exe -m pip check
```

Run the software tests:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_pipeline.py -q
```

Check that the required source files and scripts exist:

```powershell
.\.venv\Scripts\python.exe run.py --check
```

The file check does not train models or validate data contents.
The software tests check modelling components using artificial records;
they do not replace checks on the actual study data.

## 6. Run the analysis only when needed

For a fresh analysis with the required data available:

```powershell
.\.venv\Scripts\python.exe run.py
```

The runner performs these steps:

1. Prepare OULAD features.
2. Create learner-separated training, test, and validation allocations.
3. Validate the data and allocations.
4. Compare unweighted models using training cross-validation.
5. Compare balanced models using training cross-validation.
6. Select settings using mean validation positive-class F1.
7. Fit the selected configurations and evaluate the reserved test set.
8. Generate confusion matrices and ROC curves.
9. Calculate permutation importance.
10. Calculate paired learner-bootstrap intervals.
11. Assess five-seed predictive stability.
12. Record experimental settings and the installed environment.

The runner stops if `results/oulad/` already contains outputs.
This protects completed results from an accidental rerun.

Do not delete completed results just to bypass this protection.
A deliberate rerun should use a separate working copy or first preserve
both the results and the associated local processed files.
Keep archived data and models excluded from Git.

Existing published results can be inspected without retraining.
The test set must not be used to choose new settings or features.

## 7. Find the results and graphs

The actual study outputs are in `results/oulad/`.

### Tables

Open `results/oulad/tables/` for:

- Unweighted and balanced cross-validation results.
- Selected model settings.
- Primary held-out test results.
- Permutation importance.
- Paired bootstrap comparisons.
- Five-seed stability results.

The primary test table is:

```text
results/oulad/tables/test_results.csv
```

### Graphs

Open `results/oulad/figures/` for:

- `test_confusion_matrices.png`
- `test_roc_curves.png`
- `permutation_importance_logistic_regression.png`
- `permutation_importance_ebm.png`
- `permutation_importance_xgboost.png`

### Reproducibility records

- `results/oulad/selected_settings.json`
- `results/oulad/experimental_settings.json`
- `results/oulad/environment_freeze.txt`

The environment record was captured after analysis.

Old template outputs in the local `template_archive/` folder are not
study findings and are excluded from version control.

## 8. Interpret the findings accurately

Positive-class precision, recall, and F1 refer to Fail or Withdrawn.

Logistic regression was selected using training validation and achieved
the highest primary test F1 among the compared models. Other models
performed better on some other metrics.

Permutation importance describes how much a model relies on a predictor.
It does not establish causality or the direction of an association.

Bootstrap intervals describe uncertainty under the specified resampling
procedure. Five-seed standard deviations describe variation across those
runs; they are not confidence intervals.

Predictive stability does not demonstrate explanation stability.
Educator comprehension and intervention effectiveness were not evaluated.

## 9. Check files before committing

Review the working tree:

```powershell
git status --short
```

Check tracked data and environment paths:

```powershell
git ls-files data .venv
```

Expected tracked files under `data/` are its README and `.gitkeep`
placeholders. The virtual environment should not be tracked.

Stage only intended source code, documentation, tests, configuration,
and aggregate study outputs. Review staged changes before committing:

```powershell
git diff --cached --stat
```

Keep raw records, processed records, individual predictions, fitted models,
credentials, and local archives out of the repository.

If learner-level data is accidentally committed, follow institutional
reporting and remediation procedures. Removing a file from the current
folder does not remove it from Git history.

## 10. Understand GitHub checks

The GitHub workflow installs dependencies and runs the software tests
using Python 3.14 on a Windows runner.

It does not run the full OULAD analysis because the source data is not
included in the repository.

A successful workflow confirms that the configured software checks pass.
It does not independently verify every research finding.

## Attribution and ethics

This repository is adapted from the CAIDS Reference Pipeline by
Themba Masombuka, Centre for Augmented Intelligence and Data Sciences,
University of South Africa.

Retain the original template's licence and attribution.

The HRCOS82 class ethics clearance reference is **9396**.
Follow applicable clearance conditions, institutional requirements,
and data-protection obligations when handling learner records.
