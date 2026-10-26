# Study data

This study uses the Open University Learning Analytics Dataset (OULAD).

Raw data, processed learner records, individual predictions, and fitted models
are stored locally and excluded from version control. This README and the
`.gitkeep` placeholders are included in the repository.

## Ethics and data protection

This research is conducted under ethics clearance **9396** and follows the
Protection of Personal Information Act (POPIA).

Learner data must be handled in accordance with the applicable clearance
conditions and institutional requirements. Raw data, processed learner records,
individual predictions, and fitted models are kept locally and excluded from
the repository.

## Required source files

Place these original OULAD files in `data/raw/`:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentVle.csv`

Keep the original source files unchanged. Data preparation is performed by
`src/prepare_oulad.py`.

## Observation window and eligibility

Activity features use days 0–30 inclusive, representing prediction after day 30.

Enrolments with a recorded withdrawal date on or before day 30 are excluded.
Enrolments with missing withdrawal dates are retained.
Eligible enrolments without activity in the window receive zero activity counts.

Late registrations are retained, so not every learner has the same length
of exposure within the observation window.

## Prediction target

The binary target is `final_result_binary`:

- 0: Pass or Distinction.
- 1: Fail or Withdrawn.

This is an academic-risk outcome, not a direct measure of learner engagement.

## Generated local files

The workflow creates the following under `data/processed/`:

- `oulad_30day_activity_verified.csv`: prepared enrolment-level features and target.
- `enrolment_splits_seed42.csv`: primary train/test allocation and validation folds.
- `fitted_models/`: fitted preprocessing and model pipelines.
- `test_predictions.csv`: individual held-out predictions and probabilities.

Learner identifiers are retained for grouping and matching records.
They are not used as model predictors.

The primary training and test sets contain no overlapping learners.
Validation folds also separate learners.

## Reproducibility records

Aggregate study results and figures are stored in `results/oulad/`.

Configuration, package versions, and file fingerprints are recorded in:

- `results/oulad/experimental_settings.json`
- `results/oulad/environment_freeze.txt`

The record describes the environment when it was captured after analysis.

## Repository handling

Keep raw and processed data excluded through `.gitignore`.
Only aggregate results and approved study documentation should be staged.

Before committing, check:

    git ls-files data .venv
    git status --short

The expected tracked contents under `data/` are this README and the
`.gitkeep` placeholders. The virtual environment should not be tracked.