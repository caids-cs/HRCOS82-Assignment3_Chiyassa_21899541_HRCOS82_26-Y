# Study adaptations to the CAIDS reference protocol

Author: Carol Chiyassa
Student number: 21899541
Module: HRCOS82
Assignment 3
Ethics clearance reference: 9396 (HRCOS82 class)

## Purpose

The CAIDS template provides the repository structure and reference protocol.
This document records adaptations made for the OULAD study.
The original protocol is retained in `docs/PROTOCOL.md`.

## Unit of analysis and target

Each row represents a learner's enrolment in a module presentation.
A learner may therefore contribute more than one enrolment.

The positive class is Fail or Withdrawn; the negative class is Pass or
Distinction. This target represents academic risk. Early activity features
describe aspects of behavioural engagement but do not constitute a direct
engagement label.

## Observation window

The study uses days 0–30 inclusive, with prediction after day 30.
This replaces the template's proportional observation-window cutoff.

A fixed window supports comparison at a common point relative to module start.
However, late registrations have shorter exposure within that window.

Enrolments with recorded withdrawal on or before day 30 are excluded.
Missing withdrawal dates are retained.
Multiple observation windows were not evaluated; this is a study limitation.

## Data representation

The study uses an enrolment-level table prepared directly from OULAD's
student information, registration, and virtual learning environment records.

This replaces the template's generic events.csv and labels.csv schema.
Preparation is implemented in `src/prepare_oulad.py`.

Assessment scores are excluded because their availability at the prediction
point was not verified.

## Learner-separated evaluation

The primary split and validation folds group records by learner.
This prevents different enrolments belonging to one learner from appearing
on both sides of a training/evaluation split.

The primary split contains 21,973 training enrolments and 5,493 test
enrolments, with no learner overlap.

Five validation folds within the training partition are shared across models.
Preprocessing is fitted within each training fold.

## Models and selection

The candidate models are logistic regression, Explainable Boosting Machines
(EBM), and XGBoost. A majority-class classifier provides a reference baseline.

Unweighted and balanced training are compared using training validation.
Balanced sample weights are calculated from the fitting partition only.

Mean validation positive-class F1 determines the selected weighting settings
and primary model. Logistic regression is selected before test evaluation.

F1 balances precision and recall without assigning an unsupported numerical
cost to either error type. Recall and precision are also reported separately.
Thresholds are not tuned.

## Test evaluation and additional analyses

Fixed model configurations are evaluated on the reserved primary test set.
Test results are not used to select weighting settings or model parameters.

Saved test predictions support confusion matrices, ROC curves, and paired
learner-level bootstrap comparisons.

Permutation importance describes reliance on predictors; it does not establish
causal effects or the direction of an association. Test-set importance is not
used to select new features or retune the models.

Five-seed analysis assesses sensitivity to splitting and fitting randomness
with the selected settings held fixed. It is not independent confirmation
of model selection or a measurement of explanation stability.

## Reproducibility

The project uses a Python virtual environment and `requirements.txt`.
Miniconda is not required.

The workflow is coordinated by `run.py`.
Actual study outputs are stored in `results/oulad/`.

Configuration and environment records are stored in:

- `results/oulad/experimental_settings.json`
- `results/oulad/environment_freeze.txt`

These replace the template's `results/run_config.json`.
The environment record was captured after analysis.

## Ethics and interpretation

The HRCOS82 class ethics clearance reference is 9396.
Data handling follows applicable clearance conditions and institutional
requirements.

Raw data, processed learner records, individual predictions, and fitted models
are excluded from version control.

A false positive could result in unnecessary outreach or labelling.
A false negative could mean a learner needing support is missed.
Predictions should therefore support human review rather than punitive or
automatic decisions.

Educator comprehension and intervention effectiveness were not evaluated.

## Template attribution

This repository is adapted from the CAIDS Reference Pipeline by
Themba Masombuka, Centre for Augmented Intelligence and Data Sciences,
University of South Africa.

The original template's licence and attribution must be retained.
