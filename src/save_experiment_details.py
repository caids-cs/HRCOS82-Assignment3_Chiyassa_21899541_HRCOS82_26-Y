from pathlib import Path
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform
import subprocess
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results" / "oulad"

with open(
    RESULTS / "selected_settings.json",
    encoding="utf-8",
) as file:
    selected = json.load(file)

data = pd.read_csv(
    PROCESSED / "oulad_30day_activity_verified.csv"
)

splits = pd.read_csv(
    PROCESSED / "enrolment_splits_seed42.csv"
)

# Record the versions installed in the active environment.
packages = {}

for package in [
    "pandas",
    "numpy",
    "scikit-learn",
    "scipy",
    "matplotlib",
    "interpret-core",
    "xgboost",
    "joblib",
]:
    try:
        packages[package] = version(package)
    except PackageNotFoundError:
        packages[package] = "not installed"

# Read settings from the actual saved fitted models.
model_parameters = {}

for name in selected["balanced_by_model"]:
    model = joblib.load(
        PROCESSED / "fitted_models" / f"{name}.joblib"
    )

    model_parameters[name] = (
        model.named_steps["clf"].get_params()
    )

# A hash is a fingerprint: changed file contents produce a new hash.
def file_hash(path):
    fingerprint = hashlib.sha256()

    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            fingerprint.update(block)

    return fingerprint.hexdigest()

files_to_record = [
    ROOT / "data" / "raw" / "studentInfo.csv",
    ROOT / "data" / "raw" / "studentRegistration.csv",
    ROOT / "data" / "raw" / "studentVle.csv",
    PROCESSED / "oulad_30day_activity_verified.csv",
    PROCESSED / "enrolment_splits_seed42.csv",
]

# Also fingerprint the current Python scripts.
files_to_record.extend(sorted((ROOT / "src").glob("*.py")))

fingerprints = {
    path.relative_to(ROOT).as_posix(): file_hash(path)
    for path in files_to_record
}

partition_counts = {}

for partition in ["train", "test"]:
    records = splits.loc[splits["split"].eq(partition)]

    partition_counts[partition] = {
        "enrolments": len(records),
        "learners": int(records["id_student"].nunique()),
    }

details = {
    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    "record_note": (
        "Environment and files recorded after analysis. "
        "This timestamp is not the original experiment start time."
    ),
    "python_version": platform.python_version(),
    "operating_system": platform.platform(),
    "package_versions": packages,
    "dataset": "Open University Learning Analytics Dataset",
    "eligible_enrolments": len(data),
    "unique_learners": int(data["id_student"].nunique()),
    "predictor_days_inclusive": [0, 30],
    "prediction_point": "After day 30",
    "eligibility_rule": (
        "Exclude recorded withdrawal on or before day 30; "
        "retain missing withdrawal dates."
    ),
    "target": {
        "0": "Pass or Distinction",
        "1": "Fail or Withdrawn",
    },
    "assessment_features_included": False,
    "partitions": partition_counts,
    "cross_validation": (
        "Five saved StratifiedGroupKFold folds within training, "
        "grouped by id_student"
    ),
    "selected_settings": selected,
    "model_parameters": model_parameters,
    "permutation_importance": {
        "metric": "positive-class F1",
        "repeats": 10,
        "random_seed": 42,
    },
    "paired_bootstrap": {
        "sampling_unit": "test learner",
        "repetitions": 1000,
        "random_seed": 42,
        "interval": "95% percentile",
    },
    "stability_seeds": [1, 7, 13, 21, 42],
    "sha256_fingerprints": fingerprints,
}

destination = RESULTS / "experimental_settings.json"

with open(destination, "w", encoding="utf-8") as file:
    json.dump(details, file, indent=2, default=str)

# Save an exact package-version snapshot separately.
frozen = subprocess.run(
    [sys.executable, "-m", "pip", "freeze"],
    capture_output=True,
    text=True,
    check=True,
)

environment_file = RESULTS / "environment_freeze.txt"
environment_file.write_text(frozen.stdout, encoding="utf-8")

print("Python:", details["python_version"])
print("Operating system:", details["operating_system"])

print("\nPackage versions:")
for name, installed_version in packages.items():
    print(f"{name}: {installed_version}")

print("\nSaved:", destination)
print("Saved:", environment_file)