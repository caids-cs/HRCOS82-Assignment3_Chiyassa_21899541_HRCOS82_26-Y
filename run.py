from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("Prepare verified features", "prepare_oulad.py"),
    ("Create learner-separated splits", "prepare_splits.py"),
    ("Validate data and splits", "check_training_data.py"),
    ("Run unweighted validation", "train_models.py"),
    ("Run balanced validation", "train_models_balanced.py"),
    ("Select training settings", "select_models.py"),
    ("Evaluate final models", "evaluate_models.py"),
    ("Generate predictive graphs", "plot_test_results.py"),
    ("Calculate permutation importance", "explain_models.py"),
    ("Calculate bootstrap intervals", "bootstrap_comparison.py"),
    ("Check five-seed stability", "check_model_stability.py"),
    ("Record experimental details", "save_experiment_details.py"),
]

RAW_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentVle.csv",
]


def main():
    parser = argparse.ArgumentParser(
        description="Run the OULAD model comparison."
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check required files without running the analysis.",
    )

    args = parser.parse_args()

    required_files = [
        ROOT / "data" / "raw" / name
        for name in RAW_FILES
    ]

    required_files += [
        ROOT / "src" / filename
        for _, filename in STEPS
    ]

    required_files.append(ROOT / "src" / "modelling.py")

    missing = [
        path for path in required_files
        if not path.is_file()
    ]

    if missing:
        print("Missing required files:")

        for path in missing:
            print(" -", path.relative_to(ROOT))

        return 1

    print("All required source files and scripts are present.")

    if args.check:
        print("File check passed. No analysis was run.")
        return 0

    # Protect completed results before any preparation step runs.
    existing_results = ROOT / "results" / "oulad"

    if existing_results.exists() and any(existing_results.iterdir()):
        print(
            "\nExisting study outputs were found in results/oulad.\n"
            "No scripts were run.\n"
            "Archive the existing outputs before a full rerun.\n"
            "Use python run.py --check for a file check only."
        )
        return 1

    for number, (description, filename) in enumerate(STEPS, start=1):
        print(
            f"\nSTEP {number}/{len(STEPS)}: {description}",
            flush=True,
        )

        result = subprocess.run(
            [
                sys.executable,
                "-u",
                str(ROOT / "src" / filename),
            ],
            cwd=ROOT,
        )

        if result.returncode != 0:
            print(
                f"\nStopped because {filename} failed. "
                "Review the error above."
            )
            return result.returncode

    print("\nComplete: results are in results/oulad.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())