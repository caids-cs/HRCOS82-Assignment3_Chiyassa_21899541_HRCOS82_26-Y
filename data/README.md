# Data

**Nothing in this directory is committed.** `data/raw/` and `data/processed/` are
git-ignored, and `.gitignore` additionally blocks common data file extensions.

## Why

Research in this programme runs under ethics clearance **9396** and the Protection
of Personal Information Act (Act 4 of 2013). Learner data may be used only in
de-identified form, must be stored in the location your supervisor specifies, and
may not be disclosed to third parties. A git repository — even a private one — is
not an approved storage location.

## How to work

- Put your raw extract in `data/raw/` on your own machine. It stays there.
- Write a script that reads from `data/raw/` and writes to `data/processed/`.
- Never edit a raw file. If it needs fixing, fix it in a script.
- Record in `results/run_config.json` where the data came from and when it was
  extracted, so the analysis can be traced without the data being present.

## Expected format

Two CSV files in the canonical schema:

**events.csv**

| entity_id | timestamp | event_type | value |
|---|---|---|---|
| student_0001 | 2026-02-03 08:14:22 | login | |
| student_0001 | 2026-02-03 08:19:40 | quiz_attempt | 64 |

**labels.csv**

| entity_id | label |
|---|---|
| student_0001 | 0 |
| student_0002 | 1 |

`value` is optional and may be blank. `label` is binary, 0 or 1.

If your source data does not look like this, write an adapter in `src/` that makes
it look like this. Do not modify the pipeline.

## If you commit data by mistake

Tell your supervisor immediately. Deleting the file is not sufficient — git keeps
the history. This is treated as a data-protection incident.
