# CAIDS Reference Pipeline

A shared starting point for predictive modelling on event logs, used across the
research programme of the Centre for Augmented Intelligence and Data Sciences at
UNISA.

The same code handles two kinds of data that look different and are not:

| | entity | events | label |
|---|---|---|---|
| **E-learning** | a student | LMS interactions | dropped out / did not |
| **DevOps** | a release | pipeline activities | deployment failed / did not |

Once both are written in the canonical schema (`entity_id, timestamp, event_type, value`),
one feature builder, one splitting rule, one model set and one evaluation serve both.

---

## Run it in three minutes

You do not need any data. The repository generates its own.

```bash
conda env create -f environment.yml
conda activate caids
python run.py
```

That produces figures in `results/figures/` and tables in `results/tables/`.

Other things to try:

```bash
python run.py --domain devops          # the other synthetic domain
python run.py --cutoff 0.5             # predict from the first half of the window
python run.py --no-balance             # see what class imbalance does to recall
python run.py --events data/raw/events.csv --labels data/raw/labels.csv
```

---

## What the pipeline enforces

Three rules are built into the code rather than left to your discipline.

**Split before you fit anything.** Scalers and imputers live inside a scikit-learn
`Pipeline`, so they are fitted on the training fold only. Fitting them beforehand
inflates every number you would then report — it is the most common serious error
in this kind of work, and it is invisible in the output.

**Always run a baseline.** A majority-class classifier and a logistic regression run
every time. If your random forest cannot beat logistic regression, that is a finding
and you must report it, not bury it.

**Never trust accuracy on imbalanced data.** With an 18% positive rate, predicting
"nobody" scores 82%. The pipeline prints the majority-class accuracy next to your
model's so the comparison is unavoidable, and it reports recall, F1 and average
precision alongside.

It also re-runs under several random seeds, because one run is an anecdote.

---

## Layout

```
run.py                one command, end to end
src/data.py           the canonical schema, loaders, synthetic generators
src/features.py       event log -> one row per entity, with the cutoff rule
src/modelling.py      splitting, models, metrics, stability, importance
src/report.py         figures and tables
data/raw/             your data goes here and is NEVER committed
data/processed/       generated, also never committed
results/              figures and tables — DO commit these
docs/PROTOCOL.md      the modelling protocol every project follows
docs/STUDENT_START_HERE.md
docs/SUPERVISION.md   how this repository is used in supervision
tests/                run with `pytest`
```

---

## Data: read this before you add any

**No learner data, personal data or institutional data is ever committed to this
repository.** Not to a private one either.

Research in this programme runs under ethics clearance **9396** and the Protection
of Personal Information Act. Raw data lives in the secure location your supervisor
gives you; `data/raw/` is git-ignored so that an accidental `git add .` cannot
publish it. See `data/README.md`.

If you believe you have committed data by mistake, stop and tell your supervisor
immediately. Do not simply delete the file — git keeps the history.

---

## Using it for your own project

Do not modify `src/` to fit your data. Write a small adapter that converts your
data into the canonical schema, put it in `src/`, and leave the rest alone. That is
what makes your results comparable with the rest of the cohort — and comparability
is what turns a set of separate projects into a research programme.

Add your own models to `make_models()` in `src/modelling.py`. Keep the baselines.

---

## Citing this

If you use this pipeline in a dissertation or a paper, cite it — see `CITATION.cff`.

## Licence

MIT. See `LICENSE`.
