# Start here

You have been given your own repository inside the CAIDS organisation. Only you and
your supervisor can see it.

This page gets you from nothing to a first result. Allow an hour.

---

## Step 1 — Install the two things you need

**Python, via Miniconda.** Download it from the Anaconda site and install with the
defaults. If you already have Anaconda, you have this.

**GitHub Desktop.** Download it from desktop.github.com. It is a normal application
with buttons. You do not need the command line for this module, and if the command
line is not already comfortable for you, do not start there.

Sign in to GitHub Desktop with the account you used to accept the invitation.

---

## Step 2 — Get your copy of the repository

In GitHub Desktop: **File → Clone repository →** pick your repository from the list
→ choose where to put it on your computer → **Clone**.

You now have a folder on your machine that is linked to the one on GitHub.

---

## Step 3 — Set up the environment

Open **Anaconda Prompt** (Windows) or a terminal (Mac/Linux), move into the folder
you just cloned, and run:

```
conda env create -f environment.yml
conda activate caids
```

This takes a few minutes the first time. It installs the exact library versions
everyone in the cohort is using, which is why your results will be comparable with
theirs.

---

## Step 4 — Run it

```
python run.py
```

You should see a table of model results, and new files in `results/`.

Open `results/figures/model_comparison.png`. That is your first output.

Note what happened: **logistic regression probably beat the random forest.** That is
not a mistake in the code. It is the single most useful thing this pipeline will
teach you.

---

## Step 5 — Send your work back

In GitHub Desktop you will see the changed files listed on the left.

1. Write a short summary in the box at the bottom left — "first run on synthetic
   data", not "update"
2. Click **Commit to main**
3. Click **Push origin**

Your supervisor can now see exactly what you did and when.

Do this **at least once a week**, even when the work is unfinished. A repository
with one commit the night before submission tells your supervisor nothing, and it
tells an examiner something worse.

---

## The rules that matter

**Never put data in the repository.** No student records, no exported LMS files, no
spreadsheets of marks. `data/raw/` is blocked from committing for exactly this
reason. If you think you have committed data by accident, tell your supervisor
immediately — do not just delete the file, because the history keeps it.

**Never change `src/` to fit your data.** Write an adapter that turns your data into
the canonical schema instead. Everyone in the cohort works to the same protocol;
that is what makes your results mean something next to theirs.

**Read `docs/PROTOCOL.md`.** It is the standard your methodology chapter will be
marked against.

---

## Where to ask

Use the **Issues** tab in your own repository. Click New issue, describe what you
tried, paste the error, and your supervisor gets an email. It is better than a
WhatsApp message because it stays attached to the project and you can both find it
again in three months.

---

## If your connection is unreliable

Everything except cloning and pushing works offline. Commit as often as you like
while you have no connection — commits are local — and push when you are next online.
If cloning keeps failing, tell your supervisor rather than struggling; there is a
download-as-ZIP fallback.
