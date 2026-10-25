"""
Loading and generating event data.

THE CANONICAL SCHEMA
--------------------
Everything in this pipeline speaks one format, whatever the domain:

    events   : entity_id | timestamp | event_type | value
    labels   : entity_id | label

An "entity" is whatever you are predicting about.
  * e-learning : the entity is a student, the events are LMS interactions,
                 the label is (for example) whether they dropped out.
  * DevOps     : the entity is a release, the events are pipeline activities,
                 the label is whether the deployment failed.

That is the whole trick. Once both domains are expressed in this schema the
same feature extraction, the same splitting rule, the same models and the same
evaluation apply to both. If your data does not look like this, write a small
adapter that makes it look like this - do not modify the rest of the pipeline.

`value` is optional and may be NaN. Use it for a numeric payload attached to an
event (a quiz score, a build duration, a number of changed lines).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EVENT_COLUMNS = ["entity_id", "timestamp", "event_type", "value"]
LABEL_COLUMNS = ["entity_id", "label"]


# --------------------------------------------------------------------------
# Loading real data
# --------------------------------------------------------------------------

def load_events(path: str) -> pd.DataFrame:
    """Read an event table from CSV and check it against the canonical schema."""
    events = pd.read_csv(path, parse_dates=["timestamp"])
    return validate_events(events)


def load_labels(path: str) -> pd.DataFrame:
    """Read a label table from CSV and check it against the canonical schema."""
    labels = pd.read_csv(path)
    return validate_labels(labels)


def validate_events(events: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in EVENT_COLUMNS if c not in events.columns]
    if missing:
        raise ValueError(
            f"Event table is missing required column(s): {missing}. "
            f"Expected {EVENT_COLUMNS}. Write an adapter rather than changing the pipeline."
        )
    events = events[EVENT_COLUMNS].copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    events["value"] = pd.to_numeric(events["value"], errors="coerce")
    return events.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)


def validate_labels(labels: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in LABEL_COLUMNS if c not in labels.columns]
    if missing:
        raise ValueError(f"Label table is missing required column(s): {missing}.")
    labels = labels[LABEL_COLUMNS].copy()
    if not set(labels["label"].unique()) <= {0, 1}:
        raise ValueError("`label` must be binary, coded 0 and 1.")
    return labels.reset_index(drop=True)


# --------------------------------------------------------------------------
# Synthetic data
# --------------------------------------------------------------------------
# The repository ships with generators so that the pipeline runs the moment it
# is cloned, with no data and no ethics clearance required. Use them to learn
# the pipeline and to test changes. Never report a synthetic result as a
# finding.

def make_synthetic(domain: str = "elearning", n_entities: int = 900,
                   positive_rate: float = 0.18, days: int = 84,
                   seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a synthetic event log and labels in the canonical schema.

    Parameters
    ----------
    domain : "elearning" or "devops"
    n_entities : number of students / releases
    positive_rate : proportion of the positive class (dropouts / failures).
        Deliberately low, so that the accuracy trap is visible from day one.
    days : length of the observation window
    seed : random seed, so the data is reproducible
    """
    rng = np.random.default_rng(seed)

    if domain == "elearning":
        types = ["login", "view_resource", "forum_post", "quiz_attempt", "submit"]
        weights_neg = np.array([0.34, 0.38, 0.06, 0.14, 0.08])
        weights_pos = np.array([0.46, 0.34, 0.03, 0.13, 0.04])
        base_rate_neg, base_rate_pos = 3.1, 1.4
        prefix = "student"
    elif domain == "devops":
        types = ["commit", "build", "test_fail", "config_change", "deploy"]
        weights_neg = np.array([0.42, 0.30, 0.07, 0.13, 0.08])
        weights_pos = np.array([0.39, 0.27, 0.12, 0.15, 0.07])
        base_rate_neg, base_rate_pos = 2.6, 3.0
        prefix = "release"
    else:
        raise ValueError("domain must be 'elearning' or 'devops'")

    n_pos = int(round(n_entities * positive_rate))
    is_positive = np.zeros(n_entities, dtype=int)
    is_positive[:n_pos] = 1
    rng.shuffle(is_positive)

    start = pd.Timestamp("2026-02-02")
    rows = []

    for i in range(n_entities):
        entity = f"{prefix}_{i:04d}"
        pos = bool(is_positive[i])
        weights = weights_pos if pos else weights_neg
        base_rate = base_rate_pos if pos else base_rate_neg

        # Individual variation, so the classes overlap and the problem is not trivial.
        intensity = base_rate * rng.lognormal(mean=0.0, sigma=0.55)

        for day in range(days):
            # Positives decay across the window; negatives stay roughly steady.
            if pos:
                decay = np.exp(-2.2 * (day / days) ** 1.4)
            else:
                decay = 1.0 - 0.12 * (day / days)
            lam = max(intensity * decay, 0.02)
            n_events = rng.poisson(lam)
            if n_events == 0:
                continue
            kinds = rng.choice(types, size=n_events, p=weights)
            for kind in kinds:
                ts = start + pd.Timedelta(days=day, seconds=int(rng.integers(0, 86400)))
                if kind in ("quiz_attempt", "submit"):
                    value = float(np.clip(rng.normal(58 if pos else 68, 16), 0, 100))
                elif kind in ("build", "deploy"):
                    value = float(max(rng.normal(285 if pos else 250, 110), 5))
                elif kind == "config_change":
                    value = float(max(rng.normal(38 if pos else 26, 30), 1))
                else:
                    value = np.nan
                rows.append((entity, ts, kind, value))

    events = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    labels = pd.DataFrame({
        "entity_id": [f"{prefix}_{i:04d}" for i in range(n_entities)],
        "label": is_positive,
    })

    # Entities with no events at all would otherwise vanish; keep them with a
    # single marker event so the feature builder still sees them.
    silent = set(labels["entity_id"]) - set(events["entity_id"])
    if silent:
        extra = pd.DataFrame([(e, start, types[0], np.nan) for e in sorted(silent)],
                             columns=EVENT_COLUMNS)
        events = pd.concat([events, extra], ignore_index=True)

    return validate_events(events), validate_labels(labels)
