"""
Turning an event log into one row per entity.

THE OBSERVATION WINDOW
----------------------
The single most important idea in this file is the *cutoff*. A prediction is
only useful if it can be made early enough to act on. So features are built
from events that occurred BEFORE the cutoff, and nothing after it.

If you build features from the whole window and then predict an outcome that
happened inside that window, you have leaked the future into the past. The
model will look excellent and be worthless. This is the most common serious
error in student dissertations in this area, and the cutoff is how you avoid it.

Set `cutoff_fraction=0.3` to answer "can we predict from the first 30% of the
period?" - which is the question an institution actually cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def apply_cutoff(events: pd.DataFrame, cutoff_fraction: float = 0.3) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Keep only events before the cutoff point of the observation window."""
    if not 0 < cutoff_fraction <= 1:
        raise ValueError("cutoff_fraction must be in (0, 1].")
    t0, t1 = events["timestamp"].min(), events["timestamp"].max()
    cutoff = t0 + (t1 - t0) * cutoff_fraction
    return events[events["timestamp"] <= cutoff].copy(), cutoff


def build_features(events: pd.DataFrame, cutoff_fraction: float = 0.3,
                   top_k_types: int = 8) -> pd.DataFrame:
    """Build one feature row per entity from the pre-cutoff events.

    The features are deliberately generic - they describe the *shape* of an
    event stream, not the meaning of any particular event. That is what lets
    the same code serve an LMS clickstream and a delivery pipeline log.

    Returns a DataFrame indexed by entity_id.
    """
    windowed, cutoff = apply_cutoff(events, cutoff_fraction)
    if windowed.empty:
        raise ValueError("No events fall before the cutoff. Increase cutoff_fraction.")

    t0 = windowed["timestamp"].min()
    span_hours = max((cutoff - t0).total_seconds() / 3600.0, 1.0)

    g = windowed.groupby("entity_id")

    feats = pd.DataFrame(index=g.size().index)
    feats.index.name = "entity_id"

    # --- volume -----------------------------------------------------------
    feats["n_events"] = g.size()
    feats["events_per_day"] = feats["n_events"] / (span_hours / 24.0)
    feats["n_active_days"] = g["timestamp"].apply(lambda s: s.dt.normalize().nunique())
    feats["n_event_types"] = g["event_type"].nunique()

    # --- recency and regularity ------------------------------------------
    last = g["timestamp"].max()
    first = g["timestamp"].min()
    feats["hours_since_last_event"] = (cutoff - last).dt.total_seconds() / 3600.0
    feats["hours_to_first_event"] = (first - t0).dt.total_seconds() / 3600.0
    feats["active_span_hours"] = (last - first).dt.total_seconds() / 3600.0

    def gap_stats(s: pd.Series) -> pd.Series:
        gaps = s.sort_values().diff().dt.total_seconds().dropna() / 3600.0
        if len(gaps) == 0:
            return pd.Series({"gap_mean_h": np.nan, "gap_std_h": np.nan, "gap_max_h": np.nan})
        return pd.Series({"gap_mean_h": gaps.mean(), "gap_std_h": gaps.std(),
                          "gap_max_h": gaps.max()})

    feats = feats.join(g["timestamp"].apply(gap_stats).unstack())

    # --- trend: is activity rising or falling across the window? ----------
    half = t0 + (cutoff - t0) / 2
    early = windowed[windowed["timestamp"] <= half].groupby("entity_id").size()
    late = windowed[windowed["timestamp"] > half].groupby("entity_id").size()
    early = early.reindex(feats.index, fill_value=0)
    late = late.reindex(feats.index, fill_value=0)
    feats["events_early_half"] = early
    feats["events_late_half"] = late
    feats["activity_trend"] = (late - early) / (late + early).replace(0, np.nan)

    # --- per-event-type counts and shares ---------------------------------
    top_types = windowed["event_type"].value_counts().head(top_k_types).index
    counts = (windowed[windowed["event_type"].isin(top_types)]
              .pivot_table(index="entity_id", columns="event_type",
                           values="timestamp", aggfunc="size")
              .reindex(feats.index).fillna(0))
    counts.columns = [f"n_{c}" for c in counts.columns]
    feats = feats.join(counts)
    for c in counts.columns:
        feats[c.replace("n_", "share_", 1)] = feats[c] / feats["n_events"].replace(0, np.nan)

    # --- numeric payload --------------------------------------------------
    if windowed["value"].notna().any():
        v = windowed.dropna(subset=["value"]).groupby("entity_id")["value"]
        feats["value_mean"] = v.mean().reindex(feats.index)
        feats["value_std"] = v.std().reindex(feats.index)
        feats["value_min"] = v.min().reindex(feats.index)
        feats["value_max"] = v.max().reindex(feats.index)
        feats["value_n"] = v.size().reindex(feats.index).fillna(0)

    return feats.replace([np.inf, -np.inf], np.nan)


def align(features: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Line features up with labels, keeping only entities present in both."""
    merged = features.join(labels.set_index("entity_id"), how="inner")
    y = merged.pop("label").astype(int)
    return merged, y
