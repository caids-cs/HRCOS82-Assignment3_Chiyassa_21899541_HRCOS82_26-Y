from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

KEY = ["code_module", "code_presentation", "id_student"]

# Recognise OULAD's missing-value markers.
info = pd.read_csv(RAW / "studentInfo.csv", na_values=["?"])
registration = pd.read_csv(
    RAW / "studentRegistration.csv", na_values=["?"]
)

for column in ["date_registration", "date_unregistration"]:
    registration[column] = pd.to_numeric(
        registration[column], errors="raise"
    )

assert not info.duplicated(KEY).any()
assert not registration.duplicated(KEY).any()

data = info.merge(
    registration,
    on=KEY,
    how="left",
    validate="one_to_one",
    indicator=True,
)
assert data["_merge"].eq("both").all()
data = data.drop(columns="_merge")

# Preserve the original study's eligibility rule.
data = data.loc[
    data["date_unregistration"].isna()
    | data["date_unregistration"].gt(30)
].copy()

# Read the large activity file in manageable chunks.
daily_parts = []

for chunk in pd.read_csv(
    RAW / "studentVle.csv",
    usecols=KEY + ["date", "sum_click"],
    chunksize=500_000,
):
    early = chunk.loc[chunk["date"].between(0, 30)].copy()
    assert early["sum_click"].notna().all()
    assert early["sum_click"].ge(0).all()

    daily_parts.append(
        early.groupby(KEY + ["date"], as_index=False)["sum_click"].sum()
    )

# Combine daily totals, including days spanning chunk boundaries.
daily = (
    pd.concat(daily_parts, ignore_index=True)
    .groupby(KEY + ["date"], as_index=False)["sum_click"]
    .sum()
)

assert daily["date"].between(0, 30).all()
daily["active"] = daily["sum_click"].gt(0).astype(int)

activity = (
    daily.groupby(KEY)
    .agg(
        total_clicks=("sum_click", "sum"),
        active_days=("active", "sum"),
    )
    .reset_index()
)

data = data.merge(
    activity, on=KEY, how="left", validate="one_to_one"
)

# Retain eligible enrolments with no recorded early activity.
for column in ["total_clicks", "active_days"]:
    data[column] = data[column].fillna(0).astype(int)

assert data["active_days"].between(0, 31).all()

label_map = {
    "Pass": 0,
    "Distinction": 0,
    "Fail": 1,
    "Withdrawn": 1,
}
assert data["final_result"].isin(label_map).all()

data["final_result_binary"] = (
    data["final_result"].map(label_map).astype(int)
)

# Report eligibility details before dropping administrative dates.
print("Eligible enrolments:", len(data))
print("Unique learners:", data["id_student"].nunique())
print("Maximum active days:", data["active_days"].max())
print("No recorded early activity:", data["total_clicks"].eq(0).sum())
print("Registered after day 0:", data["date_registration"].gt(0).sum())
print("Unknown registration dates:", data["date_registration"].isna().sum())
print("\nTarget counts:")
print(data["final_result_binary"].value_counts().sort_index())

# IDs are retained for grouping, never as model predictors.
# Final outcome text and withdrawal dates must not become predictors.
data = data.drop(
    columns=["final_result", "date_registration", "date_unregistration"]
)

destination = OUT / "oulad_30day_activity_verified.csv"
data.to_csv(destination, index=False)
print("\nSaved:", destination)