"""
AI-Powered Predictive Maintenance — Data Processing Pipeline
=============================================================
Runs the full pipeline from raw CSVs to the final model dataset.

Outputs
-------
data/processed/merged_dataset.csv
data/processed/cleaned_dataset.csv
data/processed/engineered_dataset.csv
data/final/final_model_dataset.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
PROC_DIR = os.path.join(BASE_DIR, "data", "processed")
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

os.makedirs(PROC_DIR,  exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

# Force UTF-8 output to avoid cp1252 errors on Windows
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — merge_dataset  →  merged_dataset.csv + cleaned_dataset.csv
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 1 : Loading raw CSVs")

telemetry = pd.read_csv(os.path.join(RAW_DIR, "PdM_telemetry.csv"))
errors    = pd.read_csv(os.path.join(RAW_DIR, "PdM_errors.csv"))
maint     = pd.read_csv(os.path.join(RAW_DIR, "PdM_maint.csv"))
failures  = pd.read_csv(os.path.join(RAW_DIR, "PdM_failures.csv"))
machines  = pd.read_csv(os.path.join(RAW_DIR, "PdM_machines.csv"))

print(f"  telemetry : {telemetry.shape}")
print(f"  errors    : {errors.shape}")
print(f"  maint     : {maint.shape}")
print(f"  failures  : {failures.shape}")
print(f"  machines  : {machines.shape}")

# ── Datetime conversion ───────────────────────────────────────────────────────
section("STEP 1.1 : Datetime conversion")

telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])
errors["datetime"]    = pd.to_datetime(errors["datetime"])
maint["datetime"]     = pd.to_datetime(maint["datetime"])
failures["datetime"]  = pd.to_datetime(failures["datetime"])
print("  Done.")

# ── Remove duplicates ─────────────────────────────────────────────────────────
section("STEP 1.2 : Remove duplicates")

telemetry.drop_duplicates(inplace=True)
errors.drop_duplicates(inplace=True)
maint.drop_duplicates(inplace=True)
failures.drop_duplicates(inplace=True)
machines.drop_duplicates(inplace=True)
print("  Done.")

# ── Build merged dataset ──────────────────────────────────────────────────────
section("STEP 1.3 : Building merged dataset")

merged_df = telemetry.copy()

# Merge machines
merged_df = merged_df.merge(machines, on="machineID", how="left")

# Encode + aggregate errors
errors_encoded = pd.get_dummies(errors, columns=["errorID"])
errors_encoded = (
    errors_encoded
    .groupby(["machineID", "datetime"], as_index=False)
    .sum()
)

# Encode + aggregate maintenance
maint_encoded = pd.get_dummies(maint, columns=["comp"])
maint_encoded = (
    maint_encoded
    .groupby(["machineID", "datetime"], as_index=False)
    .sum()
)

# Failure label
failures["failure_flag"] = 1
failure_df = (
    failures[["machineID", "datetime", "failure_flag"]]
    .groupby(["machineID", "datetime"], as_index=False)
    .max()
)

# Merge all
merged_df = merged_df.merge(errors_encoded,  on=["machineID", "datetime"], how="left")
merged_df = merged_df.merge(maint_encoded,   on=["machineID", "datetime"], how="left")
merged_df = merged_df.merge(failure_df,      on=["machineID", "datetime"], how="left")

# Fill NaNs (no event = 0)
merged_df.fillna(0, inplace=True)

print(f"  Merged shape    : {merged_df.shape}")
print(f"  Duplicate rows  : {merged_df.duplicated().sum()}")
print(f"  Missing values  : {merged_df.isnull().sum().sum()}")

# ── Save merged_dataset.csv ───────────────────────────────────────────────────
merged_path = os.path.join(PROC_DIR, "merged_dataset.csv")
merged_df.to_csv(merged_path, index=False)
print(f"\n  ✓ Saved → {merged_path}")

# ── cleaned_dataset.csv  (merged + sorted + reset index) ─────────────────────
section("STEP 1.4 : Saving cleaned_dataset.csv")

cleaned_df = merged_df.sort_values(["machineID", "datetime"]).reset_index(drop=True)
cleaned_path = os.path.join(PROC_DIR, "cleaned_dataset.csv")
cleaned_df.to_csv(cleaned_path, index=False)
print(f"  Cleaned shape   : {cleaned_df.shape}")
print(f"  ✓ Saved → {cleaned_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — feature_engineering  →  engineered_dataset.csv
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 2 : Feature Engineering")

df = cleaned_df.copy()

# ── Error & maintenance aggregates ───────────────────────────────────────────
error_cols = ["errorID_error1", "errorID_error2", "errorID_error3",
              "errorID_error4", "errorID_error5"]
maint_cols = ["comp_comp1", "comp_comp2", "comp_comp3", "comp_comp4"]

df["total_error_count"]       = df[error_cols].sum(axis=1)
df["total_maintenance_count"] = df[maint_cols].sum(axis=1)

# ── Age category ─────────────────────────────────────────────────────────────
df["age_category"] = pd.cut(
    df["age"],
    bins=[0, 5, 10, 25],
    labels=["New", "Mid", "Old"],
    include_lowest=True
)

# ── Rolling features (sorted by machineID + datetime already) ────────────────
print("  Computing rolling features (this may take a minute)…")

for col, alias in [("volt", "voltage"), ("pressure", "pressure"), ("vibration", "vibration")]:
    df[f"{alias}_std_24h"] = (
        df.groupby("machineID")[col]
        .rolling(window=24, min_periods=1)
        .std()
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    df[f"rolling_{alias}_mean"] = (
        df.groupby("machineID")[col]
        .rolling(window=24, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

# ── Composite features ────────────────────────────────────────────────────────
df["health_index"]           = (df["volt"] + df["pressure"] + df["vibration"] + df["rotate"]) / 4
df["volt_vibration_ratio"]   = df["volt"] / (df["vibration"] + 1e-5)
df["error_maintenance_ratio"]= df["total_error_count"] / (df["total_maintenance_count"] + 1)

# ── Synthetic features ────────────────────────────────────────────────────────
df["production_load"]       = (df["rotate"] / df["rotate"].max()) * 100
df["current"]               = df["volt"] / 10
df["energy_consumption"]    = df["volt"] * df["current"]
df["machine_stress_index"]  = df["pressure"] * df["vibration"]

print(f"  Engineered shape  : {df.shape}")
print(f"  Total features    : {len(df.columns)}")
print(f"  Missing values    : {df.isnull().sum().sum()}")

# ── Save engineered_dataset.csv ───────────────────────────────────────────────
eng_path = os.path.join(PROC_DIR, "engineered_dataset.csv")
df.to_csv(eng_path, index=False)
print(f"\n  ✓ Saved → {eng_path}")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — final_model_dataset.csv
#   Drop non-numeric/non-model columns, keep only model-ready features
# ══════════════════════════════════════════════════════════════════════════════
section("STEP 3 : Building final_model_dataset.csv")

# Encode model column (categorical → numeric)
df["model_encoded"] = df["model"].astype("category").cat.codes

# Encode age_category
df["age_category_encoded"] = df["age_category"].astype("category").cat.codes

# Drop columns that are NOT needed for modelling
drop_cols = ["datetime", "model", "age_category"]
final_df = df.drop(columns=drop_cols, errors="ignore")

# Ensure no missing values remain
final_df = final_df.fillna(0)

print(f"  Final model shape : {final_df.shape}")
print(f"  Missing values    : {final_df.isnull().sum().sum()}")
print(f"  Columns           :")
for c in final_df.columns:
    print(f"    {c}")

final_path = os.path.join(FINAL_DIR, "final_model_dataset.csv")
final_df.to_csv(final_path, index=False)
print(f"\n  ✓ Saved → {final_path}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
section("PIPELINE COMPLETE — Output files")

outputs = {
    "data/processed/merged_dataset.csv"    : merged_path,
    "data/processed/cleaned_dataset.csv"   : cleaned_path,
    "data/processed/engineered_dataset.csv": eng_path,
    "data/final/final_model_dataset.csv"   : final_path,
}

all_ok = True
for label, path in outputs.items():
    exists = os.path.isfile(path)
    size   = f"{os.path.getsize(path) / 1024 / 1024:.1f} MB" if exists else "MISSING"
    status = "✓" if exists else "✗"
    print(f"  {status}  {label:45s}  {size}")
    if not exists:
        all_ok = False

print()
if all_ok:
    print("  ✅  All output files generated successfully!")
else:
    print("  ❌  Some files are missing — check errors above.")
    sys.exit(1)
