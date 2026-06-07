import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

# ── Step 2 — Load Dataset ─────────────────────────────────────────────────────
df = pd.read_csv(
    "data/final/final_model_dataset.csv"
)

print("Dataset loaded:", df.shape)

# ── Step 3 — Create Features ──────────────────────────────────────────────────
X = df.drop(
    columns=[
        "failure_flag",
        "machineID"
    ]
)

print("Features shape:", X.shape)

# ── Step 4 — Load Model ───────────────────────────────────────────────────────
model = joblib.load(
    "models/downtime_risk_model.pkl"
)

print("Model loaded.")

# ── Step 5 — Sample Data ──────────────────────────────────────────────────────
X_sample = X.sample(
    n=5000,
    random_state=42
)

print("Sample shape:", X_sample.shape)

# ── Step 6 — Create Explainer ─────────────────────────────────────────────────
print("Creating SHAP explainer...")

explainer = shap.TreeExplainer(
    model
)

# ── Step 7 — Generate SHAP Values ────────────────────────────────────────────
print("Generating SHAP values...")

shap_values = explainer.shap_values(
    X_sample
)

# For binary classification — handle both old (list) and new (3D array) SHAP API
import numpy as np
if isinstance(shap_values, list):
    # Old SHAP: list of arrays, one per class
    shap_values = shap_values[1]
elif hasattr(shap_values, 'ndim') and shap_values.ndim == 3:
    # New SHAP: shape (n_samples, n_features, n_classes) — take class 1 (failure)
    shap_values = shap_values[:, :, 1]

print("SHAP values shape (class=failure):", shap_values.shape)

# ── Step 8 — Create Visualization Folder ─────────────────────────────────────
os.makedirs(
    "visualizations",
    exist_ok=True
)

# ── Step 9 — Summary Plot ────────────────────────────────────────────────────
print("Saving shap_summary.png...")

shap.summary_plot(
    shap_values,
    X_sample,
    show=False
)

plt.savefig(
    "visualizations/shap_summary.png",
    bbox_inches="tight",
    dpi=150
)

plt.close()
print("Saved: visualizations/shap_summary.png")

# ── Step 10 — Feature Importance Plot ────────────────────────────────────────
print("Saving shap_feature_importance.png...")

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

plt.savefig(
    "visualizations/shap_feature_importance.png",
    bbox_inches="tight",
    dpi=150
)

plt.close()
print("Saved: visualizations/shap_feature_importance.png")

# ── Step 11 — Top Features ────────────────────────────────────────────────────
importance = np.abs(
    shap_values
).mean(axis=0)

feature_importance = pd.DataFrame({
    "Feature": X_sample.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\n" + "="*50)
print("TOP 15 FEATURE IMPORTANCE (SHAP)")
print("="*50)
print(
    feature_importance.head(15).to_string(index=True)
)

# ── Step 12 — Save CSV ────────────────────────────────────────────────────────
feature_importance.to_csv(
    "visualizations/feature_importance.csv",
    index=False
)

print("\nSaved: visualizations/feature_importance.csv")

# ── Final Verification ────────────────────────────────────────────────────────
print("\n" + "="*50)
print("OUTPUT FILES VERIFICATION")
print("="*50)

outputs = [
    "visualizations/shap_summary.png",
    "visualizations/shap_feature_importance.png",
    "visualizations/feature_importance.csv",
]

for f in outputs:
    status = "OK" if os.path.isfile(f) else "MISSING"
    size   = f"{os.path.getsize(f)/1024:.1f} KB" if os.path.isfile(f) else ""
    print(f"  [{status}]  {f}  {size}")
