import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import joblib
import os

# ── Step 2 — Load Dataset ─────────────────────────────────────────────────────
df = pd.read_csv("data/final/final_model_dataset.csv")

print("Dataset Shape:", df.shape)

# ── Step 3 — Features & Target ────────────────────────────────────────────────
X = df.drop(
    columns=[
        "failure_flag",
        "machineID"
    ]
)

y = df["failure_flag"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)

# ── Step 4 — Train/Test Split ─────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\ny_train value counts:")
print(y_train.value_counts())
print("\ny_test value counts:")
print(y_test.value_counts())

# ── Step 5 — Create Decision Tree ────────────────────────────────────────────
dt_model = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# ── Step 6 — Train ────────────────────────────────────────────────────────────
print("\nTraining model...")
dt_model.fit(
    X_train,
    y_train
)
print("Training complete.")

# ── Step 7 — Prediction ───────────────────────────────────────────────────────
y_pred = dt_model.predict(X_test)

# ── Step 8 — Metrics ──────────────────────────────────────────────────────────
accuracy  = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n" + "="*40)
print("MODEL PERFORMANCE METRICS")
print("="*40)
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# ── Step 9 — Confusion Matrix ─────────────────────────────────────────────────
cm = confusion_matrix(
    y_test,
    y_pred
)

print("\n" + "="*40)
print("CONFUSION MATRIX")
print("="*40)
print(cm)

# ── Step 10 — Classification Report ──────────────────────────────────────────
print("\n" + "="*40)
print("CLASSIFICATION REPORT")
print("="*40)
print(
    classification_report(
        y_test,
        y_pred
    )
)

# ── Step 11 — Save Model ──────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(
    dt_model,
    "models/downtime_risk_model.pkl"
)

print("Model Saved --> models/downtime_risk_model.pkl")
