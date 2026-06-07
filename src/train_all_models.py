import pandas as pd
import numpy as np
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def train_and_save():
    # 1. Load Dataset
    data_path = "data/final/final_model_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run pipeline first.")
        
    df = pd.read_csv(data_path)
    print("Dataset Shape:", df.shape)

    # 2. Features & Target
    X = df.drop(columns=["failure_flag", "machineID"])
    y = df["failure_flag"]

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Calculate class imbalance ratio for XGBoost scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.4f}")

    # 4. Define Models
    models = {
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            class_weight="balanced",
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        ),
        "XGBoost": XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            n_estimators=100,
            max_depth=6,
            random_state=42,
            n_jobs=-1
        )
    }

    # Output directory
    os.makedirs("models", exist_ok=True)
    metrics_log = {}

    # 5. Train & Evaluate each model
    for name, clf in models.items():
        print(f"\nTraining {name}...")
        clf.fit(X_train, y_train)
        print(f"Training complete for {name}.")
        
        # Save model pickle
        prefix = name.lower().replace(" ", "_")
        model_filename = f"models/downtime_risk_model_{prefix}.pkl"
        joblib.dump(clf, model_filename)
        print(f"Saved: {model_filename}")
        
        # Evaluate
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")
        
        # Generate classification report as a dictionary
        report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        # Format metrics log structure
        formatted_report = {
            "Healthy (Class 0)": {
                "precision": float(report_dict["0.0"]["precision"]),
                "recall": float(report_dict["0.0"]["recall"]),
                "f1-score": float(report_dict["0.0"]["f1-score"]),
                "support": int(report_dict["0.0"]["support"])
            },
            "Failure (Class 1)": {
                "precision": float(report_dict["1.0"]["precision"]),
                "recall": float(report_dict["1.0"]["recall"]),
                "f1-score": float(report_dict["1.0"]["f1-score"]),
                "support": int(report_dict["1.0"]["support"])
            },
            "Macro Average": {
                "precision": float(report_dict["macro avg"]["precision"]),
                "recall": float(report_dict["macro avg"]["recall"]),
                "f1-score": float(report_dict["macro avg"]["f1-score"]),
                "support": int(report_dict["macro avg"]["support"])
            },
            "Weighted Average": {
                "precision": float(report_dict["weighted avg"]["precision"]),
                "recall": float(report_dict["weighted avg"]["recall"]),
                "f1-score": float(report_dict["weighted avg"]["f1-score"]),
                "support": int(report_dict["weighted avg"]["support"])
            }
        }
        
        metrics_log[name] = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": [[int(val) for val in row] for row in cm],
            "classification_report": formatted_report
        }

    # 6. Save metrics log to JSON
    json_path = "models/model_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics_log, f, indent=4)
    print(f"\nAll model metrics saved to: {json_path}")

if __name__ == "__main__":
    train_and_save()
