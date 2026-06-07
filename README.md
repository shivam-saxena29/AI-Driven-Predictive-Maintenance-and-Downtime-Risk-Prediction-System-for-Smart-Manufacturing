# AI-Powered Predictive Maintenance for Smart Manufacturing

An advanced, end-to-end Machine Learning and Explainable AI (XAI) dashboard for predicting machine failures and scheduling proactive maintenance in smart manufacturing. Developed with ensemble models (Random Forest, LightGBM, XGBoost) and SHAP diagnostics to provide detailed, feature-level insights into asset risks.

---

## 🌟 Key Features

- **Fleet Risk Rankings:** Real-time health monitoring and severity classification across a fleet of 100 industrial machines.
- **Explainable AI (XAI):** Global and local feature importance via SHAP values, explaining exactly *why* a machine is predicted at risk.
- **Sensor Health Metrics:** Sub-component stress monitoring and live warnings based on rolling statistics.
- **Batch CSV Predictions:** Upload custom sensor telemetry logs (CSV format) to calculate downtime risk for multiple machines simultaneously and download predictions.
- **End-to-End Data Pipeline:** Streamlined processing from raw sensor telemetry, error logs, and maintenance history to final model-ready features.
- **Interactive Control Center:** Modern, responsive UI built with Streamlit and styled with vanilla CSS.


---

## 📂 Project Structure

```text
├── .gitignore               # Configured Git ignore paths (data CSVs, model PKLs, venv)
├── .streamlit/
│   └── config.toml          # Custom theme configuration for Streamlit
├── README.md                # Project documentation (this file)
├── data/
│   ├── raw/                 # Raw datasets (PdM_telemetry.csv, PdM_errors.csv, etc.)
│   ├── processed/           # Processed datasets (merged, cleaned, engineered)
│   └── final/               # Model-ready dataset (final_model_dataset.csv)
├── deployment/              # Deployment scripts and configuration
├── docs/                    # Additional project documentation
├── models/                  # Saved models & evaluations (downtime_risk_model_*.pkl, model_metrics.json)
├── notebooks/               # Jupyter Notebooks for exploratory data analysis (EDA) & prototyping
├── reports/                 # Output charts & generated reports
├── requirements.txt         # Project dependencies
├── run_pipeline.py          # Data ingestion, cleaning, and feature engineering script
├── src/
│   ├── check_ranges.py      # Telemetry bounds validation checks
│   ├── model_training.py    # Training functions and evaluation loops
│   ├── shap_analysis.py     # SHAP explainability calculations
│   └── train_all_models.py  # Orchestrator to train and evaluate all ensemble models
├── streamlit_app/
│   ├── app.py               # Streamlit control center dashboard entry point
│   ├── pages/               # Individual Streamlit pages (Risk, EDA, SHAP, Metrics)
│   └── utils/               # Styling, risk scoring helpers, and custom components
└── visualizations/          # Saved shap charts and feature importance CSVs
```

---

## 🔧 Installation & Setup

Follow these steps to run the application locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd "AI powered predictive maintainence for smart manufacturing"
```

### 2. Set Up a Virtual Environment
We recommend using a Python 3.10+ virtual environment:

**On Windows:**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📊 Dataset Ingestion

The models are trained using the **Microsoft Azure Predictive Maintenance Dataset** (representing 1 year of telemetry, errors, failures, and maintenance logs for 100 machines).

1. Download the raw CSV dataset from [Kaggle's Azure Predictive Maintenance Dataset](https://www.kaggle.com/datasets/arnabbiswas95/microsoft-azure-predictive-maintenance).
2. Extract the downloaded files and place the raw CSVs into the `data/raw/` directory:
   - `PdM_telemetry.csv`
   - `PdM_errors.csv`
   - `PdM_failures.csv`
   - `PdM_machines.csv`
   - `PdM_maint.csv`

---

## 🚀 Running the Project

### Step 1: Run the Processing Pipeline
Process the raw datasets, perform feature engineering, and output the model-ready dataset:
```bash
python run_pipeline.py
```
This script creates the merged, cleaned, and engineered datasets inside `data/processed/` and `data/final/`.

### Step 2: Train the Ensemble Models
Train the Random Forest, XGBoost, and LightGBM models, evaluate their performance, and save the binary classifiers:
```bash
python src/train_all_models.py
```
The trained models and performance statistics will be saved to the `models/` directory.

### Step 3: Launch the Streamlit Dashboard
Run the interactive Control Center dashboard:
```bash
streamlit run streamlit_app/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to view the interface.

---

## ⚙️ Ensemble Models Performance

| Model | Accuracy | Precision | Recall (Class 1) | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Random Forest** | 99.88% | 41.37% | **96.53%** | 57.92% |
| **LightGBM** | 99.96% | 69.63% | **92.36%** | 79.40% |
| **XGBoost** | 99.96% | 72.47% | **89.58%** | 80.12% |

> **Note:** Models are optimized with recall priority to minimize false negatives (unpredicted failures that result in costly unplanned downtime).
