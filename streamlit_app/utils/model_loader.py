import os
import joblib
import streamlit as st

@st.cache_resource
def load_downtime_model(model_name="Random Forest"):
    """
    Loads and caches the trained model based on model_name.
    Supported: "Random Forest", "LightGBM", "XGBoost"
    """
    prefix = model_name.lower().replace(" ", "_")
    filename = f"downtime_risk_model_{prefix}.pkl"
    
    # Find project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
    model_path = os.path.join(project_root, "models", filename)
    
    if not os.path.exists(model_path):
        # Fallback to current working directory model path
        model_path = os.path.join("models", filename)
        
    # Fallback to standard model name if the segmented model file doesn't exist
    if not os.path.exists(model_path):
        model_path = os.path.join(project_root, "models", "downtime_risk_model.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join("models", "downtime_risk_model.pkl")
        
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found. Checked: {model_path}")
        
    model = joblib.load(model_path)
    return model
