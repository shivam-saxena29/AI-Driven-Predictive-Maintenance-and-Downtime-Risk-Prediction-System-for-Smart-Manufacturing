import pandas as pd
import numpy as np

# Feature range dictionary computed from the training dataset
FEATURE_RANGES = {
    'volt': (97.333603782359, 255.124717259791),
    'rotate': (138.432075304341, 695.020984403396),
    'pressure': (51.2371057734253, 185.951997730866),
    'vibration': (14.877053998383, 76.7910723016723),
    'age': (0.0, 20.0),
    'errorID_error1': (0.0, 1.0),
    'errorID_error2': (0.0, 1.0),
    'errorID_error3': (0.0, 1.0),
    'errorID_error4': (0.0, 1.0),
    'errorID_error5': (0.0, 1.0),
    'comp_comp1': (0.0, 1.0),
    'comp_comp2': (0.0, 1.0),
    'comp_comp3': (0.0, 1.0),
    'comp_comp4': (0.0, 1.0),
    'total_error_count': (0.0, 3.0),
    'total_maintenance_count': (0.0, 2.0),
    'voltage_std_24h': (0.0, 50.02580934308538),
    'rolling_voltage_mean': (136.878588459334, 223.853296420621),
    'pressure_std_24h': (0.0, 28.90198713546357),
    'rolling_pressure_mean': (76.0053324677532, 153.42215706937),
    'vibration_std_24h': (0.0, 13.203534203871127),
    'rolling_vibration_mean': (22.9732894818614, 61.9321244590576),
    'health_index': (112.48132654964058, 252.59234105851775),
    'volt_vibration_ratio': (1.8836086905618137, 10.900427300800535),
    'error_maintenance_ratio': (0.0, 3.0),
    'production_load': (19.91768283416229, 100.0),
    'current': (9.7333603782359, 25.5124717259791),
    'energy_consumption': (947.3830425261252, 6508.86213568883),
    'machine_stress_index': (1440.2215948590604, 9290.37539706537),
    'model_encoded': (0.0, 3.0),
    'age_category_encoded': (0.0, 2.0)
}

# Mapping of internal feature names to user-friendly UI labels
FRIENDLY_NAMES = {
    'volt': 'Voltage (Volt)',
    'rotate': 'Rotation Speed (RPM)',
    'pressure': 'Pressure (psi)',
    'vibration': 'Vibration (mm/s)',
    'age': 'Machine Age (Years)',
    'errorID_error1': 'Error 1 Active',
    'errorID_error2': 'Error 2 Active',
    'errorID_error3': 'Error 3 Active',
    'errorID_error4': 'Error 4 Active',
    'errorID_error5': 'Error 5 Active',
    'comp_comp1': 'Component 1 Replaced',
    'comp_comp2': 'Component 2 Replaced',
    'comp_comp3': 'Component 3 Replaced',
    'comp_comp4': 'Component 4 Replaced',
    'total_error_count': 'Total Errors (24h)',
    'total_maintenance_count': 'Total Maintenance Events',
    'voltage_std_24h': 'Voltage Std Dev (24h)',
    'rolling_voltage_mean': '24h Avg Voltage',
    'pressure_std_24h': 'Pressure Std Dev (24h)',
    'rolling_pressure_mean': '24h Avg Pressure',
    'vibration_std_24h': 'Vibration Std Dev (24h)',
    'rolling_vibration_mean': '24h Avg Vibration',
    'health_index': 'Composite Health Index',
    'volt_vibration_ratio': 'Voltage/Vibration Ratio',
    'error_maintenance_ratio': 'Error/Maintenance Ratio',
    'production_load': 'Production Load (%)',
    'current': 'Current (Amp)',
    'energy_consumption': 'Energy Consumption (Watt)',
    'machine_stress_index': 'Machine Stress Index',
    'model_encoded': 'Machine Model (Encoded)',
    'age_category_encoded': 'Age Category (Encoded)'
}

def get_risk_level(prob: float) -> dict:
    """
    Classifies risk based on prediction probability.
    Returns a dictionary containing the level, badge text, styling color (hex and status),
    and a short description.
    """
    prob_percent = prob * 100.0
    if prob_percent < 30.0:
        return {
            "level": "Low",
            "badge": "Low Risk",
            "color": "#10B981",       # Emerald green
            "status": "success",
            "desc": f"The machine is operating within safe parameters. Probability of downtime is {prob_percent:.1f}%."
        }
    elif prob_percent <= 70.0:
        return {
            "level": "Medium",
            "badge": "Medium Risk",
            "color": "#F59E0B",       # Amber/Yellow
            "status": "warning",
            "desc": f"Elevated warning status. Probability of downtime is {prob_percent:.1f}%. Preventive maintenance check is advised."
        }
    else:
        return {
            "level": "High",
            "badge": "High Risk",
            "color": "#EF4444",       # Crimson/Red
            "status": "error",
            "desc": f"CRITICAL WARNING! High likelihood of machine downtime ({prob_percent:.1f}%). Immediate corrective action is required."
        }

def calculate_contributions(model, input_df: pd.DataFrame) -> list:
    """
    Calculates the contribution score of each feature for a specific prediction.
    Formula: contribution_score = feature_importance * normalized_feature_value
    where normalized values are scaled between 0 and 1 using dataset statistics.
    Returns a sorted list of dictionaries representing feature contributions.
    """
    # Handle Pipeline model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]
    else:
        estimator = model

    # Extract feature importances or coefficients
    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        # Use absolute coefficients as importance weights
        importances = np.abs(estimator.coef_[0])
        # Scale coefficients to sum to 1 to match feature_importance range
        coef_sum = np.sum(importances)
        if coef_sum > 0:
            importances = importances / coef_sum
    else:
        importances = np.ones(len(input_df.columns))

    # Get feature names
    if hasattr(model, "feature_names_in_"):
        features = model.feature_names_in_
    elif hasattr(estimator, "feature_names_in_"):
        features = estimator.feature_names_in_
    else:
        features = input_df.columns
    
    # Extract the single row of user inputs as a dictionary
    row_dict = input_df.iloc[0].to_dict()
    
    contributions = []
    for feat, imp in zip(features, importances):
        user_val = row_dict.get(feat, 0.0)
        
        # Get ranges
        min_val, max_val = FEATURE_RANGES.get(feat, (0.0, 1.0))
        
        # Calculate normalized value [0, 1]
        if max_val > min_val:
            norm_val = (user_val - min_val) / (max_val - min_val)
        else:
            norm_val = 0.0
            
        # Clamp to [0, 1]
        norm_val = max(0.0, min(1.0, norm_val))
        
        # Contribution score
        score = float(imp * norm_val)
        
        contributions.append({
            "feature": feat,
            "friendly_name": FRIENDLY_NAMES.get(feat, feat),
            "raw_value": float(user_val),
            "normalized_value": float(norm_val),
            "importance": float(imp),
            "contribution_score": score
        })
        
    # Sort contributions by score in descending order
    contributions.sort(key=lambda x: x["contribution_score"], reverse=True)
    return contributions

def generate_recommendations(risk_level: str, input_df: pd.DataFrame, top_factors: list) -> list:
    """
    Generates tailored recommendations based on risk level, raw sensor inputs, and top factors.
    Returns a list of recommendations (strings).
    """
    recs = []
    
    # 1. Base recommendations by Risk Level
    if risk_level == "Low":
        recs.append("**Regular Schedule**: Continue standard production and maintenance logs.")
        recs.append("**Routine Inspection**: No immediate actions needed. Perform next scheduled sensor inspection as planned.")
    elif risk_level == "Medium":
        recs.append("**Schedule Inspection**: Plan a physical inspection of the machine within the next 24-48 operating hours.")
        recs.append("**Targeted Maintenance**: Review the telemetry history of the top contributing features for sensor drift.")
    else:  # High Risk
        recs.append("**Emergency Shutoff/Throttle**: Immediately reduce rotation speed (RPM) or perform a controlled shutoff to prevent catastrophic failure.")
        recs.append("**Immediate Inspection Required**: Deploy a maintenance technician to diagnose the machine immediately.")

    # 2. Factor-specific recommendations
    # We inspect the top 3 factors contributing to the risk
    top_3_factors = top_factors[:3]
    for factor in top_3_factors:
        feat_name = factor["feature"]
        score = factor["contribution_score"]
        
        # Only suggest actions if the factor has a positive contribution score
        if score <= 0.0:
            continue
            
        if feat_name in ["vibration", "rolling_vibration_mean", "vibration_std_24h", "machine_stress_index"]:
            recs.append("**Vibration Warning**: High mechanical strain detected. Check shaft alignment, bearing wear, and tighten structural mounts.")
        elif feat_name in ["volt", "rolling_voltage_mean", "voltage_std_24h", "energy_consumption"]:
            recs.append("**Electrical Anomaly**: Fluctuations or high voltage levels. Inspect power supply, check circuit breakers, and test for winding short circuits.")
        elif feat_name in ["pressure", "rolling_pressure_mean", "pressure_std_24h"]:
            recs.append("**Pressure Regulation**: High system pressure. Check seals, gaskets, fluid lines, and clear potential blockages in the exhaust or filter.")
        elif feat_name in ["rotate", "production_load"]:
            recs.append("**Rotation Overload**: Speed is exceeding optimal operating thresholds. Reduce output rate to ease torque demands.")
        elif feat_name == "total_maintenance_count" or "comp_comp" in feat_name:
            recs.append("**Maintenance Closeness**: High maintenance activity or recent parts replacement. Verify that components were calibrated correctly post-service.")
        elif "errorID_error" in feat_name or feat_name == "total_error_count":
            recs.append("**Error Flags Active**: Control panel error signals are active. Check error history log and run a diagnostics scan to reset error registers.")
            
    # Deduplicate recommendations while preserving order
    unique_recs = []
    for r in recs:
        if r not in unique_recs:
            unique_recs.append(r)
            
    return unique_recs
