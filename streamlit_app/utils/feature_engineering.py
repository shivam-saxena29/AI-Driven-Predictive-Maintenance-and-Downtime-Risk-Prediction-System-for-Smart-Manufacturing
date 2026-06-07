import pandas as pd

# The exact list of features expected by the trained model (31 columns)
MODEL_FEATURES = [
    'volt', 'rotate', 'pressure', 'vibration', 'age',
    'errorID_error1', 'errorID_error2', 'errorID_error3', 'errorID_error4', 'errorID_error5',
    'comp_comp1', 'comp_comp2', 'comp_comp3', 'comp_comp4',
    'total_error_count', 'total_maintenance_count',
    'voltage_std_24h', 'rolling_voltage_mean',
    'pressure_std_24h', 'rolling_pressure_mean',
    'vibration_std_24h', 'rolling_vibration_mean',
    'health_index', 'volt_vibration_ratio', 'error_maintenance_ratio',
    'production_load', 'current', 'energy_consumption', 'machine_stress_index',
    'model_encoded', 'age_category_encoded'
]

# Max rotate value from training set used to calculate production_load
MAX_ROTATE = 695.020984403396

def prepare_input(
    volt: float,
    rotate: float,
    pressure: float,
    vibration: float,
    age: float,
    error1: int,
    error2: int,
    error3: int,
    error4: int,
    error5: int,
    comp1: int,
    comp2: int,
    comp3: int,
    comp4: int,
    model_name: str
) -> pd.DataFrame:
    """
    Takes raw user inputs and outputs a 1-row pandas DataFrame containing all 31
    features expected by the model in the correct order, after performing feature engineering.
    """
    # 1. Simple Aggregates
    total_error_count = int(error1 + error2 + error3 + error4 + error5)
    total_maintenance_count = int(comp1 + comp2 + comp3 + comp4)

    # 2. Rolling Window Estimations for single-point prediction
    # If the current sensor values deviate from normal historical means (volt: 170.19, pressure: 100.80, vibration: 40.40),
    # we approximate the standard deviations proportionally. This keeps predictions realistic and sensitive to anomalous inputs.
    voltage_std_24h = max(1.5, abs(volt - 170.19) * 0.4)
    rolling_voltage_mean = volt
    
    pressure_std_24h = max(1.0, abs(pressure - 100.8) * 0.4)
    rolling_pressure_mean = pressure
    
    vibration_std_24h = max(1.0, abs(vibration - 40.4) * 0.4)
    rolling_vibration_mean = vibration

    # 3. Composite & Synthetic features
    health_index = (volt + pressure + vibration + rotate) / 4.0
    volt_vibration_ratio = volt / (vibration + 1e-5)
    error_maintenance_ratio = total_error_count / (total_maintenance_count + 1.0)
    
    production_load = (rotate / MAX_ROTATE) * 100.0
    current = volt / 10.0
    energy_consumption = volt * current
    machine_stress_index = pressure * vibration

    # 4. Model Encoding
    # Mapping based on alphabetically sorted training categories:
    # ['model1', 'model2', 'model3', 'model4']
    model_mapping = {
        'model1': 0.0,
        'model2': 1.0,
        'model3': 2.0,
        'model4': 3.0
    }
    # Clean the model name string just in case
    clean_model_name = str(model_name).lower().strip()
    model_encoded = model_mapping.get(clean_model_name, 0.0)

    # 5. Age Category Encoding
    # Bins: [0, 5, 10, 25] with labels ['New', 'Mid', 'Old']
    # Sorting alphabetically: ['Mid', 'New', 'Old'] -> codes [0, 1, 2]
    # 'New' (<=5) -> code 1
    # 'Mid' (5 < age <= 10) -> code 0
    # 'Old' (> 10) -> code 2
    if age <= 5.0:
        age_category_encoded = 1.0
    elif age <= 10.0:
        age_category_encoded = 0.0
    else:
        age_category_encoded = 2.0

    # 6. Construct Feature Dictionary
    feature_dict = {
        'volt': volt,
        'rotate': rotate,
        'pressure': pressure,
        'vibration': vibration,
        'age': age,
        'errorID_error1': float(error1),
        'errorID_error2': float(error2),
        'errorID_error3': float(error3),
        'errorID_error4': float(error4),
        'errorID_error5': float(error5),
        'comp_comp1': float(comp1),
        'comp_comp2': float(comp2),
        'comp_comp3': float(comp3),
        'comp_comp4': float(comp4),
        'total_error_count': float(total_error_count),
        'total_maintenance_count': float(total_maintenance_count),
        'voltage_std_24h': voltage_std_24h,
        'rolling_voltage_mean': rolling_voltage_mean,
        'pressure_std_24h': pressure_std_24h,
        'rolling_pressure_mean': rolling_pressure_mean,
        'vibration_std_24h': vibration_std_24h,
        'rolling_vibration_mean': rolling_vibration_mean,
        'health_index': health_index,
        'volt_vibration_ratio': volt_vibration_ratio,
        'error_maintenance_ratio': error_maintenance_ratio,
        'production_load': production_load,
        'current': current,
        'energy_consumption': energy_consumption,
        'machine_stress_index': machine_stress_index,
        'model_encoded': model_encoded,
        'age_category_encoded': age_category_encoded
    }

    # 7. Convert to DataFrame in exact expected column order
    df = pd.DataFrame([feature_dict])
    df = df[MODEL_FEATURES]
    return df
