import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# Add the project root to path to resolve imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from utils.model_loader import load_downtime_model
from utils.feature_engineering import prepare_input
from utils.risk_utils import get_risk_level, calculate_contributions, generate_recommendations
from utils.styles import load_css, load_sidebar_branding

# Set page config
st.set_page_config(
    page_title="Downtime Risk Prediction | Smart Manufacturing",
    page_icon=None,
    layout="wide"
)

# Apply global styles and sidebar branding
load_css("Prediction")
load_sidebar_branding()

# Sidebar branding and page config already loaded

# Section 1: Title and Description wrapped in a custom card
st.markdown(
    """<div class="custom-card" style="margin-bottom: 25px;">
<h2 style="margin: 0; font-size: 18px; color: #1a1a2e;">Downtime Risk Forecast & Diagnostics</h2>
<p style="margin: 5px 0 0 0; color: #4b5563; font-size: 13px;">
Run real-time predictive diagnostics on machine telemetry and operational error flags.
</p>
</div>""",
    unsafe_allow_html=True
)

# Shared Predictive Model Engine selection
with st.container(border=True):
    pred_model_engine = st.selectbox(
        "Predictive Model Engine",
        options=["Random Forest", "LightGBM", "XGBoost"],
        index=0,
        help="Select the machine learning model used to predict downtime risk."
    )

st.write("")

# Tabs layout
tab1, tab2 = st.tabs(["🎯 Single Asset Diagnosis", "📂 Batch Prediction (CSV Upload)"])

with tab1:
    # Section 2: Input Controls (divided into 2 columns with card styling)
    col1, col2 = st.columns(2)

    with col1:
        with st.container(border=True):
            st.markdown("### Real-Time Sensor Inputs")
            volt = st.slider(
                "Voltage (Volt)", 
                min_value=95.0, 
                max_value=260.0, 
                value=170.0, 
                step=0.1,
                help="Current voltage reading of the machine (normal ranges: 150-220V)."
            )
            rotate = st.slider(
                "Rotation Speed (RPM)", 
                min_value=130.0, 
                max_value=700.0, 
                value=450.0, 
                step=1.0,
                help="Current rotation speed of the rotor assembly (normal ranges: 350-550 RPM)."
            )
            pressure = st.slider(
                "Pressure (psi)", 
                min_value=50.0, 
                max_value=190.0, 
                value=100.0, 
                step=0.1,
                help="Internal chamber pressure reading (normal ranges: 80-140 psi)."
            )
            vibration = st.slider(
                "Vibration (mm/s)", 
                min_value=14.0, 
                max_value=80.0, 
                value=40.0, 
                step=0.1,
                help="Structural vibration speed amplitude (normal ranges: 30-50 mm/s)."
            )
            age = st.slider(
                "Machine Age (Years)", 
                min_value=0, 
                max_value=20, 
                value=10, 
                step=1,
                help="Age of the machine since deployment."
            )

    with col2:
        with st.container(border=True):
            st.markdown("### Machine Specifications")
            model_name = st.selectbox(
                "Machine Model", 
                options=["model1", "model2", "model3", "model4"], 
                index=2,
                help="Select the specific manufacturer model code."
            )
            
            st.write("")
            st.info(
                """**Single Point Inference Strategy:**
To compute rolling features for real-time predictions when historical windows aren't available, 
the system treats the current sensor value as the rolling average and estimates standard deviations proportionally based on deviations from historical means."""
            )

    # Section 3 & 4: Error and Maintenance Aggregates
    st.write("")
    col_err, col_maint = st.columns(2)

    with col_err:
        with st.container(border=True):
            st.markdown("### Active Error Flags (Last 24 Hours)")
            error1 = st.checkbox("Error 1 (Voltage/Resistance anomaly)")
            error2 = st.checkbox("Error 2 (Rotation/Torque overload)")
            error3 = st.checkbox("Error 3 (System overheating)")
            error4 = st.checkbox("Error 4 (Pressure regulation warning)")
            error5 = st.checkbox("Error 5 (High structural vibration)")

    with col_maint:
        with st.container(border=True):
            st.markdown("### Component Maintenance (Last 24 Hours)")
            comp1 = st.checkbox("Component 1 Replaced")
            comp2 = st.checkbox("Component 2 Replaced")
            comp3 = st.checkbox("Component 3 Replaced")
            comp4 = st.checkbox("Component 4 Replaced")

    # Section 5: Predict Button
    st.write("")
    predict_clicked = st.button("Predict Downtime Risk", type="primary", use_container_width=True)

    if predict_clicked:
        if volt <= 0 or rotate <= 0 or pressure <= 0 or vibration <= 0 or age < 0:
            st.error("Please enter valid positive machine values.")
            st.stop()
            
        with st.spinner("Analyzing machine telemetry, stress indexes, and history..."):
            # 1. Prepare data
            input_df = prepare_input(
                volt=volt,
                rotate=rotate,
                pressure=pressure,
                vibration=vibration,
                age=age,
                error1=1 if error1 else 0,
                error2=1 if error2 else 0,
                error3=1 if error3 else 0,
                error4=1 if error4 else 0,
                error5=1 if error5 else 0,
                comp1=1 if comp1 else 0,
                comp2=1 if comp2 else 0,
                comp3=1 if comp3 else 0,
                comp4=1 if comp4 else 0,
                model_name=model_name
            )
            
            # 2. Load model & perform prediction
            try:
                model = load_downtime_model(pred_model_engine)
                prob = model.predict_proba(input_df)[0][1]
                prob_percent = prob * 100.0
                risk_data = get_risk_level(prob)
                contributions = calculate_contributions(model, input_df)
                recs = generate_recommendations(risk_data["level"], input_df, contributions)
                
                # Layout Results in clean card formats
                st.write("---")
                st.markdown("## Analysis Results")
                
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    with st.container(border=True):
                        st.markdown("### Risk Level")
                        
                        # Premium Risk Metric Card (Styled for white theme contrast)
                        st.markdown(f"""<div style="background-color: {risk_data['color']}12; border-left: 6px solid {risk_data['color']}; padding: 20px; border-radius: 6px; margin-bottom: 20px;">
<h2 style="color: {risk_data['color']}; margin: 0; padding-bottom: 5px; font-size: 18px;">{risk_data['badge']}</h2>
<p style="margin: 0; font-size: 13px; color: #1a1a2e; font-weight: 500;">{risk_data['desc']}</p>
</div>""", unsafe_allow_html=True)
                        
                        # Plotly Gauge Chart with dark labels for white theme
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=prob_percent,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "<b>Downtime Probability</b>", 'font': {'size': 18, 'color': '#1a1a2e'}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#4b5563"},
                                'bar': {'color': risk_data['color']},
                                'bgcolor': "rgba(0,0,0,0.03)",
                                'borderwidth': 1,
                                'bordercolor': "#cbd5e1",
                                'steps': [
                                    {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.12)'},
                                    {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.12)'},
                                    {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.12)'}
                                ]
                            }
                        ))
                        fig.update_layout(
                            height=250, 
                            margin=dict(l=20, r=20, t=40, b=20),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font={'color': '#1a1a2e'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

                with res_col2:
                    with st.container(border=True):
                        st.markdown("### Top 5 Contributing Factors")
                        
                        contrib_df = pd.DataFrame(contributions[:5])
                        contrib_df_display = contrib_df[["friendly_name", "raw_value", "contribution_score"]].copy()
                        contrib_df_display.columns = ["Factor Description", "Input Value", "Contribution Weight"]
                        contrib_df_display["Contribution Weight"] = contrib_df_display["Contribution Weight"].map(lambda x: f"{x:.4f}")
                        
                        st.dataframe(
                            contrib_df_display, 
                            use_container_width=True, 
                            hide_index=True
                        )
                        
                        st.markdown("### Recommended Actions")
                        if risk_data["level"] == "High":
                            for rec in recs:
                                st.error(rec)
                        elif risk_data["level"] == "Medium":
                            for rec in recs:
                                st.warning(rec)
                        else:
                            for rec in recs:
                                st.info(rec)

            except FileNotFoundError:
                st.error("Model not found. Please train the model first by running src/model_training.py.")
                st.warning("Prediction requires models/downtime_risk_model.pkl to be generated.")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.markdown("### 📂 Batch Telemetry Predictions")
    st.write(
        "Upload a CSV file containing raw sensor logs from one or more machines. "
        "The system will run feature engineering, predict the downtime risk for each machine record, and return the classified outcomes."
    )
    
    # Generate and provide template file
    template_data = pd.DataFrame([
        {
            "machineID": 1,
            "volt": 170.2,
            "rotate": 450.0,
            "pressure": 100.8,
            "vibration": 40.4,
            "age": 10,
            "model_name": "model3",
            "error1": 0,
            "error2": 0,
            "error3": 0,
            "error4": 0,
            "error5": 0,
            "comp1": 0,
            "comp2": 0,
            "comp3": 0,
            "comp4": 0
        },
        {
            "machineID": 2,
            "volt": 220.5,
            "rotate": 320.1,
            "pressure": 130.4,
            "vibration": 52.8,
            "age": 15,
            "model_name": "model1",
            "error1": 1,
            "error2": 0,
            "error3": 0,
            "error4": 1,
            "error5": 0,
            "comp1": 0,
            "comp2": 1,
            "comp3": 0,
            "comp4": 0
        }
    ])
    template_csv = template_data.to_csv(index=False).encode('utf-8')
    
    # Card for CSV template and uploading
    col_upload, col_info = st.columns([2, 1])
    
    with col_info:
        with st.container(border=True):
            st.markdown("##### Required Format")
            st.write(
                "The CSV must contain the following sensor reading columns:\n"
                "* `volt` (Voltage)\n"
                "* `rotate` (Rotor RPM)\n"
                "* `pressure` (Pressure psi)\n"
                "* `vibration` (Vibration mm/s)\n"
                "* `age` (Machine Age Years)\n"
                "* `model_name` (e.g. model1, model2, model3, model4)\n\n"
                "Optional flags (defaults to 0 if not present):\n"
                "* `error1` to `error5` (Active errors)\n"
                "* `comp1` to `comp4` (Component replacements)"
            )
            st.download_button(
                label="📥 Download Template CSV",
                data=template_csv,
                file_name="predictive_maintenance_template.csv",
                mime="text/csv",
                use_container_width=True
            )
            
    with col_upload:
        with st.container(border=True):
            st.markdown("##### Upload Telemetry Logs")
            uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], label_visibility="collapsed")
            
            if uploaded_file is not None:
                try:
                    input_df = pd.read_csv(uploaded_file)
                    st.success(f"✓ Uploaded successfully: {uploaded_file.name} ({len(input_df)} records)")
                    st.markdown("**Preview (First 3 rows):**")
                    st.dataframe(input_df.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")
                    input_df = None
            else:
                input_df = None

    if input_df is not None:
        required_cols = ["volt", "rotate", "pressure", "vibration", "age"]
        missing_cols = [c for c in required_cols if c not in input_df.columns]
        
        if missing_cols:
            st.error(f"Cannot run prediction. Missing required columns: {', '.join(missing_cols)}")
        else:
            st.write("")
            run_batch = st.button("🚀 Run Batch Diagnostics", type="primary", use_container_width=True)
            
            if run_batch:
                with st.spinner("Processing logs, computing rolling values, running models..."):
                    try:
                        # 1. Load active model
                        model = load_downtime_model(pred_model_engine)
                        
                        # 2. Vectorized feature engineering on the copy
                        proc_df = input_df.copy()
                        
                        # Set default values for optional missing cols
                        for col in ['error1', 'error2', 'error3', 'error4', 'error5']:
                            if col not in proc_df.columns:
                                proc_df[col] = 0
                        for col in ['comp1', 'comp2', 'comp3', 'comp4']:
                            if col not in proc_df.columns:
                                proc_df[col] = 0
                                
                        # Simple Aggregates
                        proc_df['total_error_count'] = proc_df[['error1', 'error2', 'error3', 'error4', 'error5']].sum(axis=1)
                        proc_df['total_maintenance_count'] = proc_df[['comp1', 'comp2', 'comp3', 'comp4']].sum(axis=1)
                        
                        # Rename error and component columns
                        rename_dict = {
                            'error1': 'errorID_error1', 'error2': 'errorID_error2', 'error3': 'errorID_error3',
                            'error4': 'errorID_error4', 'error5': 'errorID_error5',
                            'comp1': 'comp_comp1', 'comp2': 'comp_comp2', 'comp3': 'comp_comp3', 'comp4': 'comp_comp4'
                        }
                        # Only rename columns that exist in the dataframe
                        rename_dict = {k: v for k, v in rename_dict.items() if k in proc_df.columns}
                        proc_df = proc_df.rename(columns=rename_dict)
                        
                        # Rolling window estimations
                        proc_df['voltage_std_24h'] = (proc_df['volt'] - 170.19).abs() * 0.4
                        proc_df.loc[proc_df['voltage_std_24h'] < 1.5, 'voltage_std_24h'] = 1.5
                        proc_df['rolling_voltage_mean'] = proc_df['volt']
                        
                        proc_df['pressure_std_24h'] = (proc_df['pressure'] - 100.8).abs() * 0.4
                        proc_df.loc[proc_df['pressure_std_24h'] < 1.0, 'pressure_std_24h'] = 1.0
                        proc_df['rolling_pressure_mean'] = proc_df['pressure']
                        
                        proc_df['vibration_std_24h'] = (proc_df['vibration'] - 40.4).abs() * 0.4
                        proc_df.loc[proc_df['vibration_std_24h'] < 1.0, 'vibration_std_24h'] = 1.0
                        proc_df['rolling_vibration_mean'] = proc_df['vibration']
                        
                        # Composite features
                        proc_df['health_index'] = (proc_df['volt'] + proc_df['pressure'] + proc_df['vibration'] + proc_df['rotate']) / 4.0
                        proc_df['volt_vibration_ratio'] = proc_df['volt'] / (proc_df['vibration'] + 1e-5)
                        proc_df['error_maintenance_ratio'] = proc_df['total_error_count'] / (proc_df['total_maintenance_count'] + 1.0)
                        
                        from utils.feature_engineering import MAX_ROTATE
                        proc_df['production_load'] = (proc_df['rotate'] / MAX_ROTATE) * 100.0
                        proc_df['current'] = proc_df['volt'] / 10.0
                        proc_df['energy_consumption'] = proc_df['volt'] * proc_df['current']
                        proc_df['machine_stress_index'] = proc_df['pressure'] * proc_df['vibration']
                        
                        # Model mapping
                        if 'model' in proc_df.columns and 'model_name' not in proc_df.columns:
                            proc_df = proc_df.rename(columns={'model': 'model_name'})
                        if 'model_name' not in proc_df.columns:
                            proc_df['model_name'] = 'model3'
                            
                        model_mapping = {
                            'model1': 0.0, 'model2': 1.0, 'model3': 2.0, 'model4': 3.0
                        }
                        proc_df['model_encoded'] = proc_df['model_name'].astype(str).str.lower().str.strip().map(model_mapping).fillna(0.0)
                        
                        # Age Category Encoding
                        proc_df['age_category_encoded'] = 2.0  # Default to 'Old'
                        proc_df.loc[proc_df['age'] <= 10.0, 'age_category_encoded'] = 0.0  # 'Mid'
                        proc_df.loc[proc_df['age'] <= 5.0, 'age_category_encoded'] = 1.0   # 'New'
                        
                        from utils.feature_engineering import MODEL_FEATURES
                        # Extract exact features in correct order
                        model_input = proc_df[MODEL_FEATURES]
                        
                        # 3. Batch prediction
                        probabilities = model.predict_proba(model_input)[:, 1]
                        predictions = model.predict(model_input)
                        
                        # 4. Attach back to original df
                        output_df = input_df.copy()
                        output_df["Risk_Probability"] = np.round(probabilities, 4)
                        output_df["Downtime_Risk_Percent"] = np.round(probabilities * 100.0, 2)
                        output_df["Predicted_Class"] = predictions
                        
                        # Add Severity level
                        def get_sev(p):
                            if p < 0.3: return "Healthy"
                            elif p <= 0.7: return "Warning"
                            else: return "Critical"
                        output_df["Risk_Severity"] = [get_sev(p) for p in probabilities]
                        
                        # Display summary stats
                        st.write("---")
                        st.markdown("### Analysis Summary")
                        
                        total = len(output_df)
                        crit_cnt = sum(output_df["Risk_Severity"] == "Critical")
                        warn_cnt = sum(output_df["Risk_Severity"] == "Warning")
                        heal_cnt = sum(output_df["Risk_Severity"] == "Healthy")
                        
                        # Custom cards for summary metrics
                        sc1, sc2, sc3 = st.columns(3)
                        with sc1:
                            st.markdown(f"""<div style="background-color: #10b98112; border-left: 5px solid #10b981; padding: 15px; border-radius: 6px;">
<div style="font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase;">Healthy Assets</div>
<div style="font-size: 20px; font-weight: 700; color: #1a1a2e; margin-top: 4px;">{heal_cnt} / {total}</div>
<div style="font-size: 11px; color: #10b981; font-weight: 500;">{heal_cnt/total*100:.1f}% of uploaded fleet</div>
</div>""", unsafe_allow_html=True)
                        with sc2:
                            st.markdown(f"""<div style="background-color: #f59e0b12; border-left: 5px solid #f59e0b; padding: 15px; border-radius: 6px;">
<div style="font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase;">Warning Assets</div>
<div style="font-size: 20px; font-weight: 700; color: #1a1a2e; margin-top: 4px;">{warn_cnt} / {total}</div>
<div style="font-size: 11px; color: #f59e0b; font-weight: 500;">{warn_cnt/total*100:.1f}% of uploaded fleet</div>
</div>""", unsafe_allow_html=True)
                        with sc3:
                            st.markdown(f"""<div style="background-color: #ef444412; border-left: 5px solid #ef4444; padding: 15px; border-radius: 6px;">
<div style="font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase;">Critical Assets</div>
<div style="font-size: 20px; font-weight: 700; color: #1a1a2e; margin-top: 4px;">{crit_cnt} / {total}</div>
<div style="font-size: 11px; color: #ef4444; font-weight: 500;">{crit_cnt/total*100:.1f}% of uploaded fleet</div>
</div>""", unsafe_allow_html=True)
                            
                        # Show output preview
                        st.markdown("#### Predictions Table")
                        st.dataframe(output_df, use_container_width=True)
                        
                        # Download output button
                        output_csv = output_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions CSV",
                            data=output_csv,
                            file_name="batch_predictions_results.csv",
                            mime="text/csv",
                            type="primary",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"Error executing batch inference: {e}")

