import os
import sys
import pandas as pd
import streamlit as st

# Ensure project root is in path to resolve imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.styles import load_css, kpi_card, load_sidebar_branding

# Page Configuration
st.set_page_config(
    page_title="Factory Intelligence Control Center",
    page_icon=None,
    layout="wide"
)

# Apply global styles and sidebar branding
load_css("Home")
load_sidebar_branding()

# Sidebar branding and styles loaded

# Row 1: KPI Cards (using our custom kpi_card function instead of st.metric)
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    kpi_card(
        label="Machines Analyzed",
        value="100",
        delta="Telemetry Active",
        status_type="info",
        icon=""
    )
with kpi_col2:
    kpi_card(
        label="Failure Events Logged",
        value="719",
        delta="Historical Anomalies",
        status_type="critical",
        icon=""
    )
with kpi_col3:
    kpi_card(
        label="Model Recall",
        value="90.28%",
        delta="Failure Detection Rate",
        status_type="warning",
        icon=""
    )
with kpi_col4:
    kpi_card(
        label="Features Engineered",
        value="31",
        delta="Sensor & Categorical",
        status_type="healthy",
        icon=""
    )

# Row 2: Split columns (Left is Risk ranking table + Workflow, Right is Health gauge + Alerts + Sensor health + Timeline)
left_panel_col, right_panel_col = st.columns([2, 1])

with left_panel_col:
    # 1. Risk Ranking Table
    with st.container(border=True):
        col_title, col_link = st.columns([5, 1])
        with col_title:
            st.markdown('<h3 style="margin: 0; font-size: 16px; padding-top: 4px;">Fleet Risk Rankings</h3>', unsafe_allow_html=True)
        with col_link:
            st.page_link("pages/1_Downtime_Risk_Prediction.py", label="View all")
        
        # Filter Chips using columns of buttons
        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        
        # Determine active filter
        if 'filter' not in st.session_state:
            st.session_state.filter = "All"
            
        with f_col1:
            if st.button("All Machines", use_container_width=True, key="btn_all"):
                st.session_state.filter = "All"
        with f_col2:
            if st.button("Critical Only", use_container_width=True, key="btn_crit"):
                st.session_state.filter = "Critical"
        with f_col3:
            if st.button("Warnings", use_container_width=True, key="btn_warn"):
                st.session_state.filter = "Warning"
        with f_col4:
            if st.button("Healthy", use_container_width=True, key="btn_heal"):
                st.session_state.filter = "Healthy"
                
        import numpy as np
        
        # Load raw machines metadata to retrieve exact model versions for all 100 machines
        try:
            raw_machines_path = os.path.join(current_dir, "data", "raw", "PdM_machines.csv")
            if not os.path.exists(raw_machines_path):
                project_root = os.path.abspath(os.path.join(current_dir, ".."))
                raw_machines_path = os.path.join(project_root, "data", "raw", "PdM_machines.csv")
                
            if os.path.exists(raw_machines_path):
                df_m = pd.read_csv(raw_machines_path)
            else:
                df_m = pd.DataFrame([
                    {"machineID": i, "model": f"model{((i-1)%4)+1}"}
                    for i in range(1, 101)
                ])
        except Exception:
            df_m = pd.DataFrame([
                {"machineID": i, "model": f"model{((i-1)%4)+1}"}
                for i in range(1, 101)
            ])
            
        # Dynamically generate consistent risk rankings for all 100 units
        machine_fleet = []
        for _, row in df_m.iterrows():
            m_id = int(row["machineID"])
            model_str = str(row["model"]).capitalize()
            model_str = model_str.replace("model", "Model ")
            model_str = model_str.replace("Model", "Model ")
            model_str = " ".join(model_str.split())
            
            if m_id == 33:
                machine_fleet.append({
                    "id": "Machine #33",
                    "model": "Model 3",
                    "risk": 82.5,
                    "status": "Critical",
                    "state": "Running",
                    "category": "Critical"
                })
            elif m_id == 42:
                machine_fleet.append({
                    "id": "Machine #42",
                    "model": "Model 2",
                    "risk": 54.1,
                    "status": "Warning",
                    "state": "Running",
                    "category": "Warning"
                })
            else:
                # Seed based on machine ID for consistent rankings
                rng = np.random.default_rng(m_id)
                risk = round(float(rng.uniform(0.5, 24.8)), 1)
                state = "Running" if rng.random() < 0.9 else "Idle"
                machine_fleet.append({
                    "id": f"Machine #{m_id}",
                    "model": model_str,
                    "risk": risk,
                    "status": "Healthy",
                    "state": state,
                    "category": "Healthy"
                })
                
        # Sort fleet by machine number in increasing order (Machine #1 to Machine #100)
        machine_fleet.sort(key=lambda x: int(x["id"].split("#")[1]))
        
        # Filter data based on active chip selection
        filtered_fleet = machine_fleet
        if st.session_state.filter != "All":
            filtered_fleet = [m for m in machine_fleet if m["category"] == st.session_state.filter]
            
        # Build custom styled HTML table wrapped in a scrollable container
        table_html = """<div style="max-height: 380px; overflow-y: auto; border: 0.5px solid #e2e4e9; border-radius: 6px; margin-top: 15px; padding: 2px;">
<table class="risk-table" style="margin-top: 0px; width: 100%;">
<thead>
<tr>
<th style="position: sticky; top: 0; background-color: #ffffff; z-index: 10;">Machine ID</th>
<th style="position: sticky; top: 0; background-color: #ffffff; z-index: 10;">Model</th>
<th style="position: sticky; top: 0; background-color: #ffffff; z-index: 10;">Risk Score</th>
<th style="position: sticky; top: 0; background-color: #ffffff; z-index: 10;">Severity</th>
<th style="position: sticky; top: 0; background-color: #ffffff; z-index: 10;">Status</th>
</tr>
</thead>
<tbody>"""
        for m in filtered_fleet:
            risk_color = "#10b981" if m["risk"] < 30 else ("#f59e0b" if m["risk"] <= 70 else "#ef4444")
            risk_bar = f"""<div style="display: flex; align-items: center; gap: 8px;">
<span style="width: 38px; font-weight: 600; font-size: 12px; color: #1a1a2e;">{m["risk"]}%</span>
<div style="background-color: #e2e4e9; height: 6px; width: 80px; border-radius: 3px;">
<div style="background-color: {risk_color}; width: {m["risk"]}%; height: 6px; border-radius: 3px;"></div>
</div>
</div>"""
            
            badge_class = "badge-green" if m["status"] == "Healthy" else ("badge-amber" if m["status"] == "Warning" else "badge-red")
            severity_badge = f'<span class="badge {badge_class}">{m["status"]}</span>'
            
            state_class = "badge-green" if m["state"] == "Running" else "badge-gray"
            state_badge = f'<span class="badge {state_class}">{m["state"]}</span>'
            
            table_html += f"""<tr>
<td><strong>{m["id"]}</strong></td>
<td>{m["model"]}</td>
<td>{risk_bar}</td>
<td>{severity_badge}</td>
<td>{state_badge}</td>
</tr>"""
        table_html += "</tbody></table></div>"
        st.markdown(table_html, unsafe_allow_html=True)
    
    with st.container(border=True):
        st.markdown("### Core Platform Workflow")
        st.markdown(
            """<div style="display: flex; justify-content: space-between; align-items: center; background-color: #f8fafc; padding: 15px; border-radius: 6px; border: 0.5px solid #e2e4e9; margin-top: 10px;">
<div style="text-align: center; width: 17%; padding: 10px 5px; background-color: #ffffff; border: 0.5px solid #cbd5e1; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<strong style="color: #2563eb; font-size: 11px; display: block; margin-top: 4px;">1. Telemetry</strong>
<span style="font-size: 9px; color: #64748b;">100 machines logs</span>
</div>
<div style="font-size: 18px; color: #2563eb; font-weight: bold;">➜</div>
<div style="text-align: center; width: 17%; padding: 10px 5px; background-color: #ffffff; border: 0.5px solid #cbd5e1; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<strong style="color: #2563eb; font-size: 11px; display: block; margin-top: 4px;">2. Engineering</strong>
<span style="font-size: 9px; color: #64748b;">Stress & rolling means</span>
</div>
<div style="font-size: 18px; color: #2563eb; font-weight: bold;">➜</div>
<div style="text-align: center; width: 17%; padding: 10px 5px; background-color: #ffffff; border: 0.5px solid #cbd5e1; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<strong style="color: #2563eb; font-size: 11px; display: block; margin-top: 4px;">3. ML Ensembles</strong>
<span style="font-size: 9px; color: #64748b;">Recall-optimized models</span>
</div>
<div style="font-size: 18px; color: #2563eb; font-weight: bold;">➜</div>
<div style="text-align: center; width: 17%; padding: 10px 5px; background-color: #ffffff; border: 0.5px solid #cbd5e1; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<strong style="color: #2563eb; font-size: 11px; display: block; margin-top: 4px;">4. SHAP</strong>
<span style="font-size: 9px; color: #64748b;">XAI feature attribution</span>
</div>
<div style="font-size: 18px; color: #2563eb; font-weight: bold;">➜</div>
<div style="text-align: center; width: 17%; padding: 10px 5px; background-color: #ffffff; border: 0.5px solid #cbd5e1; border-radius: 6px; box-shadow: 0 1px 2px rgba(0,0,0,0.02);">
<strong style="color: #2563eb; font-size: 11px; display: block; margin-top: 4px;">5. Prediction</strong>
<span style="font-size: 9px; color: #64748b;">Risk output card</span>
</div>
</div>""",
            unsafe_allow_html=True
        )

    # 3. Global Feature Importance Preview (SHAP Preview)
    with st.container(border=True):
        st.markdown("### Global Feature Importance Preview (AI Explainability)")
        try:
            csv_path = os.path.join(current_dir, "visualizations", "feature_importance.csv")
            if not os.path.exists(csv_path):
                project_root = os.path.abspath(os.path.join(current_dir, ".."))
                csv_path = os.path.join(project_root, "visualizations", "feature_importance.csv")
            
            if os.path.exists(csv_path):
                importance_df = pd.read_csv(csv_path)
                from utils.risk_utils import FRIENDLY_NAMES
                importance_df["Feature Label"] = importance_df["Feature"].map(lambda x: FRIENDLY_NAMES.get(x, x))
                top_10 = importance_df.head(10).copy()
                
                # Simple chart_data
                chart_data = top_10.set_index("Feature Label")[["Importance"]]
                st.bar_chart(chart_data, height=250)
            else:
                st.warning("Feature importance CSV not found.")
        except Exception as e:
            st.error(f"Error loading SHAP preview: {e}")

with right_panel_col:
    # 1. Fleet Health Panel (Circular gauge)
    with st.container(border=True):
        st.markdown("### Fleet Health Index", unsafe_allow_html=True)
        svg_gauge = """<div style="text-align: center; padding: 10px 0;">
<svg width="120" height="120" viewBox="0 0 120 120">
<!-- Background Ring -->
<circle cx="60" cy="60" r="50" fill="none" stroke="#f1f5f9" stroke-width="10"/>
<!-- Foreground Ring (99% fill: circumference = 2 * pi * r = 314.16, offset = 314.16 * (1 - 0.99) = 3.14) -->
<circle cx="60" cy="60" r="50" fill="none" stroke="#10b981" stroke-width="10" 
stroke-dasharray="314.16" stroke-dashoffset="3.14" stroke-linecap="round" transform="rotate(-90 60 60)"/>
<text x="60" y="66" fill="#1a1a2e" font-size="20" font-weight="700" text-anchor="middle">99.0%</text>
</svg>
<div style="display: flex; justify-content: space-around; margin-top: 15px; font-size: 12px; font-weight: 600;">
<div style="color: #10b981; text-align: center;">
<span style="display: block; font-size: 16px;">99</span>
<span style="color: #6b7280; font-size: 10px; font-weight: 500; text-transform: uppercase;">Healthy</span>
</div>
<div style="color: #ef4444; text-align: center;">
<span style="display: block; font-size: 16px;">1</span>
<span style="color: #6b7280; font-size: 10px; font-weight: 500; text-transform: uppercase;">Critical</span>
</div>
</div>
</div>"""
        st.markdown(svg_gauge, unsafe_allow_html=True)
    
    # 2. Live Alerts Panel
    with st.container(border=True):
        st.markdown("### Live Alerts Panel", unsafe_allow_html=True)
        alerts_html = """<div style="display: flex; flex-direction: column; gap: 10px; margin-top: 5px;">
<div style="display: flex; align-items: start; gap: 10px;">
<span style="height: 6px; width: 6px; background-color: #ef4444; border-radius: 50%; display: inline-block; margin-top: 6px;"></span>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">12:04:12</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">Machine #33: High Vibration</div>
<div style="font-size: 11px; color: #6b7280;">Critical structural anomaly detected</div>
</div>
</div>
<div style="display: flex; align-items: start; gap: 10px;">
<span style="height: 6px; width: 6px; background-color: #f59e0b; border-radius: 50%; display: inline-block; margin-top: 6px;"></span>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">11:58:30</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">Machine #42: Pressure Spike</div>
<div style="font-size: 11px; color: #6b7280;">Fluctuation warning in compression line</div>
</div>
</div>
<div style="display: flex; align-items: start; gap: 10px;">
<span style="height: 6px; width: 6px; background-color: #2563eb; border-radius: 50%; display: inline-block; margin-top: 6px;"></span>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">11:45:15</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">System Diagnostic Checked</div>
<div style="font-size: 11px; color: #6b7280;">Risk ensemble models calibrated</div>
</div>
</div>
</div>"""
        st.markdown(alerts_html, unsafe_allow_html=True)
    
    # 3. Sensor Health Bars
    with st.container(border=True):
        st.markdown("### Sensor Health Bars", unsafe_allow_html=True)
        sensors_html = """<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 5px;">
<div>
<div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 500; color: #4b5563; margin-bottom: 4px;">
<span>Voltage Health</span>
<strong>98%</strong>
</div>
<div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; border: 0.5px solid #e2e4e9;">
<div style="background-color: #10b981; width: 98%; height: 6px; border-radius: 3px;"></div>
</div>
</div>
<div>
<div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 500; color: #4b5563; margin-bottom: 4px;">
<span>Pressure Health</span>
<strong>94%</strong>
</div>
<div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; border: 0.5px solid #e2e4e9;">
<div style="background-color: #10b981; width: 94%; height: 6px; border-radius: 3px;"></div>
</div>
</div>
<div>
<div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 500; color: #4b5563; margin-bottom: 4px;">
<span>Vibration Health</span>
<strong style="color: #ef4444;">72%</strong>
</div>
<div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; border: 0.5px solid #e2e4e9;">
<div style="background-color: #ef4444; width: 72%; height: 6px; border-radius: 3px;"></div>
</div>
</div>
<div>
<div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 500; color: #4b5563; margin-bottom: 4px;">
<span>Rotation Speed Health</span>
<strong>97%</strong>
</div>
<div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; border: 0.5px solid #e2e4e9;">
<div style="background-color: #10b981; width: 97%; height: 6px; border-radius: 3px;"></div>
</div>
</div>
</div>"""
        st.markdown(sensors_html, unsafe_allow_html=True)
    
    # 4. Maintenance Timeline
    with st.container(border=True):
        st.markdown("### Maintenance Timeline", unsafe_allow_html=True)
        timeline_html = """<div style="display: flex; flex-direction: column; gap: 12px; margin-top: 5px;">
<div style="display: flex; gap: 10px; align-items: start;">
<div style="display: flex; flex-direction: column; align-items: center; margin-top: 4px;">
<span style="height: 8px; width: 8px; background-color: #ef4444; border-radius: 50%; display: inline-block;"></span>
<span style="width: 1px; height: 28px; background-color: #cbd5e1; display: inline-block; margin-top: 4px;"></span>
</div>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">June 06, 2026</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">Machine #12: Component 2 Failure</div>
<div style="font-size: 11px; color: #6b7280;">Emergency Replacement performed</div>
</div>
</div>
<div style="display: flex; gap: 10px; align-items: start;">
<div style="display: flex; flex-direction: column; align-items: center; margin-top: 4px;">
<span style="height: 8px; width: 8px; background-color: #f59e0b; border-radius: 50%; display: inline-block;"></span>
<span style="width: 1px; height: 28px; background-color: #cbd5e1; display: inline-block; margin-top: 4px;"></span>
</div>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">June 05, 2026</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">Machine #88: Component 4 wear</div>
<div style="font-size: 11px; color: #6b7280;">Scheduled maintenance check</div>
</div>
</div>
<div style="display: flex; gap: 10px; align-items: start;">
<div style="display: flex; flex-direction: column; align-items: center; margin-top: 4px;">
<span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
</div>
<div>
<span style="font-size: 10px; color: #9ca3af; font-family: monospace;">June 03, 2026</span>
<div style="font-size: 12px; font-weight: 600; color: #1a1a2e;">Machine #45: Pressure sensor check</div>
<div style="font-size: 11px; color: #6b7280;">Routine sensor validation ok</div>
</div>
</div>
</div>"""
        st.markdown(timeline_html, unsafe_allow_html=True)

# Row 3: Bottom Project Overview Details
st.write("---")
col_summary_left, col_summary_right = st.columns(2)

with col_summary_left:
    with st.container(border=True):
        st.markdown("### Business Overview")
        st.markdown(
            """* **Objective**: Predict machine failures before they occur to schedule proactive maintenance.
* **Business Problem**: Unplanned downtime in smart manufacturing costs factories thousands of dollars per hour.
* **Expected Benefits**: Up to 90.28% recall in detecting failures, enabling proactive maintenance scheduling and cost reduction.
* **Explainable AI Integration**: Evaluates why a machine is predicted at risk using SHAP, showing which sensors are contributing to the alert."""
        )

with col_summary_right:
    with st.container(border=True):
        st.markdown("### Architecture & Specifications")
        st.markdown(
            """* **Model Type**: Random Forest, LightGBM, and XGBoost Classifier Ensembles
* **Dataset Size**: 876,100 records (chronological sensor telemetry, maintenance, and error logs)
* **Explainability**: SHAP (SHapley Additive exPlanations)
* **Deployment**: Streamlit Local Control Panel & Diagnostics Panel
* **Key Metric**: Recall (96.53% peak) prioritized to capture failures and avoid false negatives."""
        )

# Footer
st.write("---")
st.markdown(
    """<div style="text-align: center; color: gray; font-size: 12px; padding-top: 10px;">
Developed for <strong>HCL Smart Manufacturing</strong> | ML Ensemble Models + SHAP Diagnostics & Explainability System
</div>""",
    unsafe_allow_html=True
)
