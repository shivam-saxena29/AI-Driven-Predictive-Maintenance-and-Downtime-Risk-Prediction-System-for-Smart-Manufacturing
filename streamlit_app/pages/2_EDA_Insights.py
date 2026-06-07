import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="EDA Insights | Smart Manufacturing",
    page_icon=None,
    layout="wide"
)

# Add project root to path to resolve imports correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
sys_path_root = os.path.abspath(os.path.join(current_dir, ".."))
if sys_path_root not in sys.path:
    sys.path.append(sys_path_root)

from utils.styles import load_css, load_sidebar_branding, kpi_card

# Apply custom styling and sidebar branding
load_css("EDA")
load_sidebar_branding()

# Sidebar branding and styles loaded

# Page Title & Description wrapped in a custom card
st.markdown(
    """<div class="custom-card" style="margin-bottom: 25px;">
<h2 style="margin: 0; font-size: 18px; color: #1a1a2e;">EDA Insights</h2>
<p style="margin: 5px 0 0 0; color: #4b5563; font-size: 13px;">
Explore telemetry distribution, failure counts, historical errors, and correlation trends across the smart manufacturing dataset.
</p>
</div>""",
    unsafe_allow_html=True
)

# ── Performance Optimized Data Loaders ──────────────────────────────────────────
@st.cache_data
def load_model_telemetry():
    cols = [
        'failure_flag', 'volt', 'rotate', 'pressure', 'vibration',
        'health_index', 'machine_stress_index', 'production_load'
    ]
    path = "data/final/final_model_dataset.csv"
    if not os.path.exists(path):
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        path = os.path.join(project_root, "data", "final", "final_model_dataset.csv")
    return pd.read_csv(path, usecols=cols)

@st.cache_data
def load_raw_errors():
    path = "data/raw/PdM_errors.csv"
    if not os.path.exists(path):
        project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
        path = os.path.join(project_root, "data", "raw", "PdM_errors.csv")
    return pd.read_csv(path)

# Section 1: Dataset Overview KPI Cards
st.subheader("Dataset Overview")
kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

with kpi_col1:
    kpi_card(
        label="Total Records (Sensors)",
        value="876,100",
        delta="Telemetry Points",
        status_type="info",
        icon=""
    )
with kpi_col2:
    kpi_card(
        label="Machines Monitored",
        value="100",
        delta="Active Units",
        status_type="healthy",
        icon=""
    )
with kpi_col3:
    kpi_card(
        label="Downtime Events Logged",
        value="719",
        delta="Logged Failures",
        status_type="critical",
        icon=""
    )
with kpi_col4:
    kpi_card(
        label="Features Available",
        value="33",
        delta="Dataset Columns",
        status_type="warning",
        icon=""
    )

# Load Datasets with loading spinner
try:
    with st.spinner("Loading telemetry distributions & error logs..."):
        df_model = load_model_telemetry()
        df_errors = load_raw_errors()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.info("Please run the data pipeline (`run_pipeline.py`) first to generate files.")
    st.stop()

# Row 1: Failure Distribution (Pie) & Error Log Frequency (Bar)
col_fail, col_err = st.columns(2)

with col_fail:
    with st.container(border=True):
        st.markdown("### Downtime Class Imbalance")
        failure_counts = df_model['failure_flag'].value_counts()
        failure_counts.index = ['Healthy (0)' if x == 0 else 'Failure (1)' for x in failure_counts.index]
        
        fig_pie = px.pie(
            names=failure_counts.index,
            values=failure_counts.values,
            color=failure_counts.index,
            color_discrete_map={'Healthy (0)': '#10B981', 'Failure (1)': '#EF4444'},
            hole=0.4
        )
        fig_pie.update_layout(
            margin=dict(l=20, r=20, t=20, b=20),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            font={'color': '#1a1a2e'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.caption("Failures represent only ~0.08% of the dataset, illustrating extreme class imbalance.")

with col_err:
    with st.container(border=True):
        st.markdown("### Historical Error Log Counts")
        error_counts = df_errors['errorID'].value_counts().reset_index()
        error_counts.columns = ['Error Type', 'Occurrences']
        error_counts = error_counts.sort_values('Error Type')
        
        fig_bar = px.bar(
            error_counts,
            x='Error Type',
            y='Occurrences',
            color='Error Type',
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_bar.update_layout(
            margin=dict(l=20, r=20, t=25, b=20),
            height=280,
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': '#1a1a2e'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.caption("Distribution of the 5 error codes triggered during machine operations.")

st.write("")

# Row 2: Sensor Telemetry Distribution Tabs (Histogram & Boxplot)
with st.container(border=True):
    st.markdown("### Sensor Telemetry Analysis")
    st.markdown(
        """Select tabs to examine sensor distributions. Plotly visualizations are rendered 
using a representative sample of 10,000 rows to ensure lag-free performance."""
    )
    
    # Sample 10,000 rows for fast distribution rendering in browser
    np.random.seed(42)
    df_sensor_sample = df_model.sample(n=min(10000, len(df_model))).copy()
    
    sensor_tabs = st.tabs(["Voltage", "Pressure", "Vibration", "Rotation Speed"])
    sensor_cols = {
        "Voltage": ("volt", "Volt (V)", "#3B82F6"),
        "Pressure": ("pressure", "psi", "#10B981"),
        "Vibration": ("vibration", "mm/s", "#F59E0B"),
        "Rotation Speed": ("rotate", "RPM", "#8B5CF6")
    }
    
    for tab, label in zip(sensor_tabs, ["Voltage", "Pressure", "Vibration", "Rotation Speed"]):
        col_name, unit, color = sensor_cols[label]
        with tab:
            col_hist, col_box = st.columns(2)
            with col_hist:
                fig_hist = px.histogram(
                    df_sensor_sample,
                    x=col_name,
                    nbins=50,
                    color_discrete_sequence=[color]
                )
                fig_hist.update_layout(
                    title_text=f"{label} Distribution Density ({unit})",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': '#1a1a2e'}
                )
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_box:
                fig_box = px.box(
                    df_sensor_sample,
                    y=col_name,
                    color_discrete_sequence=[color]
                )
                fig_box.update_layout(
                    title_text=f"{label} Range & Outliers ({unit})",
                    height=280,
                    margin=dict(l=20, r=20, t=40, b=20),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': '#1a1a2e'}
                )
                st.plotly_chart(fig_box, use_container_width=True)

st.write("")

# Row 3: Correlation Heatmap
with st.container(border=True):
    st.markdown("### Correlation Analysis")
    corr_features = [
        'volt', 'rotate', 'pressure', 'vibration', 
        'health_index', 'machine_stress_index', 'production_load'
    ]
    existing_corr_cols = [c for c in corr_features if c in df_model.columns]
    corr_matrix = df_model[existing_corr_cols].corr()
    
    friendly_labels = {
        'volt': 'Voltage',
        'rotate': 'Rotation Speed',
        'pressure': 'Pressure',
        'vibration': 'Vibration',
        'health_index': 'Health Index',
        'machine_stress_index': 'Stress Index',
        'production_load': 'Production Load'
    }
    corr_labels = [friendly_labels.get(c, c) for c in existing_corr_cols]
    
    fig_heat = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        colorscale='RdBu',
        zmin=-1.0,
        zmax=1.0,
        text=np.round(corr_matrix.values, 2),
        texttemplate="%{text}",
        hoverongaps=False
    ))
    fig_heat.update_layout(
        title='Correlation Heatmap (Key Sensor & Derived Features)',
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': '#1a1a2e'}
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.write("")

# Row 4: SHAP Business Insights Preview & Business Insights Panel
with st.container(border=True):
    st.markdown("### SHAP-Powered Business Insights")
    
    col_shap, col_insights = st.columns(2)
    
    with col_shap:
        st.markdown("#### **Top 3 Failure Risk Drivers (from SHAP)**")
        st.markdown("""1. **Top Driver: Maintenance History**
   - The cumulative maintenance count has a high correlation split. Frequent or recent component replacements are strongly flagged.

2. **Second Driver: Machine Stress**
   - The interaction of high pressure and high vibration (`machine_stress_index`) is a secondary predictor of mechanical failure.
   
3. **Third Driver: Production Load**
   - High rotation speeds (RPM) relative to peak limits (`production_load`) increase thermal and torque wear.""")
    
    with col_insights:
        st.markdown("#### **Operational Control Guidelines**")
        
        st.info(
            "**Maintenance Quality checks**: Machines with frequent maintenance logs show higher downtime risk. "
            "Post-maintenance calibration audits are highly recommended to ensure components are aligned correctly."
        )
        
        st.warning(
            "**Stress Limit Controls**: High vibration coupled with high pressure increases machine stress exponentially. "
            "Limit peak pressure thresholds during high-vibration batches."
        )
        
        st.success(
            "**Recall Focus**: Our model is highly recall-optimized (90.28%). "
            "It acts as a conservative safety warning system—preferring minor false alarms over missing critical machine downtime."
        )
