import os
import sys
import pandas as pd
import streamlit as st
import plotly.express as px

# Resolve paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

from utils.risk_utils import FRIENDLY_NAMES
from utils.styles import load_css, load_sidebar_branding, kpi_card

# Page Configuration
st.set_page_config(
    page_title="SHAP Diagnostics | Smart Manufacturing",
    page_icon=None,
    layout="wide"
)

# Apply global styles and sidebar branding
load_css("SHAP")
load_sidebar_branding()

# Sidebar branding and styles loaded

# Page Title & Description wrapped in a custom card
st.markdown(
    """<div class="custom-card" style="margin-bottom: 25px;">
<h2 style="margin: 0; font-size: 18px; color: #1a1a2e;">SHAP Insights</h2>
<p style="margin: 5px 0 0 0; color: #4b5563; font-size: 13px;">
Explain how the machine learning model evaluates downtime risk. SHAP (SHapley Additive exPlanations) values provide mathematically rigorous feature attribution.
</p>
</div>""",
    unsafe_allow_html=True
)

# Section 1: Explainable AI Overview
st.info(
    """**Explainable AI (XAI) Overview:**
Tree ensemble classifiers (Random Forest, LightGBM, XGBoost) are highly interpretable, but understanding complex feature interactions requires a global framework. 
We use **SHAP** to decode the model, attributing a contribution score to each variable. 
This guarantees that our AI system is completely transparent and audit-ready for control-room operators."""
)

# Section 2: Global Feature Importance Chart (with loading spinner inside card wrapper)
with st.container(border=True):
    st.subheader("Top 15 Most Influential Features (Global SHAP)")
    
    try:
        with st.spinner("Loading SHAP feature importance data..."):
            csv_path = os.path.join(project_root, "visualizations", "feature_importance.csv")
            if os.path.exists(csv_path):
                df_imp = pd.read_csv(csv_path)
                
                # Map feature names to user-friendly labels
                df_imp["Feature Label"] = df_imp["Feature"].map(lambda x: FRIENDLY_NAMES.get(x, x))
                
                # Sort descending and take top 15
                df_imp_sorted = df_imp.sort_values(by="Importance", ascending=True)
                top_15 = df_imp_sorted.tail(15)
                
                # Plotly horizontal bar chart
                fig_imp = px.bar(
                    top_15,
                    x="Importance",
                    y="Feature Label",
                    orientation="h",
                    color="Importance",
                    color_continuous_scale="viridis",
                    labels={"Importance": "SHAP Global Importance", "Feature Label": "Feature"},
                )
                fig_imp.update_layout(
                    margin=dict(l=20, r=20, t=10, b=20),
                    height=450,
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font={'color': '#1a1a2e'}
                )
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning(f"Could not locate feature_importance.csv at: {csv_path}")
    except Exception as e:
        st.error(f"Error loading global importance chart: {e}")

st.write("")

# Section 3 & 4: SHAP Plots side-by-side inside card wrappers
with st.container(border=True):
    st.subheader("Global SHAP Summary Plots")
    col_plot1, col_plot2 = st.columns(2)
    
    with col_plot1:
        summary_img = os.path.join(project_root, "visualizations", "shap_summary.png")
        if os.path.exists(summary_img):
            st.image(
                summary_img, 
                caption="SHAP Summary Plot (Beeswarm): Shows how high (red) or low (blue) feature values impact the failure probability.", 
                use_container_width=True
            )
        else:
            st.warning("SHAP Beeswarm summary image not found.")
            
    with col_plot2:
        bar_img = os.path.join(project_root, "visualizations", "shap_feature_importance.png")
        if os.path.exists(bar_img):
            st.image(
                bar_img, 
                caption="SHAP Feature Importance Plot: Mean absolute SHAP values ranking the overall impact on prediction output.", 
                use_container_width=True
            )
        else:
            st.warning("SHAP global bar chart image not found.")

st.write("")

# Section 5: Top Risk Drivers (using our custom kpi_card function in 5-columns)
st.subheader("Top 5 Machine Failure Drivers")
st.markdown("Global attribution weights assigned by the model:")
m1, m2, m3, m4, m5 = st.columns(5)

with m1:
    kpi_card(
        label="1. Maintenance",
        value="0.3825",
        delta="Strongest Driver",
        status_type="critical"
    )
with m2:
    kpi_card(
        label="2. Component 2",
        value="0.0333",
        delta="Replacement History",
        status_type="warning"
    )
with m3:
    kpi_card(
        label="3. Stress Index",
        value="0.0244",
        delta="Pressure x Vibration",
        status_type="warning"
    )
with m4:
    kpi_card(
        label="4. Prod. Load",
        value="0.0166",
        delta="RPM / Max RPM",
        status_type="info"
    )
with m5:
    kpi_card(
        label="5. Health Index",
        value="0.0142",
        delta="Telemetry Average",
        status_type="healthy"
    )

st.write("")

# Section 6: Business Interpretation
with st.container(border=True):
    st.subheader("Expert Business Interpretation")
    col_ins1, col_ins2, col_ins3 = st.columns(3)
    
    with col_ins1:
        st.markdown("#### **Insight 1: Maintenance History**")
        st.info(
            "**Recent maintenance events and replacements** are the single strongest predictors of failures. "
            "This indicates that machines requiring frequent component replacements are often in a highly degraded state, "
            "or post-service calibration errors are triggering subsequent downtime."
        )
        
    with col_ins2:
        st.markdown("#### **Insight 2: Composite Stress**")
        st.warning(
            "**Machine Stress Index** (pressure times vibration) has high importance. "
            "Machines operating under combined pressure fluctuations and high-amplitude vibration "
            "degrade significantly faster than machines under single-axis load."
        )
        
    with col_ins3:
        st.markdown("#### **Insight 3: Production Load**")
        st.success(
            "**High production loads (high RPM operations)** increase the thermal and torque load on physical rotors. "
            "Running machines near maximum capacity accelerates structural degradation, shortening replacement windows."
        )

# Section 7: Explainability Summary
st.write("")
st.success(
    """**Operational Confidence:**
The predictive model is **not a black box**. 
Every single alert generated on the prediction panel is accompanied by localized feature attributions. 
This enables plant administrators to verify, trust, and act on AI decisions with complete visibility."""
)
