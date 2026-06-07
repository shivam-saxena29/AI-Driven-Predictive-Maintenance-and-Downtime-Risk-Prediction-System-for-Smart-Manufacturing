import streamlit as st

def load_css(active_page="Home"):
    """
    Injects global CSS to style the app according to the white industrial theme.
    Hides native streamlit headers and page links, and styles card elements.
    """
    href_targets = {
        "Home": 'a[href="/"]',
        "Prediction": 'a[href*="1_Downtime_Risk_Prediction"]',
        "EDA": 'a[href*="2_EDA_Insights"]',
        "SHAP": 'a[href*="3_SHAP_Insights"]',
        "Performance": 'a[href*="4_Model_Performance"]'
    }
    target = href_targets.get(active_page, 'a[href="/"]')
    
    css = f"""
    <style>
        /* White theme backgrounds */
        .stApp {{
            background-color: #f7f8fa !important;
            color: #1a1a2e !important;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }}
        
        /* Hide native header and default sidebar navigation list */
        header[data-testid="stHeader"] {{
            visibility: hidden !important;
            height: 0px !important;
        }}
        [data-testid="stSidebarNav"] {{
            display: none !important;
        }}
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {{
            background-color: #ffffff !important;
            border-right: 0.5px solid #e2e4e9 !important;
            padding-top: 1rem !important;
        }}
        
        /* Muted gray style for custom page links */
        [data-testid="stSidebar"] [data-testid="stPageLink"] a {{
            color: #4b5563 !important;
            border-left: 3px solid transparent !important;
            padding-left: 12px !important;
            background-color: transparent !important;
            font-size: 13px !important;
            font-weight: 500 !important;
            margin: 2px 0px !important;
            transition: all 0.15s ease !important;
            text-decoration: none !important;
        }}
        
        [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{
            color: #2563eb !important;
            background-color: #f3f4f6 !important;
        }}
        
        /* Active page highlight: Blue left border + blue background hint */
        [data-testid="stSidebar"] [data-testid="stPageLink"] {target} {{
            color: #2563eb !important;
            font-weight: 600 !important;
            border-left: 3px solid #2563eb !important;
            background-color: #eff6ff !important;
        }}
        
        /* Section labels in sidebar */
        .sidebar-section-label {{
            font-size: 10px;
            font-weight: 700;
            color: #9ca3af;
            margin-top: 18px;
            margin-bottom: 6px;
            padding-left: 15px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
        }}
        
        /* Hide page link icons on sidebar to keep it clean and modern */
        [data-testid="stSidebar"] [data-testid="stPageLinkIcon"] {{
            display: none !important;
        }}
        
        /* Mute st.metric boxes since we render custom ones */
        [data-testid="metric-container"] {{
            display: none !important;
        }}
        
        /* Cards container styling */
        .custom-card {{
            background-color: #ffffff;
            border: 0.5px solid #e2e4e9;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.02);
            margin-bottom: 20px;
            color: #1a1a2e;
        }}

        /* High contrast primary button style overrides */
        button[data-testid="stBaseButton-primary"],
        div[data-testid="stButton"] button[type="primary"],
        div.stButton > button {{
            background-color: #2563eb !important;
            color: #ffffff !important;
            border: none !important;
        }}
        button[data-testid="stBaseButton-primary"]:hover,
        div[data-testid="stButton"] button[type="primary"]:hover,
        div.stButton > button:hover {{
            background-color: #1d4ed8 !important;
            color: #ffffff !important;
        }}
        button[data-testid="stBaseButton-primary"] p,
        button[data-testid="stBaseButton-primary"] span,
        button[data-testid="stBaseButton-primary"] div,
        div.stButton > button p,
        div.stButton > button span,
        div.stButton > button div {{
            color: #ffffff !important;
            font-weight: 600 !important;
        }}
        
        /* Typography adjustments */
        h1, h2, h3, h4 {{
            color: #1a1a2e !important;
            font-weight: 700 !important;
            margin-bottom: 8px !important;
        }}
        p {{
            font-size: 13px !important;
            color: #4b5563 !important;
            line-height: 1.5 !important;
        }}
        
        /* Custom styled table rules */
        .risk-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        .risk-table th {{
            text-align: left;
            padding: 10px;
            font-size: 11px;
            font-weight: 600;
            color: #6b7280;
            border-bottom: 1px solid #e2e4e9;
            text-transform: uppercase;
        }}
        .risk-table td {{
            padding: 12px 10px;
            border-bottom: 0.5px solid #e2e4e9;
            color: #1a1a2e;
        }}
        
        /* Status Badges */
        .badge {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.3px;
        }}
        .badge-green {{ background-color: #d1fae5; color: #065f46; }}
        .badge-amber {{ background-color: #fef3c7; color: #92400e; }}
        .badge-red {{ background-color: #fee2e2; color: #991b1b; }}
        .badge-gray {{ background-color: #f3f4f6; color: #374151; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def kpi_card(label: str, value: str, delta: str, status_type: str = "info", icon: str = ""):
    """
    Renders custom KPI card with left-border accent color and icon.
    status_type: 'info' (blue), 'critical' (red), 'warning' (amber), 'healthy' (green)
    """
    colors = {
        "info": "#2563eb",      # Blue
        "critical": "#ef4444",  # Red
        "warning": "#f59e0b",   # Amber
        "healthy": "#10b981"    # Green
    }
    accent = colors.get(status_type, "#2563eb")
    
    # Style delta text color
    if "+" in delta or "healthy" in delta.lower() or "detection" in delta.lower() or "tracks" in delta.lower():
        delta_color = "#10b981"
    elif "-" in delta or "critical" in delta.lower() or "anomalies" in delta.lower():
        delta_color = "#ef4444"
    else:
        delta_color = "#6b7280"
        
    html = f"""<div style="background-color: #ffffff; border: 0.5px solid #e2e4e9; border-left: 4.5px solid {accent}; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.02); margin-bottom: 15px;">
<div style="font-size: 11px; font-weight: 600; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; display: flex; align-items: center; gap: 6px;">
<span>{icon}</span> {label}
</div>
<div style="font-size: 22px; font-weight: 700; color: #1a1a2e; margin-top: 4px; margin-bottom: 2px;">{value}</div>
<div style="font-size: 11px; font-weight: 500; color: {delta_color};">{delta}</div>
</div>"""
    st.markdown(html, unsafe_allow_html=True)

def load_sidebar_branding():
    """
    Renders grouped custom sidebar navigation and sync footer.
    """
    st.sidebar.markdown(
        """<div style="padding: 10px 14px; margin-bottom: 15px; background-color: #ffffff; border: 0.5px solid #e2e4e9; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.01);">
<div style="font-size: 13px; font-weight: 700; color: #1a1a2e; display: flex; align-items: center; gap: 8px;">
⚙️ Smart Manufacturing
</div>
<div style="font-size: 10px; color: #6b7280; font-weight: 500; margin-top: 2px;">
Predictive Maintenance System
</div>
</div>
<div class="sidebar-section-label">Overview</div>""",
        unsafe_allow_html=True
    )
    
    # Overview Links
    st.sidebar.page_link("app.py", label="Home Dashboard")
    st.sidebar.page_link("pages/2_EDA_Insights.py", label="EDA Insights")
    
    # Diagnostics Links (with critical alert badge counter next to Downtime Risk Prediction)
    st.sidebar.markdown('<div class="sidebar-section-label">Diagnostics & Predictions</div>', unsafe_allow_html=True)
    st.sidebar.page_link(
        "pages/1_Downtime_Risk_Prediction.py", 
        label="Risk Forecast (5)"
    )
    st.sidebar.page_link("pages/3_SHAP_Insights.py", label="SHAP Diagnostics")
    
    # Performance Link
    st.sidebar.markdown('<div class="sidebar-section-label">Performance</div>', unsafe_allow_html=True)
    st.sidebar.page_link("pages/4_Model_Performance.py", label="Model Performance")
    
    # About Section (collapsible details panel)
    st.sidebar.markdown(
        """<details style="margin-top: 30px; padding: 10px 14px; background-color: #ffffff; border: 0.5px solid #e2e4e9; border-radius: 8px; font-size: 12px; color: #4b5563; cursor: pointer;">
<summary style="font-weight: 600; color: #1a1a2e; list-style: none; display: flex; justify-content: space-between; align-items: center;">
<span>About this Project</span>
<span style="font-size: 10px; color: #9ca3af;">▼</span>
</summary>
<div style="margin-top: 10px; line-height: 1.5; font-size: 11px;">
<p style="margin: 0 0 6px 0; font-weight: 600; color: #1a1a2e;">Overview</p>
<p style="margin: 0 0 10px 0;">This Predictive Maintenance System analyzes real-time telemetry from 100 industrial machines to predict failure and downtime before they occur, preventing costly unplanned outages.</p>

<p style="margin: 0 0 6px 0; font-weight: 600; color: #1a1a2e;">Data & Model</p>
<p style="margin: 0 0 10px 0;">Trained on 876,100 historical telemetry points. The core engine runs Random Forest, LightGBM, and XGBoost Classifier models optimized for maximum safety.</p>

<p style="margin: 0 0 6px 0; font-weight: 600; color: #1a1a2e;">Explainable AI (SHAP)</p>
<p style="margin: 0 0 0 0;">SHAP values decode the model predictions, showing engineers exactly which sensors (vibration, pressure, voltage, speed) are driving the failure risk.</p>
</div>
</details>""",
        unsafe_allow_html=True
    )
    
    # Sidebar Footer (stuck at bottom of sidebar list)
    st.sidebar.markdown(
        """<div style="margin-top: 25px; padding-top: 15px; border-top: 0.5px solid #e2e4e9; font-size: 10px; color: #9ca3af; text-align: center; font-family: monospace;">
Last sync: 1 min ago • v1.0
</div>""",
        unsafe_allow_html=True
    )
