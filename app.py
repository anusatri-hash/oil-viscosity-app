import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------
# 1. Page Configuration & Custom CSS (Crimson Dark Red Theme)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oil Viscosity Dashboard",
    page_icon="🛢️",
    layout="wide"
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stSidebar"] {
        background-color: #6B0D0D;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    .kpi-card {
        background-color: #7A1212;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-title { font-size: 14px; opacity: 0.9; margin-bottom: 5px; }
    .kpi-value { font-size: 26px; font-weight: bold; }
    .kpi-sub { font-size: 12px; opacity: 0.8; }
    
    .status-normal { background-color: #1B5E20; border: 2px solid #4CAF50; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    .status-warning { background-color: #E65100; border: 2px solid #FF9800; color: white; padding: 15px; border-radius: 10px; text-align: center; }
    .status-alarm { background-color: #FFD4D4; border: 2px solid #D32F2F; color: #B71C1C !important; padding: 15px; border-radius: 10px; text-align: center; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. Load Machine Learning Model (.pkl)
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        return None

model = load_ml_model()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION & INPUTS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛢️ **OIL VISCOSITY**")
    st.caption("Predictive Maintenance (ML Connected)")
    st.write("---")
    
    st.subheader("📥 Model Inputs")
    machine_id = st.text_input("Machine ID", "HYD-01")
    op_hours = st.number_input("Operating Hours", min_value=0, value=2000, step=100)
    fresh_40 = st.number_input("Fresh Viscosity 40°C (Baseline)", min_value=1.0, value=46.0, step=1.0)
    fresh_100 = st.number_input("Fresh Viscosity 100°C", min_value=1.0, value=6.8, step=0.1)
    meas_100 = st.number_input("Measured Viscosity 100°C", min_value=1.0, value=6.5, step=0.1)
    meas_40_actual = st.number_input("Measured Viscosity 40°C (Actual)", min_value=1.0, value=52.0, step=0.5)
    
    st.write("---")
    st.markdown("🟢 **Database Status**<br><small>Connected: oil_viscosity_db</small>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. ML PREDICTION & ALERT LOGIC
# ---------------------------------------------------------
if model is not None:
    input_data = np.array([[op_hours, fresh_40, fresh_100, meas_100]])
    predicted_visc_40 = float(model.predict(input_data)[0])
    model_status_msg = "🟢 Connected to model.pkl successfully"
else:
    predicted_visc_40 = fresh_40 * (1 + (op_hours / 10000) * 0.8)
    model_status_msg = "⚠️ model.pkl not found (Using fallback calculation)"

dev_percent = ((meas_40_actual - fresh_40) / fresh_40) * 100

if abs(dev_percent) >= 15:
    status_label = "Alarm"
    status_class = "status-alarm"
    status_msg = f"Viscosity deviation {dev_percent:+.2f}% (Exceeds Alarm limit ±15%)"
elif abs(dev_percent) >= 10:
    status_label = "Warning"
    status_class = "status-warning"
    status_msg = f"Viscosity deviation {dev_percent:+.2f}% (Exceeds Warning limit ±10%)"
else:
    status_label = "Normal"
    status_class = "status-normal"
    status_msg = f"Viscosity within normal range ({dev_percent:+.2f}%)"

# ---------------------------------------------------------
# 5. DASHBOARD HEADER & FILTERS
# ---------------------------------------------------------
st.markdown("""
<div style="background-color: #8B0000; padding: 15px; border-radius: 8px; color: white; margin-bottom: 15px;">
    <h2 style="margin:0; color:white;">📈 Dashboard</h2>
    <p style="margin:0; opacity:0.8;">Hydraulic Oil Viscosity Trend & Predictive Maintenance Alerts</p>
</div>
""", unsafe_allow_html=True)

st.caption(model_status_msg)

# Filters Row
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    st.selectbox("Machine Select", ["HYD-01", "HYD-02", "HYD-03"])
with f_col2:
    st.selectbox("Cycle Select", ["CYCLE-03", "CYCLE-02", "CYCLE-01"])
with f_col3:
    st.date_input("Date Range", [])

st.write("")

# ---------------------------------------------------------
# 6. EXECUTIVE KPI CARDS
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">💧 Latest Measured (40°C)</div>
        <div class="kpi-value">{meas_40_actual:.2f} cSt</div>
        <div class="kpi-sub">(at {op_hours:,} Hours)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🧠 Predicted Value (ML)</div>
        <div class="kpi-value">{predicted_visc_40:.2f} cSt</div>
        <div class="kpi-sub">(at {op_hours + 200:,} Hours)</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">🛢️ Baseline (Fresh Oil)</div>
        <div class="kpi-value">{fresh_40:.2f} cSt</div>
        <div class="kpi-sub">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="{status_class}">
        <div class="kpi-title">⚠️ Current Status</div>
        <div class="kpi-value">{status_label}</div>
        <div class="kpi-sub">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 7. TIME-SERIES & CONTROL CHARTS
# ---------------------------------------------------------
c_col1, c_col2 = st.columns([1.2, 1])

# Left Chart: Viscosity Trend
with c_col1:
    st.subheader("📈 Viscosity Trend")
    
    hours_axis = np.linspace(0, max(2500, op_hours + 500), 10)
    
    if model is not None:
        sim_X = np.column_stack([
            hours_axis,
            np.full_like(hours_axis, fresh_40),
            np.full_like(hours_axis, fresh_100),
            np.full_like(hours_axis, meas_100)
        ])
        trend_y = model.predict(sim_X)
    else:
        trend_y = fresh_40 * (1 + (hours_axis / 10000) * 0.8)

    fig_trend = go.Figure()
    fig_trend.add_trace(go.Scatter(x=hours_axis, y=trend_y, mode='lines', name='Predicted (ML)', line=dict(color='#FB8C00', dash='dash', width=2)))
    fig_trend.add_trace(go.Scatter(x=[op_hours], y=[meas_40_actual], mode='markers', name='Measured (Actual)', marker=dict(color='#1E88E5', size=12)))
    
    fig_trend.add_hline(y=fresh_40, line_color="#4CAF50", annotation_text="Baseline")
    fig_trend.add_hline(y=fresh_40 * 1.10, line_dash="dash", line_color="#FDD835", annotation_text="Warning (+10%)")
    fig_trend.add_hline(y=fresh_40 * 1.15, line_dash="dash", line_color="#E53935", annotation_text="Alarm (+15%)")

    fig_trend.update_layout(
        xaxis_title="Operating Hours (h)",
        yaxis_title="Kinematic Viscosity (cSt)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom
