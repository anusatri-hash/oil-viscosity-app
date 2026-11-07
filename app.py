import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------
# 1. การตั้งค่าหน้าเว็บ (Page Config & CSS Custom Theme)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oil Viscosity Dashboard",
    page_icon="🛢️",
    layout="wide"
)

# Custom CSS ตกแต่งธีมสีแดงเข้ม (Dark Red / Crimson Theme) ให้เหมือนในรูป
st.markdown("""
<style>
    /* ซ่อน Header และ Footer หลักของ Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* สีพื้นหลัง Sidebar */
    [data-testid="stSidebar"] {
        background-color: #6B0D0D;
        color: white;
    }
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* สไตล์การ์ด KPI ด้านบน */
    .kpi-card {
        background-color: #7A1212;
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .kpi-title {
        font-size: 14px;
        opacity: 0.9;
        margin-bottom: 5px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: bold;
    }
    .kpi-sub {
        font-size: 12px;
        opacity: 0.8;
    }
    
    /* การ์ดสถานะ Alarm (สีแดงสว่าง/ชมพูอ่อน) */
    .kpi-alarm {
        background-color: #FFD4D4;
        border: 2px solid #D32F2F;
        color: #B71C1C !important;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
    }
    .kpi-alarm .kpi-value {
        color: #D32F2F !important;
    }

    /* ตาราง Dataframe */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# 2. โหลดโมเดล Machine Learning
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('model.pkl')
    except:
        return None

model = load_ml_model()

# ---------------------------------------------------------
# 3. SIDEBAR NAVIGATION & STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛢️ **OIL VISCOSITY**")
    st.caption("Predictive Maintenance")
    st.write("---")
    
    menu = st.radio(
        "เมนูหลัก",
        ["📊 Dashboard", "📤 อัปโหลดข้อมูล (CSV)", "🗄️ ข้อมูลจาก MySQL", "🧠 พยากรณ์ความหนืด (ML)", "🔔 ประวัติการแจ้งเตือน", "⚙️ ตั้งค่าเกณฑ์ (Control Chart)"],
        label_visibility="collapsed"
    )
    
    st.write("---")
    st.markdown("🟢 **เชื่อมต่อฐานข้อมูล**<br><small>MySQL : oil_viscosity_db</small>", unsafe_allow_html=True)

# ---------------------------------------------------------
# 4. TOP BAR & FILTERS (หัวข้อ + ตัวกรองด้านบน)
# ---------------------------------------------------------
# Header Bar สีแดงเข้ม
st.markdown("""
<div style="background-color: #8B0000; padding: 15px; border-radius: 8px; color: white; margin-bottom: 20px;">
    <h2 style="margin:0; color:white;">📈 Dashboard</h2>
    <p style="margin:0; opacity:0.8;">ติดตามแนวโน้มความหนืดน้ำมันไฮดรอลิก และแจ้งเตือนการบำรุงรักษา</p>
</div>
""", unsafe_allow_html=True)

# ตัวกรอง 3 ช่อง
f_col1, f_col2, f_col3 = st.columns(3)
with f_col1:
    selected_machine = st.selectbox("เครื่องจักร", ["HYD-01", "HYD-02", "HYD-03"])
with f_col2:
    selected_cycle = st.selectbox("รอบการใช้งาน (Cycle)", ["CYCLE-03", "CYCLE-02", "CYCLE-01"])
with f_col3:
    selected_date = st.date_input("ช่วงข้อมูล", [])

# ---------------------------------------------------------
# 5. KPI CARDS (การ์ดสรุป 4 ช่อง)
# ---------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">💧 ความหนืดล่าสุด (วัดได้)</div>
        <div class="kpi-value">52.00 cSt</div>
        <div class="kpi-sub">(ที่ 2,000 ชั่วโมง)</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">🧠 ค่าพยากรณ์ (ML)</div>
        <div class="kpi-value">54.10 cSt</div>
        <div class="kpi-sub">(ที่ 2,200 ชั่วโมง)</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown("""
    <div class="kpi-card">
        <div class="kpi-title">🛢️ ค่าอ้างอิง (Fresh Oil)</div>
        <div class="kpi-value">46.00 cSt</div>
        <div class="kpi-sub">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown("""
    <div class="kpi-alarm">
        <div class="kpi-title" style="color:#B71C1C;">⚠️ สถานะปัจจุบัน</div>
        <div class="kpi-value">Alarm</div>
        <div class="kpi-sub">&nbsp;</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
# 6. MIDDLE SECTION (CHARTS: แนวโน้มความหนืด + CONTROL CHART)
# ---------------------------------------------------------
c_col1, c_col2 = st.columns([1.2, 1])

# --- กราฟซ้าย: แนวโน้มความหนืด (Viscosity Trend) ---
with c_col1:
    st.subheader("📈 แนวโน้มความหนืด (Viscosity Trend)")
    
    # ข้อมูลจำลองสำหรับกราฟ Trend
    hours = [0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000]
    measured_visc = [46.0, 46.3, 46.6, 47.1, 48.2, 48.7, 49.6, 50.8, 52.0]
    pred_hours = [2000, 2250]
    pred_visc = [52.0, 54.1]

    fig_trend = go.Figure()

    # เส้น Measured
    fig_trend.add_trace(go.Scatter(x=hours, y=measured_visc, mode='lines+markers', name='ค่าจริง (Measured)', line=dict(color='#1E88E5', width=2)))
    # เส้น Predicted
    fig_trend.add_trace(go.Scatter(x=pred_hours, y=pred_visc, mode='lines+markers', name='ค่าพยากรณ์ (Predicted)', line=dict(color='#FB8C00', dash='dash', width=2)))
    # เส้น Baseline (46.0)
    fig_trend.add_hline(y=46.0, line_dash="solid", line_color="#4CAF50", annotation_text="ค่าอ้างอิง (Baseline)")
    # เส้น Warning (+10% = 50.6)
    fig_trend.add_hline(y=50.6, line_dash="dash", line_color="#FDD835", annotation_text="Warning Limit (+10%)")
    # เส้น Alarm (+15% = 52.9)
    fig_trend.add_hline(y=52.9, line_dash="dash", line_color="#E53935", annotation_text="Alarm Limit (+15%)")

    fig_trend.update_layout(
        xaxis_title="Operating Hours (h)",
        yaxis_title="Kinematic Viscosity (cSt)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

# --- กราฟขวา: Control Chart (Individual Chart) ---
with c_col2:
    st.subheader("📊 Control Chart (Individual Chart)")
    
    sample_no = list(range(1, 10))
    sample_visc = [46.0, 46.8, 47.3, 47.9, 48.5, 49.6, 50.8, 52.0, 54.1]

    fig_control = go.Figure()
    fig_control.add_trace(go.Scatter(x=sample_no, y=sample_visc, mode='lines+markers', name='Viscosity', line=dict(color='#1E88E5', width=2)))
    
    # เส้น UCL / CL / LCL
    fig_control.add_hline(y=52.90, line_dash="dash", line_color="#E53935", annotation_text="UCL = 52.90")
    fig_control.add_hline(y=46.00, line_dash="solid", line_color="#4CAF50", annotation_text="CL = 46.00")
    fig_control.add_hline(y=39.10, line_dash="dash", line_color="#1E88E5", annotation_text="LCL = 39.10")

    fig_control.update_layout(
        xaxis_title="Sample No.",
        yaxis_title="Viscosity (cSt)",
        height=380,
        margin=dict(l=20, r=20, t=30, b=20)
    )
    st.plotly_chart(fig_control, use_container_width=True)

st.write("")

# ---------------------------------------------------------
# 7. BOTTOM SECTION (TABLES: ข้อมูล MySQL + ALERT LOG)
# ---------------------------------------------------------
b_col1, b_col2 = st.columns(2)

with b_col1:
    st.subheader("🗄️ ข้อมูลการทดสอบล่าสุด (จาก MySQL)")
    df_mysql = pd.DataFrame({
        "Sample No.": [5, 6, 7, 8, 9],
        "Operating Hours (h)": ["1,000", "1,250", "1,500", "1,750", "2,000"],
        "Viscosity (cSt)": [48.20, 48.70, 49.60, 50.80, 52.00],
        "Deviation (%)": ["+4.78", "+5.87", "+7.83", "+10.43", "+13.04"],
        "Status": ["Normal", "Warning", "Warning", "Warning", "Alarm"]
    })
    st.dataframe(df_mysql, use_container_width=True, hide_index=True)

with b_col2:
    st.subheader("🔔 การแจ้งเตือน (Alert Log)")
    df_alert = pd.DataFrame({
        "วันที่ / เวลา": ["30 เม.ย. 2025 10:24", "15 เม.ย. 2025 09:10"],
        "เครื่องจักร": ["HYD-01", "HYD-01"],
        "รายละเอียด": ["ค่าความหนืดเบี่ยงเบน +13.04% (เกินเกณฑ์ Alarm +15%)", "ค่าความหนืดเบี่ยงเบน +10.43% (เกินเกณฑ์ Warning +10%)"],
        "สถานะ": ["Alarm", "Warning"]
    })
    st.dataframe(df_alert, use_container_width=True, hide_index=True)
