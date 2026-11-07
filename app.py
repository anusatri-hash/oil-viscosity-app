import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

# ---------------------------------------------------------
# 1. การตั้งค่าหน้าเว็บ (Page Configuration)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Oil Viscosity PdM Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Predictive Maintenance Dashboard: Viscosity Trend & Alert")
st.markdown("ระบบติดตามแนวโน้มความหนืดน้ำมันเครื่องจักรและการแจ้งเตือนอัตโนมัติ")

# ---------------------------------------------------------
# 2. โหลดโมเดล Machine Learning
# ---------------------------------------------------------
@st.cache_resource
def load_ml_model():
    try:
        return joblib.load('model.pkl')
    except Exception as e:
        st.warning("⚠️ ไม่พบไฟล์ model.pkl ระบบจะใช้สมการประเมินเบื้องต้นแทน")
        return None

model = load_ml_model()

# ---------------------------------------------------------
# 3. ส่วนรับข้อมูลจากผู้ใช้ / อุปกรณ์ (Sidebar Inputs)
# ---------------------------------------------------------
st.sidebar.header("📥 ป้อนข้อมูลวัดค่าความหนืด")
machine_id = st.sidebar.textbox = st.sidebar.text_input("รหัสเครื่องจักร (Machine ID)", "MC-01")
operating_hours = st.sidebar.number_input("ชั่วโมงการทำงานสะสม (Operating Hours)", min_value=0, value=1200, step=50)
fresh_visc_40 = st.sidebar.number_input("ความหนืดน้ำมันใหม่ 40°C (Baseline cSt)", min_value=1.0, value=68.0, step=1.0)
fresh_visc_100 = st.sidebar.number_input("ความหนืดน้ำมันใหม่ 100°C (cSt)", min_value=1.0, value=8.5, step=0.5)
measured_visc_100 = st.sidebar.number_input("ความหนืดที่วัดได้ 100°C (cSt)", min_value=1.0, value=8.1, step=0.1)

# ทำนายค่า Viscosity 40°C
if model is not None:
    X_new = np.array([[operating_hours, fresh_visc_40, fresh_visc_100, measured_visc_100]])
    predicted_visc_40 = model.predict(X_new)[0]
else:
    # สมการสำรองกรณีไม่มีไฟล์ model.pkl
    predicted_visc_40 = fresh_visc_40 * (1 - (operating_hours / 10000) * 0.15)

# คำนวณความเบี่ยงเบน (%)
dev_percent = abs((predicted_visc_40 - fresh_visc_40) / fresh_visc_40) * 100

# ---------------------------------------------------------
# 4. ตรวจสอบสถานะและ Alert Logic
# ---------------------------------------------------------
warning_limit_low = fresh_visc_40 * 0.90   # -10%
warning_limit_high = fresh_visc_40 * 1.10  # +10%
alarm_limit_low = fresh_visc_40 * 0.85     # -15%
alarm_limit_high = fresh_visc_40 * 1.15    # +15%

if dev_percent >= 15:
    status_label = "🔴 CRITICAL ALARM"
    status_color = "red"
    alert_msg = "ความหนืดเบี่ยงเบนเกิน ±15% (สภาพน้ำมันเสื่อมสภาพวิกฤต) ต้องวางแผนเปลี่ยนถ่ายน้ำมันทันที!"
elif dev_percent >= 10:
    status_label = "🟡 WARNING"
    status_color = "orange"
    alert_msg = "ความหนืดเบี่ยงเบนช่วง ±10% ถึง ±15% (เริ่มเสื่อมสภาพ) ควรเพิ่มความถี่ในการเก็บตัวอย่าง"
else:
    status_label = "🟢 NORMAL"
    status_color = "green"
    alert_msg = "สภาพน้ำมันปกติ ทำงานได้ตามปกติ"

# ---------------------------------------------------------
# 5. การแสดงผลส่วนที่ 1: Executive Summary & Status Cards (KPIs)
# ---------------------------------------------------------
st.subheader("1. สรุปสถานะภาพรวมเครื่องจักร (Executive Summary)")

col1, col2, col3, col4 = st.columns(4)
col1.metric("รหัสเครื่องจักร", machine_id)
col2.metric("ชั่วโมงการทำงานสะสม", f"{operating_hours:,} ชม.")
col3.metric("ความหนืดพยากรณ์ (40°C)", f"{predicted_visc_40:.2f} cSt", f"{dev_percent:.1f}% Deviation")
col4.metric("สถานะปัจจุบัน", status_label)

st.divider()

# ---------------------------------------------------------
# 6. การแสดงผลส่วนที่ 2: Time-Series Trend Charts & Control Limits
# ---------------------------------------------------------
st.subheader("2. กราฟแสดงแนวโน้มความหนืด (Viscosity Trend & Control Limits)")

# จำลองข้อมูลแนวโน้มตามชั่วโมงทำงาน (Lifecycle Trend)
hours_range = np.linspace(0, max(2000, operating_hours + 500), 50)
if model is not None:
    X_sim = np.column_stack([
        hours_range,
        np.full_like(hours_range, fresh_visc_40),
        np.full_like(hours_range, fresh_visc_100),
        np.full_like(hours_range, measured_visc_100)
    ])
    trend_visc = model.predict(X_sim)
else:
    trend_visc = fresh_visc_40 * (1 - (hours_range / 10000) * 0.15)

# สร้างกราฟ Plotly
fig = go.Figure()

# เส้นเทรนด์ความหนืด
fig.add_trace(go.Scatter(x=hours_range, y=trend_visc, mode='lines', name='Predicted Viscosity (40°C)', line=dict(color='blue', width=3)))

# จุดวัดปัจจุบัน
fig.add_trace(go.Scatter(x=[operating_hours], y=[predicted_visc_40], mode='markers', name='Current Status', marker=dict(color=status_color, size=14, symbol='diamond')))

# เส้น Control Limits
fig.add_hline(y=fresh_visc_40, line_dash="dash", line_color="gray", annotation_text="Baseline (Fresh Oil)")
fig.add_hline(y=warning_limit_high, line_dash="dot", line_color="orange", annotation_text="Warning Limit (+10%)")
fig.add_hline(y=warning_limit_low, line_dash="dot", line_color="orange", annotation_text="Warning Limit (-10%)")
fig.add_hline(y=alarm_limit_high, line_dash="solid", line_color="red", annotation_text="Alarm Limit (+15%)")
fig.add_hline(y=alarm_limit_low, line_dash="solid", line_color="red", annotation_text="Alarm Limit (-15%)")

fig.update_layout(
    xaxis_title="Operating Hours (ชั่วโมง)",
    yaxis_title="Viscosity at 40°C (cSt)",
    height=450,
    margin=dict(l=20, r=20, t=30, b=20)
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------
# 7. การแสดงผลส่วนที่ 3: Automated Notification & Log Panel
# ---------------------------------------------------------
st.subheader("3. ระบบแจ้งเตือนอัตโนมัติ (Automated Notification & Log Panel)")

if status_color == "red":
    st.error(f"🚨 **[ALARM LOG]** {machine_id} - {alert_msg}")
elif status_color == "orange":
    st.warning(f"⚠️ **[WARNING LOG]** {machine_id} - {alert_msg}")
else:
    st.success(f"✅ **[NORMAL LOG]** {machine_id} - {alert_msg}")

# ตารางประวัติการแจ้งเตือน (Alert Log Table)
log_data = pd.DataFrame([
    {"Timestamp": "2026-11-07 10:00", "Machine ID": machine_id, "Operating Hours": operating_hours, "Viscosity (40°C)": round(predicted_visc_40, 2), "Level": status_label, "Action Required": alert_msg},
    {"Timestamp": "2026-11-01 09:00", "Machine ID": machine_id, "Operating Hours": max(0, operating_hours - 200), "Viscosity (40°C)": round(fresh_visc_40 * 0.94, 2), "Level": "🟢 NORMAL", "Action Required": "ตรวจสอบตามรอบปกติ"},
])

st.dataframe(log_data, use_container_width=True)
