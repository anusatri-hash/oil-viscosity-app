import streamlit as st
import joblib
import numpy as np

# โหลดโมเดล
try:
    model = joblib.load('model.pkl')
except Exception as e:
    st.error(f"ไม่สามารถโหลดโมเดลได้: {e}")

st.title('🔮 ระบบพยากรณ์ข้อมูลน้ำมันหล่อลื่น (Predictive Maintenance)')

st.write('กรอกข้อมูลคุณสมบัติเพื่อพยากรณ์ค่า Measured Viscosity at 40°C')

# โมเดลนี้เทรนด้วย 4 features: ['operating_hours', 'fresh_visc_40c', 'fresh_visc_100c', 'measured_visc_100c']
operating_hours = st.number_input('Operating Hours (ชั่วโมงการทำงาน)', value=600.0, step=10.0)
fresh_visc_40c = st.number_input('Fresh Viscosity at 40°C', value=46.0, step=0.1)
fresh_visc_100c = st.number_input('Fresh Viscosity at 100°C', value=6.8, step=0.1)
measured_visc_100c = st.number_input('Measured Viscosity at 100°C', value=6.5, step=0.1)

if st.button('ทำนายผล'):
    # จัดเรียง features ให้ตรงกับที่ใช้เทรนโมเดล
    features = np.array([[operating_hours, fresh_visc_40c, fresh_visc_100c, measured_visc_100c]])
    prediction = model.predict(features)
    st.success(f'ผลการทำนาย Measured Viscosity at 40°C: {prediction[0]:.4f} cSt')
