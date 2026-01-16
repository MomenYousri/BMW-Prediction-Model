import streamlit as st
import pandas as pd
import joblib

# 1. تحميل الموديل والـ Encoders المحفوظة
model = joblib.load('bmw_model.pkl')
encoders = joblib.load('encoders.pkl')

# إعداد شكل الصفحة
st.set_page_config(page_title="BMW Sales Prediction", layout="centered")

st.title("🚗 BMW Sales Classification AI")
st.write("أدخل بيانات السيارة لتوقع تصنيف المبيعات (High/Low)")

# 2. إنشاء واجهة لإدخال البيانات (Input Fields)
col1, col2 = st.columns(2)

with col1:
    # القوائم المنسدلة للبيانات النصية (نستخرج الخيارات من الـ Encoders المحفوظة)
    selected_model = st.selectbox("Model", encoders['Model'].classes_)
    selected_region = st.selectbox("Region", encoders['Region'].classes_)
    selected_color = st.selectbox("Color", encoders['Color'].classes_)
    selected_fuel = st.selectbox("Fuel Type", encoders['Fuel_Type'].classes_)
    selected_transmission = st.selectbox("Transmission", encoders['Transmission'].classes_)

with col2:
    # خانات الأرقام
    year = st.number_input("Year", min_value=2000, max_value=2025, value=2018)
    engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, value=2.0)
    mileage = st.number_input("Mileage (KM)", min_value=0, value=50000)
    price_usd = st.number_input("Price (USD)", min_value=0, value=30000)
    sales_volume = st.number_input("Sales Volume", min_value=0, value=100)

# 3. زر التوقع
if st.button("Predict Classification"):
    
    # تحضير البيانات بنفس ترتيب التدريب
    # أولاً: حساب الميزة الإضافية (Total Price)
    total_price = price_usd * sales_volume
    
    # ثانياً: تجميع البيانات في DataFrame
    input_data = pd.DataFrame({
        'Model': [selected_model],
        'Year': [year],
        'Region': [selected_region],
        'Color': [selected_color],
        'Fuel_Type': [selected_fuel],
        'Transmission': [selected_transmission],
        'Engine_Size_L': [engine_size],
        'Mileage_KM': [mileage],
        'Price_USD': [price_usd],
        'Sales_Volume': [sales_volume],
        'Total price': [total_price]
    })

    # ثالثاً: تحويل النصوص إلى أرقام باستخدام الـ Encoders المحفوظة
    try:
        for col, le in encoders.items():
            input_data[col] = le.transform(input_data[col])
        
        # رابعاً: التوقع باستخدام الموديل
        prediction = model.predict(input_data)
        
        # خامساً: عرض النتيجة
        if prediction[0] == 'High':
            st.success(f"📈 Prediction: **High Sales**")
        else:
            st.warning(f"📉 Prediction: **Low Sales**")
            
    except Exception as e:
        st.error(f"Error during prediction: {e}")