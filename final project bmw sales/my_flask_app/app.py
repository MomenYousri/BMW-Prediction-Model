from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# --- تصحيح الخطأ هنا: تعريف المتغير أولاً ---
model = None 

# محاولة تحميل الموديل
try:
    if os.path.exists('bmw_model.pkl'):
        model = joblib.load('bmw_model.pkl')
        print("✅ Model loaded successfully!")
    else:
        print("❌ Error: 'bmw_model.pkl' not found. Please check file location.")
except Exception as e:
    print(f"❌ Error loading model: {e}")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    
    if request.method == 'POST':
        try:
            # 1. فحص ما إذا كان الموديل قد تم تحميله بنجاح
            if model is None:
                return render_template('index.html', prediction="Error: Model file could not be loaded.")

            # 2. استقبال البيانات
            data = {
                'Model': request.form.get('Model'),
                'Year': int(request.form.get('Year')),
                'Region': request.form.get('Region'),
                'Color': request.form.get('Color'),
                'Fuel_Type': request.form.get('Fuel_Type'),
                'Transmission': request.form.get('Transmission'),
                'Engine_Size_L': float(request.form.get('Engine_Size_L')),
                'Mileage_KM': float(request.form.get('Mileage_KM')),
                'Price_USD': float(request.form.get('Price_USD'))
            }
            
            # 3. تحويل البيانات إلى DataFrame
            df = pd.DataFrame([data])
            
            # 4. إعادة ترتيب الأعمدة (مهم جداً)
            expected_order = [
                'Model', 
                'Year', 
                'Region', 
                'Color', 
                'Fuel_Type', 
                'Transmission', 
                'Engine_Size_L', 
                'Mileage_KM', 
                'Price_USD'
            ]
            
            # إعادة الترتيب لتطابق تدريب الموديل
            df = df[expected_order]
            
            # 5. التنبؤ
            result = model.predict(df)
            prediction = result[0]

        except Exception as e:
            print(f"Prediction Error: {e}")
            prediction = f"Error: {e}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)