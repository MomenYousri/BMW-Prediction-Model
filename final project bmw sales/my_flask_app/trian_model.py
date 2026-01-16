import pandas as pd
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

# 1. إنشاء بيانات وهمية بنفس هيكل بياناتك (لتعريف الموديل بالأعمدة)
data = {
    'Model': ['3 Series', '5 Series', 'X3', 'X5', 'M3', 'i8', 'X1', '7 Series', 'X6', 'M5', 'i3', 'X3'],
    'Year': [2020, 2021, 2019, 2022, 2023, 2018, 2020, 2021, 2022, 2023, 2019, 2020],
    'Region': ['Middle East', 'Europe', 'Asia', 'North America', 'Europe', 'Asia', 'Africa', 'Middle East', 'Europe', 'Europe', 'Asia', 'Middle East'],
    'Color': ['Black', 'White', 'Blue', 'Grey', 'Red', 'Silver', 'White', 'Black', 'Black', 'Blue', 'White', 'Red'],
    'Fuel_Type': ['Petrol', 'Diesel', 'Petrol', 'Hybrid', 'Petrol', 'Electric', 'Petrol', 'Petrol', 'Diesel', 'Petrol', 'Electric', 'Petrol'],
    'Transmission': ['Automatic', 'Automatic', 'Automatic', 'Automatic', 'Manual', 'Automatic', 'Automatic', 'Automatic', 'Automatic', 'Automatic', 'Automatic', 'Manual'],
    'Engine_Size_L': [2.0, 3.0, 2.0, 3.0, 3.0, 1.5, 1.5, 4.4, 3.0, 4.4, 0.0, 2.5],
    'Mileage_KM': [50000, 30000, 60000, 10000, 5000, 20000, 80000, 15000, 10000, 2000, 40000, 35000],
    'Price_USD': [30000, 50000, 40000, 60000, 80000, 70000, 25000, 90000, 65000, 100000, 20000, 45000],
    # عمود الهدف (النتيجة المتوقعة)
    'Class': ['Low', 'High', 'Low', 'High', 'High', 'High', 'Low', 'High', 'High', 'High', 'Low', 'Low']
}

df = pd.DataFrame(data)

# فصل المدخلات عن المخرجات
X = df.drop(columns=['Class'])
y = df['Class']

# 2. تحديد الأعمدة النصية والرقمية
categorical_features = ['Model', 'Region', 'Color', 'Fuel_Type', 'Transmission']
numerical_features = ['Year', 'Engine_Size_L', 'Mileage_KM', 'Price_USD']

# 3. بناء المعالج (يحول النصوص لأرقام والأرقام لمقياس موحد)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# 4. دمج المعالج مع الموديل في Pipeline واحد
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(random_state=42))  # نستخدم SVC كما ظهر في ملفك
])

# 5. تدريب الموديل
print("Training model...")
pipeline.fit(X, y)

# 6. حفظ الموديل الصحيح
joblib.dump(pipeline, 'bmw_model.pkl')
print("✅ Success! New 'bmw_model.pkl' created with preprocessing pipeline.")