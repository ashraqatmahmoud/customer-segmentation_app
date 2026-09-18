
import streamlit as st
import pandas as pd
import joblib

# عنوان الصفحة وتنسيقها
st.set_page_config(page_title="Customer Segmentation App", page_icon="🛍️")

# تحميل النموذج المحفوظ
@st.cache_resource
def load_model():
    artifacts = joblib.load("customer_segment_model.pkl")
    return artifacts

artifacts = load_model()
model = artifacts["model"]
features = artifacts["features"]

# واجهة التطبيق على الويب
st.title("🛍️ Customer Segmentation & Prediction App")
st.write("هذا التطبيق يتوقع شريحة العميل (Customer Segment) بناءً على سلوكه وخصائصه باستخدام نموذج Logistic Regression.")

st.sidebar.header("أدخل بيانات العميل الجديد")

# أدوات الإدخال (Sliders & Inputs)
age = st.sidebar.slider("Age (العمر)", 18, 70, 30)
gender = st.sidebar.selectbox("Gender (الجنس)", ["Male", "Female"])
gender_binary = 1 if gender == "Male" else 0

purchase_amount = st.sidebar.number_input("Purchase Amount (USD)", min_value=10, max_value=200, value=75)
review_rating = st.sidebar.slider("Review Rating", 1.0, 5.0, 4.5)
previous_purchases = st.sidebar.number_input("Previous Purchases", min_value=0, max_value=100, value=10)

# زر التنبؤ
if st.sidebar.button("توقع الشريحة (Predict)"):
    # تجهيز الداتا في شكل جدول بنفس الترتيب
    new_customer = pd.DataFrame([{
        "Age": age,
        "Gender_Binary": gender_binary,
        "Purchase Amount (USD)": purchase_amount,
        "Review Rating": review_rating,
        "Previous Purchases": previous_purchases
    }])
    
    new_customer = new_customer[features]
    
    # تنفيذ التنبؤ
    prediction = model.predict(new_customer)
    segment_result = prediction[0]
    
    # إظهار النتيجة
    st.success(f"✅ العميل ينتمي بنجاح إلى: **Customer Segment {segment_result}**")
