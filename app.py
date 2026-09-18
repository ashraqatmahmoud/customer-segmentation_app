import streamlit as st
import pandas as pd
import joblib

# 1. Page Configuration
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide"
)

# 2. Load the Saved Model
@st.cache_resource
def load_model():
    artifacts = joblib.load("customer_segment_model.pkl")
    return artifacts

try:
    artifacts = load_model()
    model = artifacts["model"]
    features = artifacts["features"]
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# 3. Main Dashboard Header
st.title("🛍️ Customer Personality Analysis & Segmentation Dashboard")
st.markdown("This application uses **K-Means Clustering** for customer segmentation and a **Logistic Regression** classifier to predict customer segments based on behavioral and demographic features.")
st.markdown("---")

# 4. Sidebar Inputs for New Customer Prediction
st.sidebar.header("🔍 New Customer Inputs")

age = st.sidebar.slider("Age", 18, 70, 35)
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
gender_binary = 1 if gender == "Male" else 0

purchase_amount = st.sidebar.number_input("Purchase Amount (USD)", min_value=10, max_value=200, value=60)
review_rating = st.sidebar.slider("Review Rating", 1.0, 5.0, 4.0)
previous_purchases = st.sidebar.number_input("Previous Purchases", min_value=0, max_value=100, value=25)

# 5. Layout: Two Columns (Prediction & Segment Exploration)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("🎯 Customer Segment Prediction")
    st.write("Click the button below to classify the customer into their respective behavioral segment.")
    
    if st.button("Predict Segment", type="primary", use_container_width=True):
        # Prepare DataFrame
        new_customer = pd.DataFrame([{
            "Age": age,
            "Gender_Binary": gender_binary,
            "Purchase Amount (USD)": purchase_amount,
            "Review Rating": review_rating,
            "Previous Purchases": previous_purchases
        }])
        
        new_customer = new_customer[features]
        
        # Prediction
        prediction = model.predict(new_customer)
        segment_result = prediction[0]
        
        # Display Result Card
        st.markdown("---")
        if segment_result == 0:
            st.success("✅ **Result: Customer Segment 0**")
            st.info("💡 **Profile Insights:** Primarily Female-dominant segment with average spending around $60 and stable purchase history.")
        else:
            st.warning("⭐ **Result: Customer Segment 1**")
            st.info("💡 **Profile Insights:** Primarily Male-dominant segment with comparable purchasing behavior and high engagement.")

with col2:
    st.subheader("📊 Explore Segment Profiles")
    st.write("Select a segment below to view its core statistical profile derived from our clustering analysis.")
    
    selected_segment = st.selectbox("Choose Segment to Inspect:", [0, 1])
    
    if selected_segment == 0:
        st.markdown("### 📌 Segment 0 Profile")
        st.markdown("- **Dominant Gender:** Female (Binary: 0.0)")
        st.markdown("- **Average Age:** ~44 years")
        st.markdown("- **Average Purchase Amount:** ~$60.25 USD")
        st.markdown("- **Review Rating Mean:** ~3.74 / 5.0")
        st.markdown("- **Previous Purchases Mean:** ~24.6 transactions")
    else:
        st.markdown("### 📌 Segment 1 Profile")
        st.markdown("- **Dominant Gender:** Male (Binary: 1.0)")
        st.markdown("- **Average Age:** ~44.1 years")
        st.markdown("- **Average Purchase Amount:** ~$59.54 USD")
        st.markdown("- **Review Rating Mean:** ~3.75 / 5.0")
        st.markdown("- **Previous Purchases Mean:** ~25.7 transactions")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>Developed for Project Defense | Customer Segmentation Project</p>", unsafe_allow_html=True)
