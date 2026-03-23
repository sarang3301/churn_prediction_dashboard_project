
import streamlit as st
import pandas as pd
from src.predict import predict
st.title("📊 Customer Churn Prediction")

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure", 0, 72)
    monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0)
    total_charges = st.number_input("Total Charges", 0.0, 10000.0)

with col2:
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

# Button (ALWAYS visible now)
st.markdown("---")
predict_btn = st.button("🔍 Predict")

# Create input dict
input_dict = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
}

if contract == "One year":
    input_dict["Contract_One year"] = 1
elif contract == "Two year":
    input_dict["Contract_Two year"] = 1

if internet == "Fiber optic":
    input_dict["InternetService_Fiber optic"] = 1
elif internet == "No":
    input_dict["InternetService_No"] = 1

input_df = pd.DataFrame([input_dict])

# Predict
if predict_btn:
    result = predict(input_df)

    if result[0] == 1:
        st.error("⚠️ Customer will churn")
    else:
        st.success("✅ Customer will stay")