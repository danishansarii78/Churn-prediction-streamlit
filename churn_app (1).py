import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and feature columns
model = pickle.load(open("churn_model.pkl", "rb"))
columns = pickle.load(open("churn_features.pkl", "rb"))

st.title("🔍 Customer Churn Prediction App")

# Collect input from user
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Has Partner", ["Yes", "No"])
dependents = st.selectbox("Has Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.number_input("Monthly Charges", value=70.0)
total_charges = st.number_input("Total Charges", value=2000.0)

# Prepare manual encoding for necessary features
data = {
    'tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'SeniorCitizen': 1 if senior == "Yes" else 0,
    'gender_Male': 1 if gender == "Male" else 0,
    'Partner_Yes': 1 if partner == "Yes" else 0,
    'Dependents_Yes': 1 if dependents == "Yes" else 0,
    'Contract_One year': 1 if contract == "One year" else 0,
    'Contract_Two year': 1 if contract == "Two year" else 0,
}

# Add missing columns (because original df has 20+ encoded columns)
input_df = pd.DataFrame([data])
for col in columns:
    if col not in input_df.columns:
        input_df[col] = 0  # fill missing with 0
input_df = input_df[columns]  # match training order

# Predict
if st.button("Predict Churn"):
    prob = model.predict_proba(input_df)[0][1]
    st.metric("Churn Probability", f"{prob:.2%}")
    if prob > 0.5:
        st.warning("⚠️ High likelihood of churn!")
    else:
        st.success("✅ Low risk of churn")
