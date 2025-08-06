# --- START OF FILE app.py (Final Corrected Version) ---

import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# --- Load saved model, columns, and scaler ---
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "model.pkl")
columns_path = os.path.join(base_dir, "columns.pkl")
scaler_path = os.path.join(base_dir, "scaler.pkl")

model = joblib.load(model_path)
columns = joblib.load(columns_path)
scaler = joblib.load(scaler_path)

# --- App Title ---
st.title("🏦 Loan Approval Predictor")

# --- Form Input (CoapplicantIncome is REMOVED) ---
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
# coapplicant_income input is now fully removed
loan_amount = st.number_input("Loan Amount ($)", min_value=0, value=150)
loan_amount_term = st.number_input("Loan Amount Term (in days)", min_value=0, value=360)
credit_history = st.selectbox("Credit History (1.0 = All Debts Paid)", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Rural', 'Semiurban', 'Urban'])

if st.button("Predict"):
    # --- Preprocessing the input data ---

    # 1. Create a dictionary from the user input (without CoapplicantIncome)
    input_dict = {
        'Gender': [gender],
        'Married': [married],
        'Education': [education],
        'Self_Employed': [self_employed],
        'ApplicantIncome': [applicant_income],
        'LoanAmount': [loan_amount],
        'Loan_Amount_Term': [loan_amount_term],
        'Credit_History': [credit_history],
        'Dependents': [dependents],
        'Property_Area': [property_area]
    }
    
    # 2. Create a DataFrame from the dictionary
    input_df = pd.DataFrame(input_dict)

    # 3. Encode categorical features
    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Married'] = input_df['Married'].map({'Yes': 1, 'No': 0})
    input_df['Education'] = input_df['Education'].map({'Graduate': 0, 'Not Graduate': 1})
    input_df['Self_Employed'] = input_df['Self_Employed'].map({'Yes': 1, 'No': 0})

    # 4. One-hot encode Dependents and Property_Area
    input_df = pd.get_dummies(input_df, columns=['Dependents', 'Property_Area'], drop_first=True)

    # 5. Scale the numerical features (this will now work correctly)
    numerical_cols = ['ApplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # 6. Reindex columns to match the training data
    input_data = input_df.reindex(columns=columns, fill_value=0)

    # --- Prediction ---
    st.write("---")
    st.write("#### Input Data After Processing:")
    st.dataframe(input_data)
    
    prediction = model.predict(input_data)[0]


    if prediction == 1:
        st.success("## ✅ Loan Approved!")
    else:
        st.error("## ❌ Loan Not Approved.")