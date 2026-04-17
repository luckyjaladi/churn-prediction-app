import streamlit as st
import pandas as pd
import pickle

# Title
st.title("Customer Churn Prediction")

# ------------------------
# LOAD MODEL FIRST ✅
# ------------------------
with open("churn_model.pkl", "rb") as f:
    model = pickle.load(f)

# ------------------------
# INPUT FIELDS
# ------------------------
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 50.0)

contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Convert contract to numeric
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
contract_val = contract_map[contract]

# Create empty row with all features
input_data = pd.DataFrame([[0]*len(model.feature_names_in_)], columns=model.feature_names_in_)

# Fill only known inputs
input_data['tenure'] = tenure
input_data['MonthlyCharges'] = monthly_charges
input_data['Contract'] = contract_val

# ------------------------
# PREDICTION
# ------------------------
if st.button("Predict"):
    prediction = model.predict(input_data)
    prob = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error("Customer likely to churn")
    else:
        st.success("Customer likely to stay")

    st.progress(float(prob))
    st.write(f"Churn Probability: {prob:.2f}")
