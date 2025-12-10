import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------
rf_clf = joblib.load("../models/rf_classifier.pkl")
rf_reg = joblib.load("../models/rf_regression.pkl")

# -------------------------------------------------------
# LOAD TRAINING COLUMN NAMES (IMPORTANT)
# -------------------------------------------------------
# Load the cleaned csv file to get correct feature columns
df_train = pd.read_csv("../data/cleaned_data.csv")

# For classification
X_class_cols = df_train.drop(
    ["Good_Investment", "Future_Price_5Y", "ID"], axis=1, errors="ignore"
).columns.tolist()

# For regression
X_reg_cols = df_train.drop(
    ["Future_Price_5Y", "ID"], axis=1, errors="ignore"
).columns.tolist()

# -------------------------------------------------------
# STREAMLIT UI
# -------------------------------------------------------
st.title("🏠 Real Estate Investment Predictor")

st.subheader("Enter Property Features")

size = st.number_input("Size in SqFt", min_value=100, max_value=10000, value=1000)
price = st.number_input("Price in Lakhs", min_value=5.0, max_value=5000.0, value=50.0)
bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)
security = st.selectbox("Security (0/1)", [0, 1])
parking = st.selectbox("Parking Space (0/1)", [0, 1])
facing = st.number_input("Facing (0–3)", min_value=0, max_value=3, value=1)

# -------------------------------------------------------
# CREATE INPUT ROW WITH TRAINING COLUMNS
# -------------------------------------------------------
def make_input_df():
    input_dict = {}

    # Fill all columns with 0 initially
    for col in X_class_cols:
        input_dict[col] = 0

    # Update only the features user entered
    if "Size_in_SqFt" in input_dict:
        input_dict["Size_in_SqFt"] = size

    if "Price_in_Lakhs" in input_dict:
        input_dict["Price_in_Lakhs"] = price

    if "BHK" in input_dict:
        input_dict["BHK"] = bhk

    if "Security" in input_dict:
        input_dict["Security"] = security

    if "Parking_Space" in input_dict:
        input_dict["Parking_Space"] = parking

    if "Facing" in input_dict:
        input_dict["Facing"] = facing

    return pd.DataFrame([input_dict])


# -------------------------------------------------------
# PREDICT BUTTON
# -------------------------------------------------------
if st.button("Predict"):

    input_df = make_input_df()

    # CLASSIFICATION (Good Investment 0/1)
    invest_pred = rf_clf.predict(input_df)[0]

    # REGRESSION (Future Price)
    # prepare regression df with correct columns
    reg_input = pd.DataFrame([[0] * len(X_reg_cols)], columns=X_reg_cols)

    for col in ["Size_in_SqFt", "Price_in_Lakhs", "BHK"]:
        if col in reg_input.columns:
            reg_input[col] = input_df[col]

    future_price = rf_reg.predict(reg_input)[0]

    st.success(f"Investment Decision: **{'Good' if invest_pred == 1 else 'Bad'}**")
    st.info(f"Predicted Future Price (5Y): **₹{future_price:.2f} Lakhs**")
