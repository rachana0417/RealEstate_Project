import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------------
# LOAD MODELS (correct paths for Streamlit Cloud)
# -------------------------------------------------------
rf_clf = joblib.load("rf_classifier.pkl")
rf_reg = joblib.load("rf_regression.pkl")

# -------------------------------------------------------
# LOAD TRAINING COLUMN NAMES
# -------------------------------------------------------
df_train = pd.read_csv("cleaned_data.csv")  # File must be in repo root

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
# CREATE INPUT ROW USING TRAINING COLUMNS
# -------------------------------------------------------
def make_input_df():
    input_dict = {}

    # Fill all model-required columns with default 0
    for col in X_class_cols:
        input_dict[col] = 0

    # Update only user inputs
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
# PREDICT
# -------------------------------------------------------
if st.button("Predict"):

    input_df = make_input_df()

    # ---- CLASSIFICATION PREDICTION ----
    invest_pred = rf_clf.predict(input_df)[0]

    # ---- REGRESSION PREDICTION ----
    reg_input = pd.DataFrame([[0] * len(X_reg_cols)], columns=X_reg_cols)

    # copy allowed inputs
    for col in ["Size_in_SqFt", "Price_in_Lakhs", "BHK"]:
        if col in reg_input.columns:
            reg_input[col] = input_df[col]

    future_price = rf_reg.predict(reg_input)[0]

    st.success(f"Investment Decision: **{'Good' if invest_pred == 1 else 'Bad'}**")
    st.info(f"Predicted Future Price (5Y): **₹{future_price:.2f} Lakhs**")
