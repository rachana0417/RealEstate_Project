# 🏠 Real Estate Investment Predictor  
An end-to-end Machine Learning + Streamlit application that predicts:

1. Whether a property is a good investment (Classification)
2. The estimated future price of the property (Regression)

This project is built using:
- Python  
- Scikit-Learn  
- Random Forest Algorithm  
- Joblib (Model Compression)  
- Streamlit (Deployment UI)

---

## 🚀 Project Overview  
Real estate investment involves risk due to price fluctuations, location factors, security, amenities, and market demand.  
This system analyzes property features and predicts:

### ✔ Investment Decision (Good / Bad)  
Using a **Random Forest Classifier**.

### ✔ Future Price after 5 Years  
Using a **Random Forest Regressor**.

---

## 📂 Project Structure

RealEstate_Project/
│
├── streamlit_app/
│ ├── app.py
│ ├── rf_classifier_compressed.pkl.gz
│ ├── rf_regression_compressed.pkl.gz
│ ├── requirements.txt
│
├── models/
│ ├── Original training notebooks and scripts
│
└── data/
├── cleaned_data.csv
├── india_housing_prices.csv

---

## 🧠 Machine Learning Models Used

### 1️⃣ **Random Forest Classifier**
Predicts investment decision using:
- Size_in_SqFt  
- Price_in_Lakhs  
- BHK  
- Security  
- Parking_Space  
- Facing  

### 2️⃣ **Random Forest Regressor**
Predicts future price using:
- Size_in_SqFt  
- Price_in_Lakhs  
- BHK  

The models were compressed using:

```python
joblib.dump(model, "file.pkl.gz", compress=("gzip", 9))
🎨 Streamlit App Features

Modern UI

Numeric and dropdown inputs

Auto-handling of missing model features

Outputs:

Investment Decision

Predicted Future Price

▶ How to Run the App
Step 1: Install requirements
pip install -r requirements.txt

Step 2: Run Streamlit
streamlit run app.py


Your app runs at:

http://localhost:8501

🛠 Technologies Used
Component	Technology
Frontend	Streamlit
ML Models	Random Forest
Language	Python
Compression	Joblib + gzip
Deployment	Streamlit Cloud (optional)
📈 Outputs
✔ Investment Decision: Good / Bad
✔ Predicted Future Price (5 Years): ₹ XX Lakhs
