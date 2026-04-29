import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("Data/heart.csv")

# Rename columns
df.rename(columns={
    "sex": "gender",
    "cp": "chest_pain_type",
    "trestbps": "resting_blood_pressure",
    "chol": "cholesterol",
    "fbs": "fasting_blood_sugar",
    "restecg": "resting_ecg",
    "thalach": "max_heart_rate",
    "exang": "exercise_induced_angina",
    "oldpeak": "st_depression",
    "slope": "st_slope",
    "ca": "num_major_vessels",
    "thal": "thalassemia",
    "target": "heart_disease"
}, inplace=True)

# Split
X = df.drop("heart_disease", axis=1)
y = df["heart_disease"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)

# UI
st.title("❤️ Heart Disease Prediction App")

age = st.number_input("Age", 20, 100)
gender = st.selectbox("Gender", [0,1])
chol = st.number_input("Cholesterol", 100, 400)
cp = st.selectbox("Chest Pain Type", [0,1,2,3])
bp = st.number_input("Blood Pressure", 80, 200)
thalach = st.number_input("Max Heart Rate", 60, 200)
oldpeak = st.number_input("ST Depression", 0.0, 6.0)

if st.button("Predict"):
    input_data = [[
    age, gender, cp, bp, chol, 0, 1,
    thalach, 0, oldpeak, 1, 0, 2
]]
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease")