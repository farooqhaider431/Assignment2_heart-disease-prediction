import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️")
st.title("❤️ Heart Disease Prediction System")
st.caption("AIC354 Assignment 2. For learning only, not medical advice.")

@st.cache_resource
def load():
    return joblib.load("heart_disease_model.pkl")

saved = load()
model, le_sex, le_cp = saved["model"], saved["le_sex"], saved["le_cp"]

age = st.slider("Age", 20, 90, 50)
sex = st.selectbox("Sex", list(le_sex.classes_))
cp = st.selectbox("Chest pain type", list(le_cp.classes_))
thalach = st.slider("Maximum heart rate achieved", 60, 210, 150)

if st.button("Predict"):
    row = pd.DataFrame([[age, le_sex.transform([sex])[0], le_cp.transform([cp])[0], thalach]],
                       columns=["age", "sex", "cp", "thalach"])
    if model.predict(row)[0] == 1:
        st.error("Heart disease is likely. Please consult a doctor.")
    else:
        st.success("Heart disease is not likely.")
