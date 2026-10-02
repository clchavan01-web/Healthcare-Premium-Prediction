import streamlit as st
import joblib
import pandas as pd

model = joblib.load('healthcare_model.pkl')

st.set_page_config(page_title="Healthcare premium predictor model")
st.title("Healthcare premium predictor model")
st.write("Enter your details below to estimate your annual insurance premium")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Select Your Age", 18, 100, 30)
    bmi = st.number_input("Enter Your BMI", 10.0, 50.0, 25.0)

with col2:
    smoker_choice = st.selectbox("Are you a smoker?", ["yes", "no"])

if st.button("Calculate Premium"):
    smoker = 1 if smoker_choice == "yes" else 0
    input_df = pd.DataFrame([[age, bmi, smoker]], columns=['age', 'BMI', 'smoker'])
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated annual premium: ₹{prediction:.2f}")
    st.info("Note: This is a basic ML model estimate based on training data")