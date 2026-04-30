import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('LGR_Heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_cols = joblib.load('columns.pkl')

# Set up the Streamlit app
st.title('Heart Disease Prediction by Ayesha')
st.markdown('## Input Parameters')

age = st.slider('Age', 0, 100, 54)
sex = st.selectbox('SEX',['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
resting_blood_pressure = st.number_input('Resting Blood Pressure', 94, 200, 130)
serum_cholesterol = st.number_input('Serum Cholesterol', 126, 600, 200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])
rest_ecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
max_heart_rate = st.slider('Max Heart Rate', 71, 202, 150)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', ['Y', 'N'])
oldpeak = st.slider('Oldpeak', 0.0, 6.2, 1.0)
slope = st.selectbox('Slope', ['Upsloping', 'Flat', 'Downsloping'])

if st.button('Predict'):
    raw_input = {
    'Age': age,
    'RestingBP': resting_blood_pressure,
    'Cholesterol': serum_cholesterol,
    'MaxHR': max_heart_rate,
    'Oldpeak': oldpeak,
    'FastingBS': fasting_blood_sugar,
    'ExerciseInducedAngina': exercise_induced_angina,
    'Sex_': sex,
    'ChestPainType_': chest_pain_type,
    'RestECG_': rest_ecg,
    'ST_Slop_': slope
}


    input_df = pd.DataFrame([raw_input])
    
    # One-hot encode categorical features
    input_df = pd.get_dummies(input_df)

    # Align with training columns
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    input_scaled = scaler.transform(input_df)

    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    st.write("Probability: ", proba)

    # prediction = 1 if proba[1] >= 0.3 else 0
    if prediction == 1:
        st.error('The model predicts that you have heart disease.')
    else:
        st.success('The model predicts that you do not have heart disease.')