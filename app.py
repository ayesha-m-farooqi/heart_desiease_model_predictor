import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('KNN_Heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_cols = joblib.load('columns.pkl')

# Set up the Streamlit app
st.title('Heart Disease Prediction by Ayesha')
st.markdown('## Input Parameters')

age = st.slider('Age', 29, 77, 54)
sex = st.selectbox('SEX',['M', 'F'])
chest_pain_type = st.selectbox('Chest Pain Type', ['TA', 'ATA', 'NAP', 'ASY'])
resting_blood_pressure = st.slider('Resting Blood Pressure', 94, 200, 130)
serum_cholesterol = st.slider('Serum Cholesterol', 126, 600, 200)
fasting_blood_sugar = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['0', '1'])
rest_ecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
max_heart_rate = st.slider('Max Heart Rate', 71, 202, 150)
exercise_induced_angina = st.selectbox('Exercise Induced Angina', ['0', '1'])
oldpeak = st.slider('Oldpeak', 0.0, 6.2, 1.0)
slope = st.selectbox('Slope', ['Upsloping', 'Flat', 'Downsloping'])

if st.button('Predict'):
    raw_input = {
        'age': age,
        'sex': 1 if sex == 'M' else 0,
        'chest_pain_type': chest_pain_type,
        'resting_blood_pressure': resting_blood_pressure,
        'serum_cholesterol': serum_cholesterol,
        'fasting_blood_sugar': int(fasting_blood_sugar),
        'rest_ecg': rest_ecg,
        'max_heart_rate': max_heart_rate,
        'exercise_induced_angina': int(exercise_induced_angina),
        'oldpeak': oldpeak,
        'slope': slope
    }

    input_df = pd.DataFrame([raw_input])
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_cols]
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]

    if prediction == 1:
        st.error('The model predicts that you have heart disease.')
    else:
        st.success('The model predicts that you do not have heart disease.')