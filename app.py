# app.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('xgboost_model.joblib')

# Streamlit app
st.title('Chronic Kidney Disease Prediction')

# Input form
st.header('Enter Patient Bio-chemical Profile')

col1, col2, col3 = st.columns(3)

with col1:
    bp = st.number_input('Blood Pressure (Bp)', min_value=70.0, max_value=200.0, value=120.0)
    sg = st.number_input('Specific Gravity (Sg)', min_value=1.000, max_value=1.030, value=1.010, step=0.005)
    al = st.number_input('Albumin (Al)', min_value=0.0, max_value=5.0, value=0.0)
    su = st.number_input('Sugar (Su)', min_value=0.0, max_value=5.0, value=0.0)
    rbc = st.number_input('Red Blood Cell Count (Rbc)', min_value=2.0, max_value=6.0, value=4.0)
    bu = st.number_input('Blood Urea (Bu)', min_value=5.0, max_value=150.0, value=20.0)

with col2:
    sc = st.number_input('Serum Creatinine (Sc)', min_value=0.5, max_value=20.0, value=1.0)
    sod = st.number_input('Sodium (Sod)', min_value=100.0, max_value=160.0, value=140.0)
    pot = st.number_input('Potassium (Pot)', min_value=3.0, max_value=8.0, value=4.5)
    hemo = st.number_input('Hemoglobin (Hemo)', min_value=5.0, max_value=18.0, value=12.0)
    wbc = st.number_input('White Blood Cell Count (Wbcc)', min_value=2000.0, max_value=25000.0, value=8000.0)

with col3:
    rbcc = st.number_input('Red Blood Cell Count (Rbcc) - Numerical', min_value=2.0, max_value=6.0, value=4.0)
    htn = st.number_input('Hypertension (Htn)', min_value=0, max_value=1, value=0)

# Create a DataFrame from the input data
input_data = pd.DataFrame({
    'Bp': [bp], 'Sg': [sg], 'Al': [al], 'Su': [su], 'Rbc': [rbc], 'Bu': [bu],
    'Sc': [sc], 'Sod': [sod], 'Pot': [pot], 'Hemo': [hemo], 'Wbcc': [wbc], 'Rbcc': [rbcc], 'Htn': [htn]
})

# Normalize the input data (important! Must match training data normalization)
data = pd.read_csv('new_model.csv') 
feature_cols = data.columns.drop('Class') # To get min and max value
for x in feature_cols:  # Iterate through features only
    input_data[x] = (input_data[x] - data[x].min()) / (data[x].max() - data[x].min())

# ... (rest of your code)

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success('No Chronic Kidney Disease Predicted')
    else:
        st.error('Chronic Kidney Disease Predicted')
