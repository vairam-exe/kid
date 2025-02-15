import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------
# Load the pre-trained XGBoost model using pickle
# ---------------------------

with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)


# ---------------------------
# App Title and Description
# ---------------------------
st.title("Chronic Kidney Disease Prediction App")
st.write(
    """
    This application uses an XGBoost model (with an accuracy of approximately 98% as reported)
    to predict the presence of chronic kidney disease based on biochemical and clinical parameters.
    
    **Key Findings from the Analysis:**
    - **Exploratory Data Analysis (EDA):** The notebook revealed that features such as Blood Urea, Serum Creatinine, and Hemoglobin have a strong influence on the prediction.
    - **Model Performance:** After proper preprocessing and tuning, the XGBoost classifier achieved high accuracy.
    - **Feature Importance:** Several key clinical features were found to be most predictive of chronic kidney disease.
    
    Please enter the patient’s details in the form below.
    """
)

# ---------------------------
# Sidebar: Analysis Summary
# ---------------------------
st.sidebar.title("Analysis Summary")
st.sidebar.info(
    """
    - **Dataset:** UCI Chronic Kidney Disease Dataset  
    - **Key Predictors:** Blood Urea, Serum Creatinine, Hemoglobin, etc.  
    - **Model:** XGBoost (98% accuracy reported)  
    - **Preprocessing:** Categorical values converted and missing values handled  
    """
)

# ---------------------------
# User Input Form
# ---------------------------
st.header("Patient Data Input Form")
with st.form("ckd_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=1, max_value=120, value=30)
        bp = st.number_input("Blood Pressure (mm Hg)", min_value=50, max_value=200, value=80)
        sg = st.selectbox("Specific Gravity", options=["1.005", "1.010", "1.015", "1.020", "1.025"])
        al = st.selectbox("Albumin (0-5)", options=["0", "1", "2", "3", "4", "5"])
        su = st.selectbox("Sugar (0-5)", options=["0", "1", "2", "3", "4", "5"])
        rbc = st.selectbox("Red Blood Cells", options=["normal", "abnormal"])
        pc = st.selectbox("Pus Cell", options=["normal", "abnormal"])
        pcc = st.selectbox("Pus Cell Clumps", options=["present", "not present"])
        ba = st.selectbox("Bacteria", options=["present", "not present"])
    
    with col2:
        bgr = st.number_input("Blood Glucose Random", min_value=0, max_value=300, value=100)
        bu = st.number_input("Blood Urea", min_value=0, max_value=300, value=50)
        sc = st.number_input("Serum Creatinine", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        sod = st.number_input("Sodium (mEq/L)", min_value=0, max_value=200, value=140)
        pot = st.number_input("Potassium (mEq/L)", min_value=0.0, max_value=10.0, value=4.5, step=0.1)
        hemo = st.number_input("Hemoglobin (g/dL)", min_value=0.0, max_value=20.0, value=12.0, step=0.1)
        pcv = st.number_input("Packed Cell Volume (%)", min_value=0, max_value=100, value=40)
        wc = st.number_input("White Blood Cell Count", min_value=0, max_value=20000, value=8000)
        rc = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
    
    # Additional binary/categorical features
    htn = st.selectbox("Hypertension", options=["yes", "no"])
    dm = st.selectbox("Diabetes Mellitus", options=["yes", "no"])
    cad = st.selectbox("Coronary Artery Disease", options=["yes", "no"])
    appet = st.selectbox("Appetite", options=["good", "poor"])
    pe = st.selectbox("Pedal Edema", options=["yes", "no"])
    ane = st.selectbox("Anemia", options=["yes", "no"])
    
    submit_button = st.form_submit_button("Predict")

# ---------------------------
# Preprocessing & Prediction
# ---------------------------
if submit_button:
    try:
        # Convert numeric and continuous features to the appropriate types
        age = float(age)
        bp = float(bp)
        sg = float(sg)
        al = int(al)
        su = int(su)
        bgr = float(bgr)
        bu = float(bu)
        sc = float(sc)
        sod = float(sod)
        pot = float(pot)
        hemo = float(hemo)
        pcv = int(pcv)
        wc = float(wc)
        rc = float(rc)
        
        # Map categorical variables to numerical values (adjust these mappings as per your training)
        rbc = 1 if rbc.lower() == "normal" else 0
        pc = 1 if pc.lower() == "normal" else 0
        pcc = 1 if pcc.lower() == "present" else 0
        ba = 1 if ba.lower() == "present" else 0
        htn = 1 if htn.lower() == "yes" else 0
        dm = 1 if dm.lower() == "yes" else 0
        cad = 1 if cad.lower() == "yes" else 0
        appet = 1 if appet.lower() == "good" else 0
        pe = 1 if pe.lower() == "yes" else 0
        ane = 1 if ane.lower() == "yes" else 0
        
        # Arrange the input features in the order expected by the model
        input_data = np.array([[age, bp, sg, al, su, rbc, pc, pcc, ba, bgr,
                                bu, sc, sod, pot, hemo, pcv, wc, rc, htn, dm,
                                cad, appet, pe, ane]])
        
        # Predict using the pre-trained model
        prediction = model.predict(input_data)
        
        # Interpret the result (assuming label 1 indicates CKD and 0 indicates non-CKD)
        if prediction[0] == 1:
            st.error("The model predicts that the patient **has Chronic Kidney Disease**.")
        else:
            st.success("The model predicts that the patient **does not have Chronic Kidney Disease**.")
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
