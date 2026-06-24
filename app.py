import streamlit as st
import pickle
import numpy as np

# 1. Load the pickled heart disease model
@st.cache_resource
def load_model():
    # Ensure 'heart_disease_model.pkl' is the exact name of the file in your GitHub repo
    with open('heart_disease_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("Error: 'heart_disease_model.pkl' not found. Please make sure it's in the same repository directory.")
    st.stop()

# 2. App Layout & Title
st.set_page_config(page_title="Heart Disease Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Prediction App")
st.write("Input the clinical features below to predict the presence of heart disease.")

# 3. Input UI tailored specifically to your 13 model features
st.header("Patient Clinical Metrics")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex (0 = Female, 1 = Male)", options=[0, 1])
    cp = st.selectbox("Chest Pain Type (0 to 3)", options=[0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholestoral (mg/dl)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = False, 1 = True)", options=[0, 1])
    restecg = st.selectbox("Resting Electrocardiographic Results (0, 1, 2)", options=[0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (0 = No, 1 = Yes)", options=[0, 1])
    oldpeak = st.number_input("ST Depression Induced by Exercise (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (0, 1, 2)", options=[0, 1, 2])
    ca = st.selectbox("Number of Major Vessels Colored by Flourosopy (0 to 4)", options=[0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia Status (0, 1, 2, 3)", options=[0, 1, 2, 3])

# 4. Predict button logic
if st.button("Analyze Metrics", type="primary"):
    # Group inputs in the exact array format your model was trained on
    input_data = np.array([[
        age, sex, cp, trestbps, chol, fbs, restecg, 
        thalach, exang, oldpeak, slope, ca, thal
    ]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    st.markdown("---")
    if prediction[0] == 1:
        st.error("⚠️ **Prediction:** The model detects indicators indicative of heart disease.")
    else:
        st.success("✅ **Prediction:** The model does not detect signs of heart disease.")