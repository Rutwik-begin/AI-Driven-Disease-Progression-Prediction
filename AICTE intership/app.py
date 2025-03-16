import streamlit as st
import pickle
import pandas as pd

model_file = "heart_disease_model.pkl"

try:
    with open(model_file, "rb") as file:
        cph_final = pickle.load(file)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# UI Layout
st.title("AI-Based Heart Disease Progression Predictor")
st.write("Enter patient details below to predict disease progression.")

# Define user input fields
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.radio("Sex", options=[0, 1], format_func=lambda x: "Male" if x == 1 else "Female")
cp = st.selectbox("Chest Pain Type", options=[0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, value=120)
chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", options=[0, 1])
restecg = st.selectbox("Resting ECG", options=[0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)
exang = st.radio("Exercise-Induced Angina", options=[0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)
slope = st.selectbox("Slope of the Peak ST Segment", options=[0, 1, 2])
ca = st.selectbox("Number of Major Vessels (0-3) Colored by Fluoroscopy", options=[0, 1, 2, 3])
thal = st.selectbox("Thalassemia Type", options=[0, 1, 2])

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    "age": [age], "sex": [sex], "cp": [cp], "trestbps": [trestbps],
    "chol": [chol], "fbs": [fbs], "restecg": [restecg], "thalach": [thalach],
    "exang": [exang], "oldpeak": [oldpeak], "slope": [slope], "ca": [ca], "thal": [thal]
})

# Make prediction
if st.button("Predict Disease Progression"):
    try:
        risk_score = float(cph_final.predict_partial_hazard(input_data).iloc[0])


        # Define critical risk threshold
        critical_threshold = 1.5  # Adjust based on data analysis

        # Display risk score
        st.success(f"Predicted Risk Score: {risk_score:.3f}")

        # Display condition indicator
        if risk_score >= critical_threshold:
            st.error("⚠️ High Risk: The patient is likely to be in a critical condition. Immediate medical attention is advised.")
        else:
            st.success("✅ Low to Moderate Risk: The patient is stable but should follow regular check-ups.")

    except Exception as e:
        st.error(f"Error making prediction: {e}")
