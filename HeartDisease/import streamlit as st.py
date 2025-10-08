import streamlit as st
import pickle
import numpy as np

def infer_heart_disease(model_path, scaler_path, input_features):
    """
    Perform inferencing on the heart disease dataset.

    Parameters:
    - model_path (str): Path to the saved model pickle file.
    - scaler_path (str): Path to the saved scaler pickle file.
    - input_features (list): List of input features in the order:
      [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

    Returns:
    - str: "Disease Detected" if target is 1, otherwise "No Disease".
    """
    try:
        # Load the trained model and scaler
        with open(r'C:\Users\shaik\OneDrive\Desktop\HeartDisease\Heart_disease_model.pickle', 'rb') as model_file:
            model = pickle.load(model_file)

        with open(r'C:\Users\shaik\OneDrive\Desktop\HeartDisease\scaler (2).pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)

        # Ensure the input features are in the correct format
        input_array = np.array([input_features]).reshape(1, -1)

        # Scale the input features
        scaled_input = scaler.transform(input_array)

        # Predict outcome
        prediction = model.predict(scaled_input)

        # Return result
        return "Disease Detected" if prediction[0] == 1 else "No Disease"

    except Exception as e:
        return f"An error occurred: {e}"

# Streamlit App
st.title("Heart Disease Prediction App")

# Sidebar inputs
st.sidebar.header("Input Features")
age = st.sidebar.number_input("Age", min_value=0, step=1)
sex = st.sidebar.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.sidebar.selectbox("Chest Pain Type (0, 1, 2, 3)", [0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", min_value=0, step=1)
chol = st.sidebar.number_input("Cholesterol (chol)", min_value=0, step=1)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (0 = No, 1 = Yes)", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG Results (0, 1, 2)", [0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", min_value=0, step=1)
exang = st.sidebar.selectbox("Exercise-Induced Angina (0 = No, 1 = Yes)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", min_value=0.0, step=0.1)
slope = st.sidebar.selectbox("Slope of Peak Exercise (0, 1, 2)", [0, 1, 2])
ca = st.sidebar.selectbox("Number of Major Vessels (0-3)", [0, 1, 2, 3])
thal = st.sidebar.selectbox("Thalassemia (0, 1, 2, 3)", [0, 1, 2, 3])

# Model and scaler paths
model_path = 'heart_disease_model.pkl'  # Update with your actual path
scaler_path = 'scaler.pkl'  # Update with your actual path

# Predict button
if st.button("Predict Heart Disease"):
    # Collect input features in the correct order
    input_features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Perform inference
    result = infer_heart_disease(model_path, scaler_path, input_features)
    
    # Display result
    st.write(f"Prediction: **{result}**")