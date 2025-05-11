import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Check if model files exist
model_path = 'saved_models/logistic_regression_model.joblib'
scaler_path = 'saved_models/scaler.joblib'
preprocessing_info_path = 'saved_models/preprocessing_info.joblib'

# Initialize variables
model = None
scaler = None
preprocessing_info = None

# Try to load the model files
try:
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(preprocessing_info_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        preprocessing_info = joblib.load(preprocessing_info_path)
    else:
        st.error("Model files not found. Please run the notebook to train and save the model first.")
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")

# Define preprocessing function with actual implementation
def preprocess_user_input(user_data):
    """
    Preprocess user input to match the format expected by the model.
    
    Parameters:
    -----------
    user_data : dict
        Dictionary with user input fields
        
    Returns:
    --------
    pd.DataFrame
        Processed dataframe ready for prediction
    """
    # Create a dataframe with the user data
    user_df = pd.DataFrame([user_data])
    
    # Handle categorical variables - One-hot encoding
    if 'ChestPain' in user_df.columns:
        # One-hot encode ChestPain
        chest_pain_dummies = pd.get_dummies(user_df['ChestPain'], prefix='ChestPain')
        user_df = pd.concat([user_df.drop('ChestPain', axis=1), chest_pain_dummies], axis=1)
        
        # Add missing dummy columns if any
        for col in ['ChestPain_nonanginal', 'ChestPain_nontypical', 'ChestPain_typical']:
            if col not in user_df.columns:
                user_df[col] = 0
    
    # Handle Thal
    if 'Thal' in user_df.columns:
        # One-hot encode Thal
        thal_dummies = pd.get_dummies(user_df['Thal'], prefix='Thal')
        user_df = pd.concat([user_df.drop('Thal', axis=1), thal_dummies], axis=1)
        
        # Add missing dummy columns if any
        for col in ['Thal_normal', 'Thal_reversable']:
            if col not in user_df.columns:
                user_df[col] = 0
    
    # Create age groups
    user_df['AgeGroup'] = pd.cut(user_df['Age'], bins=[0, 40, 55, 65, 100], 
                                labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
    
    # One-hot encode AgeGroup
    agegroup_dummies = pd.get_dummies(user_df['AgeGroup'], prefix='AgeGroup')
    user_df = pd.concat([user_df.drop('AgeGroup', axis=1), agegroup_dummies], axis=1)
    
    # Add missing dummy columns if any
    for col in ['AgeGroup_Middle-aged', 'AgeGroup_Senior', 'AgeGroup_Elderly']:
        if col not in user_df.columns:
            user_df[col] = 0
    
    # Create derived features
    user_df['BP_per_Age'] = user_df['RestBP'] / user_df['Age']
    user_df['HR_per_Age'] = user_df['MaxHR'] / user_df['Age']
    
    # Scale numerical features
    if scaler is not None and preprocessing_info is not None:
        num_features = preprocessing_info['numerical_features_to_scale']
        user_df[num_features] = scaler.transform(user_df[num_features])
    
    # Ensure all required columns are present in the correct order
    if preprocessing_info is not None:
        required_cols = preprocessing_info['feature_names']
        missing_cols = set(required_cols) - set(user_df.columns)
        
        # Add missing columns with default value 0
        for col in missing_cols:
            user_df[col] = 0
        
        # Reorder columns to match training data
        user_df = user_df[required_cols]
    
    return user_df

def predict_heart_disease(user_data):
    """
    Predict heart disease risk based on user input.
    
    Parameters:
    -----------
    user_data : dict
        Dictionary with user input fields
        
    Returns:
    --------
    dict
        Prediction results including probability and interpretation
    """
    if model is None:
        st.error("Model not loaded. Please run the notebook to train and save the model first.")
        return None
        
    try:
        # Preprocess the user input
        processed_input = preprocess_user_input(user_data)
        
        # Make prediction
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0, 1]
        
        # Format the result
        result = {
            'prediction': 'Heart Disease' if prediction == 1 else 'No Heart Disease',
            'probability': probability,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'success': True
        }
        
        return result
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

# Streamlit app
st.title("Heart Disease Prediction")
st.write("Enter patient information to assess heart disease risk")

# Create input form
with st.form("patient_info_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=20, max_value=100, value=50)
        sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], format_func=lambda x: x[1])[0]
        chest_pain = st.selectbox("Chest Pain Type", 
                                 options=["typical", "nontypical", "nonanginal", "asymptomatic"])
        rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, value=120)
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
    
    with col2:
        rest_ecg = st.selectbox("Resting ECG Results", 
                              options=[(0, "Normal"), 
                                      (1, "ST-T Wave Abnormality"), 
                                      (2, "Left Ventricular Hypertrophy")], 
                              format_func=lambda x: x[1])[0]
        max_hr = st.number_input("Maximum Heart Rate", min_value=50, max_value=220, value=150)
        ex_ang = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], format_func=lambda x: x[1])[0]
        oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                           options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], 
                           format_func=lambda x: x[1])[0]
        ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
        thal = st.selectbox("Thalassemia", options=["normal", "fixed", "reversable"])
    
    submit_button = st.form_submit_button("Predict")

# Make prediction when form is submitted
if submit_button:
    user_data = {
        'Age': age,
        'Sex': sex,
        'ChestPain': chest_pain,
        'RestBP': rest_bp,
        'Chol': chol,
        'Fbs': fbs,
        'RestECG': rest_ecg,
        'MaxHR': max_hr,
        'ExAng': ex_ang,
        'Oldpeak': oldpeak,
        'Slope': slope,
        'Ca': ca,
        'Thal': thal
    }
    
    result = predict_heart_disease(user_data)
    
    if result['success']:
        # Display result with nice formatting
        st.subheader("Prediction Results")
        
        # Create columns for metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Prediction", result['prediction'])
        
        with col2:
            st.metric("Probability", f"{result['probability']:.2f}")
        
        with col3:
            st.metric("Risk Level", result['risk_level'])
        
        # Add detailed recommendations based on the prediction
        st.subheader("Recommendations")
        
        if result['prediction'] == 'Heart Disease':
            st.error("The patient shows signs consistent with heart disease.")
            st.write("Medical follow-up is strongly recommended including:")
            st.write("- Consultation with a cardiologist")
            st.write("- Further diagnostic tests (ECG, stress test, etc.)")
            st.write("- Review of current lifestyle and medications")
        else:
            st.success("The patient shows low risk of heart disease.")
            st.write("Continue with regular health check-ups and healthy lifestyle including:")
            st.write("- Regular exercise")
            st.write("- Balanced diet low in saturated fats")
            st.write("- Regular monitoring of blood pressure and cholesterol")
            st.write("- Avoiding smoking and excessive alcohol consumption")
    else:
        st.error(f"Error in prediction: {result['error']}")
        st.write("Please check your inputs and try again.")