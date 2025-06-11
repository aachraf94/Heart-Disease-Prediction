import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import re
from datetime import datetime
import requests
import json

# Hugging Face API Configuration
# Use environment variable or Streamlit secrets for API token
HF_API_TOKEN = os.getenv("HF_API_TOKEN") or st.secrets.get("HF_API_TOKEN", "")
HF_API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"

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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False

# Extract medical information from user text input
def extract_medical_info(text):
    """
    Enhanced medical information extraction from user text input.
    
    Parameters:
    -----------
    text : str
        User input text
        
    Returns:
    --------
    dict
        Extracted medical information
    """
    info = {}
    text_lower = text.lower()
    
    # Extract age with multiple patterns
    age_patterns = [
        r'\b(\d{1,3})\s*(?:years?\s*old|yo|age|year-old)\b',
        r'\bi\s*(?:am|\'m)\s*(?:a\s*)?(\d{1,3})\s*(?:years?\s*old|yo)?\b',
        r'\b(\d{1,3})\s*(?:-|–)\s*year\s*old\b',
        r'\bage\s*(?:of\s*|is\s*)?(\d{1,3})\b',
        r'\b(\d{1,3})\s*yrs?\b'
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, text_lower)
        if age_match:
            age = int(age_match.group(1))
            if 20 <= age <= 100:
                info['Age'] = age
                break
    
    # Extract gender with enhanced patterns
    gender_patterns = [
        (r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?(?:male|man|men|boy|guy|gentleman)\b', 1),
        (r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?(?:female|woman|women|girl|lady)\b', 0),
        (r'\bgender\s*(?:is\s*|:\s*)?(?:male|man)\b', 1),
        (r'\bgender\s*(?:is\s*|:\s*)?(?:female|woman)\b', 0),
        (r'\bsex\s*(?:is\s*|:\s*)?(?:male|m)\b', 1),
        (r'\bsex\s*(?:is\s*|:\s*)?(?:female|f)\b', 0)
    ]
    
    for pattern, gender_value in gender_patterns:
        if re.search(pattern, text_lower):
            info['Sex'] = gender_value
            break
    
    # Extract chest pain type with enhanced patterns
    chest_pain_patterns = [
        (r'\btypical\s*(?:angina|chest\s*pain)\b', 'typical'),
        (r'\batypical\s*(?:angina|chest\s*pain)\b', 'nontypical'),
        (r'\bnon[\s-]?anginal\s*(?:chest\s*pain)?\b', 'nonanginal'),
        (r'\basymptomatic\b|no\s*chest\s*pain\b', 'asymptomatic'),
        (r'\bchest\s*pain\b(?!\s*(?:type|is))', 'typical'),  # default assumption
        (r'\bangina\b(?!\s*(?:typical|atypical))', 'typical')
    ]
    
    for pattern, chest_pain_type in chest_pain_patterns:
        if re.search(pattern, text_lower):
            info['ChestPain'] = chest_pain_type
            break
    
    # Extract blood pressure with multiple patterns
    bp_patterns = [
        r'\b(?:blood\s*pressure|bp)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})(?:\s*mmhg|mm\s*hg)?\b',
        r'\b(\d{2,3})\s*(?:mmhg|mm\s*hg)\s*(?:blood\s*pressure|bp)?\b',
        r'\b(\d{2,3})/\d{2,3}\s*(?:mmhg|mm\s*hg)?\b',  # systolic from BP reading
        r'\bpressure\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in bp_patterns:
        bp_match = re.search(pattern, text_lower)
        if bp_match:
            bp = int(bp_match.group(1))
            if 80 <= bp <= 220:
                info['RestBP'] = bp
                break
    
    # Extract cholesterol with multiple patterns
    chol_patterns = [
        r'\b(?:cholesterol|chol)\s*(?:is\s*|of\s*|level\s*is\s*|:\s*)?(\d{2,3})\s*(?:mg/dl|mg\s*/\s*dl)?\b',
        r'\b(\d{2,3})\s*(?:mg/dl|mg\s*/\s*dl)\s*(?:cholesterol|chol)?\b',
        r'\btotal\s*cholesterol\s*(?:is\s*|:\s*)?(\d{2,3})\b',
        r'\bchol\s*(?:level\s*)?(?:is\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in chol_patterns:
        chol_match = re.search(pattern, text_lower)
        if chol_match:
            chol = int(chol_match.group(1))
            if 100 <= chol <= 600:
                info['Chol'] = chol
                break
    
    # Extract heart rate with multiple patterns
    hr_patterns = [
        r'\b(?:heart\s*rate|hr)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\s*(?:bpm|beats\s*per\s*minute)?\b',
        r'\b(?:max|maximum)\s*(?:heart\s*rate|hr)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b',
        r'\b(\d{2,3})\s*(?:bpm|beats\s*per\s*minute)\b',
        r'\bpulse\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in hr_patterns:
        hr_match = re.search(pattern, text_lower)
        if hr_match:
            hr = int(hr_match.group(1))
            if 50 <= hr <= 220:
                info['MaxHR'] = hr
                break
    
    # Extract exercise-related symptoms
    exercise_patterns = [
        r'\b(?:exercise|exertion|activity)\s*(?:induced\s*)?(?:chest\s*pain|angina|discomfort)\b',
        r'\bchest\s*pain\s*(?:during|with|when)\s*(?:exercise|exertion|activity)\b',
        r'\bpain\s*(?:during|with|when)\s*(?:exercise|exertion|walking|running)\b'
    ]
    
    for pattern in exercise_patterns:
        if re.search(pattern, text_lower):
            info['ExAng'] = 1
            break
    
    return info

def generate_enhanced_fallback_response(user_message, context=""):
    """
    Generate enhanced fallback responses when AI service is unavailable
    """
    user_lower = user_message.lower()
    
    # Medical keyword responses
    if any(word in user_lower for word in ['chest pain', 'angina', 'heart pain']):
        return """🩺 **About Chest Pain & Heart Disease:**
        
Chest pain can have various causes:
• **Typical angina:** Pressure/squeezing feeling, often triggered by exertion
• **Atypical angina:** Sharp or burning pain, may occur at rest
• **Non-cardiac:** Could be muscle strain, anxiety, or other conditions

**When to seek immediate care:**
• Severe chest pain with sweating
• Pain radiating to arm, jaw, or back
• Shortness of breath
• Nausea or dizziness

⚠️ Always consult healthcare professionals for chest pain evaluation."""

    elif any(word in user_lower for word in ['blood pressure', 'bp', 'hypertension']):
        return """📊 **Blood Pressure & Heart Health:**
        
**Normal ranges:**
• Normal: Less than 120/80 mmHg
• Elevated: 120-129 (systolic) and less than 80 (diastolic)
• High (Stage 1): 130-139/80-89 mmHg
• High (Stage 2): 140/90 mmHg or higher

**Management tips:**
• Regular exercise (30 minutes daily)
• Low-sodium diet
• Maintain healthy weight
• Limit alcohol and quit smoking
• Regular monitoring

⚠️ Consult your doctor for proper blood pressure management."""

    elif any(word in user_lower for word in ['cholesterol', 'chol']):
        return """🧪 **Cholesterol & Heart Disease:**
        
**Cholesterol levels (mg/dL):**
• Total cholesterol: Less than 200 (desirable)
• LDL (bad): Less than 100 (optimal)
• HDL (good): 40+ (men), 50+ (women)

**Risk factors:**
• High LDL cholesterol
• Low HDL cholesterol
• Family history
• Poor diet and lifestyle

**Improvement strategies:**
• Heart-healthy diet
• Regular physical activity
• Weight management
• Medication if prescribed

⚠️ Regular lipid panels and medical consultation recommended."""

    elif any(word in user_lower for word in ['age', 'old', 'years']):
        return """⏰ **Age & Heart Disease Risk:**
        
Age is a significant risk factor:
• **Men:** Risk increases after age 45
• **Women:** Risk increases after age 55 (post-menopause)

**Age-related considerations:**
• Arteries naturally stiffen with age
• Blood pressure tends to increase
• Cholesterol levels may rise
• Overall cardiovascular function changes

**Preventive measures:**
• Regular health screenings
• Maintain active lifestyle
• Heart-healthy diet
• Stress management
• Regular medical check-ups

⚠️ Age-appropriate care plans should be discussed with healthcare providers."""

    elif any(word in user_lower for word in ['exercise', 'activity', 'workout']):
        return """🏃‍♂️ **Exercise & Heart Health:**
        
**Recommended activity:**
• 150 minutes moderate aerobic activity per week
• 75 minutes vigorous activity per week
• Muscle strengthening 2+ days per week

**Heart benefits:**
• Strengthens heart muscle
• Improves circulation
• Lowers blood pressure
• Manages cholesterol
• Reduces stress

**Exercise precautions:**
• Start slowly if sedentary
• Warm up and cool down
• Stay hydrated
• Stop if experiencing chest pain or severe shortness of breath

⚠️ Consult healthcare providers before starting new exercise programs."""

    else:
        return f"""🤖 **Heart Disease Information:**
        
Thank you for your question about: "{user_message}"

**Key heart disease risk factors:**
• Age (men >45, women >55)
• High blood pressure
• High cholesterol
• Smoking
• Diabetes
• Family history
• Sedentary lifestyle

**Common symptoms to watch:**
• Chest pain or discomfort
• Shortness of breath
• Fatigue
• Swelling in legs/feet
• Irregular heartbeat

**Prevention strategies:**
• Regular exercise
• Healthy diet
• No smoking
• Limited alcohol
• Stress management
• Regular check-ups

⚠️ For personalized medical advice, please consult qualified healthcare professionals."""

def get_medical_ai_response(user_message, context=""):
    """
    Alias for get_ai_response to maintain compatibility
    """
    return get_ai_response(user_message, context)

def get_ai_response(user_message, context=""):
    """
    Enhanced AI response system with better prompt engineering
    """
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Enhanced medical-focused prompt for Mistral
        system_prompt = """You are an expert medical AI assistant specializing in heart disease risk assessment. 
        
Your role is to:
- Extract medical information from patient descriptions
- Provide educational information about cardiovascular health
- Guide patients through risk assessment
- Explain medical concepts in simple terms
- Always recommend consulting healthcare professionals

Be empathetic, professional, and informative. Focus on heart disease risk factors, symptoms, and prevention."""
        
        # Enhanced context processing
        context_summary = ""
        if context:
            # Parse context to provide better responses
            if "Patient data:" in context:
                data_part = context.split("Patient data:")[1].split("|")[0].strip()
                context_summary = f"Current patient information: {data_part}\n"
            
            if "Recent conversation:" in context:
                conv_part = context.split("Recent conversation:")[1].strip()
                context_summary += f"Recent discussion: {conv_part}\n"
        
        # Format the prompt for Mistral with better structure
        full_prompt = f"""<s>[INST] {system_prompt}

{context_summary}
Patient says: "{user_message}"

Please provide a helpful response that:
1. Acknowledges any medical information shared
2. Provides relevant educational information
3. Suggests next steps if appropriate
4. Reminds about professional medical consultation

Response: [/INST]"""
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
                "repetition_penalty": 1.1
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '').strip()
                
                # Clean up the response
                if ai_response:
                    # Remove any prompt repetition
                    ai_response = ai_response.replace(system_prompt, "").strip()
                    ai_response = ai_response.replace(user_message, "").strip()
                    
                    # Ensure medical disclaimer
                    if "consult" not in ai_response.lower() and "medical advice" not in ai_response.lower():
                        ai_response += "\n\n⚠️ Please consult healthcare professionals for medical advice."
                    
                    return ai_response
        
        # Enhanced fallback
        return generate_enhanced_fallback_response(user_message, context)
        
    except requests.exceptions.Timeout:
        st.warning("🤖 AI response taking longer than usual. Using enhanced response.")
        return generate_enhanced_fallback_response(user_message, context)
    except Exception as e:
        st.warning(f"🤖 AI service temporarily unavailable. Error: {str(e)}")
        return generate_enhanced_fallback_response(user_message, context)

def generate_smart_response(user_message, extracted_info=None):
    """
    Enhanced intelligent response system with better information processing
    """
    # Build comprehensive context
    context_parts = []
    
    if st.session_state.user_data:
        # Format existing data nicely
        data_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.user_data.items()])
        context_parts.append(f"Existing patient data: {data_str}")
    
    if st.session_state.chat_history:
        # Get more relevant conversation history
        recent_history = st.session_state.chat_history[-4:]  # Last 4 exchanges
        history_str = " | ".join([f"{role}: {msg[:100]}..." if len(msg) > 100 else f"{role}: {msg}" 
                                 for role, msg, _ in recent_history])
        context_parts.append(f"Recent conversation: {history_str}")
    
    context = " | ".join(context_parts)
    
    # Enhanced information extraction and response
    if extracted_info:
        response_parts = ["🩺 **Medical Information Successfully Extracted:**\n"]
        
        # Process extracted information with enhanced descriptions
        for key, value in extracted_info.items():
            if key == 'Sex':
                response_parts.append(f"• **Gender:** {'Male' if value == 1 else 'Female'}")
            elif key == 'Age':
                age_category = ("Young adult" if value < 45 else 
                              "Middle-aged" if value < 65 else "Senior")
                response_parts.append(f"• **Age:** {value} years ({age_category})")
            elif key == 'ChestPain':
                chest_pain_desc = {
                    'typical': 'Typical angina (classic chest pain)',
                    'nontypical': 'Atypical angina',
                    'nonanginal': 'Non-anginal chest pain',
                    'asymptomatic': 'No chest pain symptoms'
                }
                response_parts.append(f"• **Chest Pain:** {chest_pain_desc.get(value, value.title())}")
            elif key == 'RestBP':
                bp_status = ("High (≥140)" if value >= 140 else 
                           "Elevated (120-139)" if value >= 120 else "Normal (<120)")
                response_parts.append(f"• **Blood Pressure:** {value} mmHg ({bp_status})")
            elif key == 'Chol':
                chol_status = ("High (≥240)" if value >= 240 else 
                             "Borderline (200-239)" if value >= 200 else "Normal (<200)")
                response_parts.append(f"• **Cholesterol:** {value} mg/dl ({chol_status})")
            elif key == 'MaxHR':
                max_hr_expected = 220 - st.session_state.user_data.get('Age', 50)
                hr_status = ("Above expected" if value > max_hr_expected else 
                           "Within normal range" if value > max_hr_expected * 0.8 else "Below expected")
                response_parts.append(f"• **Max Heart Rate:** {value} bpm ({hr_status})")
            elif key == 'ExAng':
                response_parts.append(f"• **Exercise-Induced Symptoms:** {'Yes - pain with exertion' if value == 1 else 'No symptoms with exercise'}")
        
        # Update session state with new information
        st.session_state.user_data.update(extracted_info)
        
        # Enhanced completeness check
        essential_fields = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']
        additional_fields = ['MaxHR', 'ExAng']
        
        missing_essential = [field for field in essential_fields if field not in st.session_state.user_data]
        missing_additional = [field for field in additional_fields if field not in st.session_state.user_data]
        
        if missing_essential:
            response_parts.append(f"\n📋 **Essential information still needed:** {', '.join(missing_essential)}")
            response_parts.append("Please provide these details for a complete assessment.")
        elif missing_additional:
            response_parts.append(f"\n📊 **Additional helpful information:** {', '.join(missing_additional)}")
            response_parts.append("You can provide more details or proceed with current information.")
        else:
            response_parts.append("\n✅ **Comprehensive information collected!** Ready for detailed risk assessment.")
        
        # Add AI-powered medical insight
        try:
            # Create enhanced context for AI analysis
            medical_context = f"Patient profile: {extracted_info}. Complete data: {st.session_state.user_data}"
            
            # Enhanced AI query for medical insights
            insight_query = f"""Based on the medical information provided ({extracted_info}), provide brief educational insights about:
            1. The significance of these values for heart disease risk
            2. Any notable risk factors identified
            3. General recommendations (not medical advice)
            
            Keep response concise and educational."""
            
            ai_insight = get_medical_ai_response(insight_query, medical_context)
            
            if ai_insight and len(ai_insight) > 30:
                response_parts.append(f"\n💡 **AI Medical Insight:**\n{ai_insight}")
        except Exception as e:
            # Fallback insight based on extracted information
            risk_factors = []
            if 'Age' in extracted_info and extracted_info['Age'] > 45:
                risk_factors.append("age-related risk increase")
            if 'RestBP' in extracted_info and extracted_info['RestBP'] >= 140:
                risk_factors.append("elevated blood pressure")
            if 'Chol' in extracted_info and extracted_info['Chol'] >= 240:
                risk_factors.append("high cholesterol")
            
            if risk_factors:
                response_parts.append(f"\n⚠️ **Notable factors:** {', '.join(risk_factors)} detected.")
        
        return "\n".join(response_parts)
    
    # Enhanced general query handling with AI
    try:
        # Create better context for general medical queries
        enhanced_context = context
        if st.session_state.user_data:
            data_summary = f"Patient has provided: {', '.join(st.session_state.user_data.keys())}"
            enhanced_context += f" | {data_summary}"
        
        return get_medical_ai_response(user_message, enhanced_context)
    except:
        return generate_enhanced_fallback_response(user_message, context)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False

# Extract medical information from user text input
def extract_medical_info(text):
    """
    Enhanced medical information extraction from user text input.
    
    Parameters:
    -----------
    text : str
        User input text
        
    Returns:
    --------
    dict
        Extracted medical information
    """
    info = {}
    text_lower = text.lower()
    
    # Extract age with multiple patterns
    age_patterns = [
        r'\b(\d{1,3})\s*(?:years?\s*old|yo|age|year-old)\b',
        r'\bi\s*(?:am|\'m)\s*(?:a\s*)?(\d{1,3})\s*(?:years?\s*old|yo)?\b',
        r'\b(\d{1,3})\s*(?:-|–)\s*year\s*old\b',
        r'\bage\s*(?:of\s*|is\s*)?(\d{1,3})\b',
        r'\b(\d{1,3})\s*yrs?\b'
    ]
    
    for pattern in age_patterns:
        age_match = re.search(pattern, text_lower)
        if age_match:
            age = int(age_match.group(1))
            if 20 <= age <= 100:
                info['Age'] = age
                break
    
    # Extract gender with enhanced patterns
    gender_patterns = [
        (r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?(?:male|man|men|boy|guy|gentleman)\b', 1),
        (r'\b(?:i\s*(?:am|\'m)\s*(?:a\s*)?)?(?:female|woman|women|girl|lady)\b', 0),
        (r'\bgender\s*(?:is\s*|:\s*)?(?:male|man)\b', 1),
        (r'\bgender\s*(?:is\s*|:\s*)?(?:female|woman)\b', 0),
        (r'\bsex\s*(?:is\s*|:\s*)?(?:male|m)\b', 1),
        (r'\bsex\s*(?:is\s*|:\s*)?(?:female|f)\b', 0)
    ]
    
    for pattern, gender_value in gender_patterns:
        if re.search(pattern, text_lower):
            info['Sex'] = gender_value
            break
    
    # Extract chest pain type with enhanced patterns
    chest_pain_patterns = [
        (r'\btypical\s*(?:angina|chest\s*pain)\b', 'typical'),
        (r'\batypical\s*(?:angina|chest\s*pain)\b', 'nontypical'),
        (r'\bnon[\s-]?anginal\s*(?:chest\s*pain)?\b', 'nonanginal'),
        (r'\basymptomatic\b|no\s*chest\s*pain\b', 'asymptomatic'),
        (r'\bchest\s*pain\b(?!\s*(?:type|is))', 'typical'),  # default assumption
        (r'\bangina\b(?!\s*(?:typical|atypical))', 'typical')
    ]
    
    for pattern, chest_pain_type in chest_pain_patterns:
        if re.search(pattern, text_lower):
            info['ChestPain'] = chest_pain_type
            break
    
    # Extract blood pressure with multiple patterns
    bp_patterns = [
        r'\b(?:blood\s*pressure|bp)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})(?:\s*mmhg|mm\s*hg)?\b',
        r'\b(\d{2,3})\s*(?:mmhg|mm\s*hg)\s*(?:blood\s*pressure|bp)?\b',
        r'\b(\d{2,3})/\d{2,3}\s*(?:mmhg|mm\s*hg)?\b',  # systolic from BP reading
        r'\bpressure\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in bp_patterns:
        bp_match = re.search(pattern, text_lower)
        if bp_match:
            bp = int(bp_match.group(1))
            if 80 <= bp <= 220:
                info['RestBP'] = bp
                break
    
    # Extract cholesterol with multiple patterns
    chol_patterns = [
        r'\b(?:cholesterol|chol)\s*(?:is\s*|of\s*|level\s*is\s*|:\s*)?(\d{2,3})\s*(?:mg/dl|mg\s*/\s*dl)?\b',
        r'\b(\d{2,3})\s*(?:mg/dl|mg\s*/\s*dl)\s*(?:cholesterol|chol)?\b',
        r'\btotal\s*cholesterol\s*(?:is\s*|:\s*)?(\d{2,3})\b',
        r'\bchol\s*(?:level\s*)?(?:is\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in chol_patterns:
        chol_match = re.search(pattern, text_lower)
        if chol_match:
            chol = int(chol_match.group(1))
            if 100 <= chol <= 600:
                info['Chol'] = chol
                break
    
    # Extract heart rate with multiple patterns
    hr_patterns = [
        r'\b(?:heart\s*rate|hr)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\s*(?:bpm|beats\s*per\s*minute)?\b',
        r'\b(?:max|maximum)\s*(?:heart\s*rate|hr)\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b',
        r'\b(\d{2,3})\s*(?:bpm|beats\s*per\s*minute)\b',
        r'\bpulse\s*(?:is\s*|of\s*|:\s*)?(\d{2,3})\b'
    ]
    
    for pattern in hr_patterns:
        hr_match = re.search(pattern, text_lower)
        if hr_match:
            hr = int(hr_match.group(1))
            if 50 <= hr <= 220:
                info['MaxHR'] = hr
                break
    
    # Extract exercise-related symptoms
    exercise_patterns = [
        r'\b(?:exercise|exertion|activity)\s*(?:induced\s*)?(?:chest\s*pain|angina|discomfort)\b',
        r'\bchest\s*pain\s*(?:during|with|when)\s*(?:exercise|exertion|activity)\b',
        r'\bpain\s*(?:during|with|when)\s*(?:exercise|exertion|walking|running)\b'
    ]
    
    for pattern in exercise_patterns:
        if re.search(pattern, text_lower):
            info['ExAng'] = 1
            break
    
    return info

def get_ai_response(user_message, context=""):
    """
    Enhanced AI response system with better prompt engineering
    """
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Enhanced medical-focused prompt for Mistral
        system_prompt = """You are an expert medical AI assistant specializing in heart disease risk assessment. 
        
Your role is to:
- Extract medical information from patient descriptions
- Provide educational information about cardiovascular health
- Guide patients through risk assessment
- Explain medical concepts in simple terms
- Always recommend consulting healthcare professionals

Be empathetic, professional, and informative. Focus on heart disease risk factors, symptoms, and prevention."""
        
        # Enhanced context processing
        context_summary = ""
        if context:
            # Parse context to provide better responses
            if "Patient data:" in context:
                data_part = context.split("Patient data:")[1].split("|")[0].strip()
                context_summary = f"Current patient information: {data_part}\n"
            
            if "Recent conversation:" in context:
                conv_part = context.split("Recent conversation:")[1].strip()
                context_summary += f"Recent discussion: {conv_part}\n"
        
        # Format the prompt for Mistral with better structure
        full_prompt = f"""<s>[INST] {system_prompt}

{context_summary}
Patient says: "{user_message}"

Please provide a helpful response that:
1. Acknowledges any medical information shared
2. Provides relevant educational information
3. Suggests next steps if appropriate
4. Reminds about professional medical consultation

Response: [/INST]"""
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 250,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False,
                "repetition_penalty": 1.1
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=20)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '').strip()
                
                # Clean up the response
                if ai_response:
                    # Remove any prompt repetition
                    ai_response = ai_response.replace(system_prompt, "").strip()
                    ai_response = ai_response.replace(user_message, "").strip()
                    
                    # Ensure medical disclaimer
                    if "consult" not in ai_response.lower() and "medical advice" not in ai_response.lower():
                        ai_response += "\n\n⚠️ Please consult healthcare professionals for medical advice."
                    
                    return ai_response
        
        # Enhanced fallback
        return generate_enhanced_fallback_response(user_message, context)
        
    except requests.exceptions.Timeout:
        st.warning("🤖 AI response taking longer than usual. Using enhanced response.")
        return generate_enhanced_fallback_response(user_message, context)
    except Exception as e:
        st.warning(f"🤖 AI service temporarily unavailable. Error: {str(e)}")
        return generate_enhanced_fallback_response(user_message, context)

def generate_smart_response(user_message, extracted_info=None):
    """
    Enhanced intelligent response system with better information processing
    """
    # Build comprehensive context
    context_parts = []
    
    if st.session_state.user_data:
        # Format existing data nicely
        data_str = ", ".join([f"{k}: {v}" for k, v in st.session_state.user_data.items()])
        context_parts.append(f"Existing patient data: {data_str}")
    
    if st.session_state.chat_history:
        # Get more relevant conversation history
        recent_history = st.session_state.chat_history[-4:]  # Last 4 exchanges
        history_str = " | ".join([f"{role}: {msg[:100]}..." if len(msg) > 100 else f"{role}: {msg}" 
                                 for role, msg, _ in recent_history])
        context_parts.append(f"Recent conversation: {history_str}")
    
    context = " | ".join(context_parts)
    
    # Enhanced information extraction and response
    if extracted_info:
        response_parts = ["🩺 **Medical Information Successfully Extracted:**\n"]
        
        # Process extracted information with enhanced descriptions
        for key, value in extracted_info.items():
            if key == 'Sex':
                response_parts.append(f"• **Gender:** {'Male' if value == 1 else 'Female'}")
            elif key == 'Age':
                age_category = ("Young adult" if value < 45 else 
                              "Middle-aged" if value < 65 else "Senior")
                response_parts.append(f"• **Age:** {value} years ({age_category})")
            elif key == 'ChestPain':
                chest_pain_desc = {
                    'typical': 'Typical angina (classic chest pain)',
                    'nontypical': 'Atypical angina',
                    'nonanginal': 'Non-anginal chest pain',
                    'asymptomatic': 'No chest pain symptoms'
                }
                response_parts.append(f"• **Chest Pain:** {chest_pain_desc.get(value, value.title())}")
            elif key == 'RestBP':
                bp_status = ("High (≥140)" if value >= 140 else 
                           "Elevated (120-139)" if value >= 120 else "Normal (<120)")
                response_parts.append(f"• **Blood Pressure:** {value} mmHg ({bp_status})")
            elif key == 'Chol':
                chol_status = ("High (≥240)" if value >= 240 else 
                             "Borderline (200-239)" if value >= 200 else "Normal (<200)")
                response_parts.append(f"• **Cholesterol:** {value} mg/dl ({chol_status})")
            elif key == 'MaxHR':
                max_hr_expected = 220 - st.session_state.user_data.get('Age', 50)
                hr_status = ("Above expected" if value > max_hr_expected else 
                           "Within normal range" if value > max_hr_expected * 0.8 else "Below expected")
                response_parts.append(f"• **Max Heart Rate:** {value} bpm ({hr_status})")
            elif key == 'ExAng':
                response_parts.append(f"• **Exercise-Induced Symptoms:** {'Yes - pain with exertion' if value == 1 else 'No symptoms with exercise'}")
        
        # Update session state with new information
        st.session_state.user_data.update(extracted_info)
        
        # Enhanced completeness check
        essential_fields = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol']
        additional_fields = ['MaxHR', 'ExAng']
        
        missing_essential = [field for field in essential_fields if field not in st.session_state.user_data]
        missing_additional = [field for field in additional_fields if field not in st.session_state.user_data]
        
        if missing_essential:
            response_parts.append(f"\n📋 **Essential information still needed:** {', '.join(missing_essential)}")
            response_parts.append("Please provide these details for a complete assessment.")
        elif missing_additional:
            response_parts.append(f"\n📊 **Additional helpful information:** {', '.join(missing_additional)}")
            response_parts.append("You can provide more details or proceed with current information.")
        else:
            response_parts.append("\n✅ **Comprehensive information collected!** Ready for detailed risk assessment.")
        
        # Add AI-powered medical insight
        try:
            # Create enhanced context for AI analysis
            medical_context = f"Patient profile: {extracted_info}. Complete data: {st.session_state.user_data}"
            
            # Enhanced AI query for medical insights
            insight_query = f"""Based on the medical information provided ({extracted_info}), provide brief educational insights about:
            1. The significance of these values for heart disease risk
            2. Any notable risk factors identified
            3. General recommendations (not medical advice)
            
            Keep response concise and educational."""
            
            ai_insight = get_medical_ai_response(insight_query, medical_context)
            
            if ai_insight and len(ai_insight) > 30:
                response_parts.append(f"\n💡 **AI Medical Insight:**\n{ai_insight}")
        except Exception as e:
            # Fallback insight based on extracted information
            risk_factors = []
            if 'Age' in extracted_info and extracted_info['Age'] > 45:
                risk_factors.append("age-related risk increase")
            if 'RestBP' in extracted_info and extracted_info['RestBP'] >= 140:
                risk_factors.append("elevated blood pressure")
            if 'Chol' in extracted_info and extracted_info['Chol'] >= 240:
                risk_factors.append("high cholesterol")
            
            if risk_factors:
                response_parts.append(f"\n⚠️ **Notable factors:** {', '.join(risk_factors)} detected.")
        
        return "\n".join(response_parts)
    
    # Enhanced general query handling with AI
    try:
        # Create better context for general medical queries
        enhanced_context = context
        if st.session_state.user_data:
            data_summary = f"Patient has provided: {', '.join(st.session_state.user_data.keys())}"
            enhanced_context += f" | {data_summary}"
        
        return get_medical_ai_response(user_message, enhanced_context)
    except:
        return generate_enhanced_fallback_response(user_message, context)

# Chat interface function (enhanced)
def chat_interface():
    """Display the enhanced AI-powered chatbot interface."""
    st.subheader("🤖 AI-Powered Heart Disease Risk Assessment")
    
    # Add debugging info in expander
    with st.expander("🔧 Debug Information (for development)"):
        st.write("**Current Session Data:**")
        st.json(st.session_state.user_data)
        
        if st.button("Test Extraction"):
            test_text = "I'm a 58-year-old male with chest pain. My blood pressure is 150 mmHg and my cholesterol is 280 mg/dl."
            test_extracted = extract_medical_info(test_text)
            st.write("**Test Input:**", test_text)
            st.write("**Extracted:**", test_extracted)
    
    # Add information about the AI assistant
    with st.expander("🧠 About this Enhanced AI Assistant"):
        st.write("""
        **🔋 Powered by Mistral AI:**
        - Advanced natural language understanding
        - Medical knowledge from large language models
        - Context-aware conversations
        - Intelligent medical information extraction
        
        **🩺 Capabilities:**
        - Understands complex medical queries
        - Provides personalized health education
        - Remembers conversation context
        - Offers evidence-based insights
        
        **⚠️ Important:** This AI provides educational information only and does not replace professional medical consultation.
        """)
    
    # AI Status indicator
    col1, col2 = st.columns([3, 1])
    with col2:
        try:
            # Test API connectivity
            test_response = requests.get("https://api-inference.huggingface.co/", timeout=5)
            if test_response.status_code in [200, 404]:  # 404 is expected for root endpoint
                st.success("🟢 AI Online")
            else:
                st.warning("🟡 AI Limited")
        except:
            st.error("🔴 AI Offline")
    
    # Display chat history
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            # Show welcome message
            st.markdown("""
            <div style="text-align: left; margin: 10px 0;">
                <div style="background-color: #f8f9fa; color: #333; padding: 15px; border-radius: 20px; border-left: 4px solid #007bff;">
                    👋 Welcome! I'm your AI assistant for heart disease risk assessment.<br><br>
                    I can help you understand cardiovascular health and collect information for risk evaluation.<br><br>
                    Try asking: "What are heart disease symptoms?" or share your medical details.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        for i, (role, message, timestamp) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.markdown(f"""
                <div style="text-align: right; margin: 10px 0;">
                    <div style="background-color: #007bff; color: white; padding: 10px 15px; border-radius: 20px 20px 5px 20px; display: inline-block; max-width: 70%;">
                        {message}
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 2px;">
                        You • {timestamp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="text-align: left; margin: 10px 0;">
                    <div style="background-color: #f8f9fa; color: #333; padding: 10px 15px; border-radius: 20px 20px 20px 5px; display: inline-block; max-width: 70%; border-left: 4px solid #007bff;">
                        {message}
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 2px;">
                        AI Assistant • {timestamp}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Enhanced chat input with AI suggestions
    st.write("**🤖 Try these enhanced examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🩺 Detailed Symptoms", key="detailed_symptoms_btn"):
            user_input = "I'm experiencing chest pain during exercise, I'm 62 years old, male, with blood pressure around 160 and cholesterol of 240."
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    with col2:
        if st.button("📊 Medical Values", key="medical_values_btn"):
            user_input = "I'm a 45-year-old female. My doctor said my BP is 140 mmHg and cholesterol is 220 mg/dl. Should I be concerned?"
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    with col3:
        if st.button("💡 Risk Assessment", key="risk_assessment_btn"):
            user_input = "What does it mean if I have high blood pressure and cholesterol? I'm 55 years old."
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    # Enhanced chat input
    user_input = st.text_area(
        "🗣️ Describe your medical situation or ask questions:", 
        key="chat_input", 
        placeholder="Examples:\n• 'I'm a 58-year-old male with chest pain when exercising'\n• 'My blood pressure is 150 and cholesterol is 280'\n• 'What are the warning signs of heart disease?'",
        height=100
    )
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        send_button = st.button("Send 📤", key="send_chat")
    with col2:
        clear_button = st.button("Clear Chat 🗑️", key="clear_chat")
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.user_data = {}
        st.rerun()
    
    if send_button and user_input:
        # Add user message to history
        timestamp = datetime.now().strftime("%H:%M")
        st.session_state.chat_history.append(("user", user_input, timestamp))
        
        # Extract information and generate response
        extracted_info = extract_medical_info(user_input)
        bot_response = generate_smart_response(user_input, extracted_info)
        
        # Add bot response to history
        st.session_state.chat_history.append(("bot", bot_response, timestamp))
        
        # Clear input and rerun
        st.rerun()

# Streamlit app
st.title("🩺 AI-Powered Heart Disease Prediction System")
st.markdown("*Advanced cardiovascular risk assessment with AI assistance*")

# Add tabs for different interfaces
tab1, tab2 = st.tabs(["🤖 AI Chat Assistant", "📋 Detailed Assessment Form"])

with tab1:
    chat_interface()
    
    # Show extracted data if any
    if st.session_state.user_data:
        st.subheader("📝 Medical Information Collected")
        
        # Create a nice display of collected data
        info_cols = st.columns(2)
        data_items = list(st.session_state.user_data.items())
        
        for i, (key, value) in enumerate(data_items):
            col = info_cols[i % 2]
            with col:
                if key == 'Sex':
                    st.metric("Gender", "Male" if value == 1 else "Female")
                elif key == 'Age':
                    st.metric("Age", f"{value} years")
                elif key == 'RestBP':
                    st.metric("Blood Pressure", f"{value} mmHg")
                elif key == 'Chol':
                    st.metric("Cholesterol", f"{value} mg/dl")
                elif key == 'MaxHR':
                    st.metric("Max Heart Rate", f"{value} bpm")
                elif key == 'ChestPain':
                    st.metric("Chest Pain Type", value.title())
        
        # Quick prediction button
        if len(st.session_state.user_data) >= 4:
            st.success("✅ Sufficient information collected for risk assessment!")
            
            if st.button("🔍 Get AI-Powered Risk Assessment", key="quick_predict", type="primary"):
                # Fill in default values for missing fields
                default_values = {
                    'Fbs': 0, 'RestECG': 0, 'ExAng': 0, 'Oldpeak': 0.0,
                    'Slope': 2, 'Ca': 0, 'Thal': 'normal'
                }
                
                complete_data = {**default_values, **st.session_state.user_data}
                
                result = predict_heart_disease(complete_data)
                
                if result and result['success']:
                    st.balloons()
                    
                    # Display comprehensive results
                    st.subheader("🎯 Risk Assessment Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        color = "🔴" if result['prediction'] == 'Heart Disease' else "🟢"
                        st.metric("Prediction", f"{color} {result['prediction']}")
                    with col2:
                        st.metric("Risk Probability", f"{result['probability']:.1%}")
                    with col3:
                        risk_emoji = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
                        st.metric("Risk Level", f"{risk_emoji.get(result['risk_level'], '🟡')} {result['risk_level']}")
                    
                    # Add result to chat
                    timestamp = datetime.now().strftime("%H:%M")
                    result_message = f"""🎯 **AI Risk Assessment Complete!**

🔍 **Results:**
• **Prediction:** {result['prediction']}
• **Probability:** {result['probability']:.1%}
• **Risk Level:** {result['risk_level']}

💡 **Recommendations:**
{('⚠️ Immediate medical consultation recommended. Please see a cardiologist for further evaluation.' if result['prediction'] == 'Heart Disease' else '✅ Continue regular health monitoring and maintain healthy lifestyle habits.')}

**Next Steps:**
• Consult with your healthcare provider
• Regular monitoring and check-ups
• Maintain heart-healthy lifestyle
• Follow medical professional guidance"""
                    
                    st.session_state.chat_history.append(("bot", result_message, timestamp))

with tab2:
    st.write("Enter detailed patient information to assess heart disease risk")
    
    # Pre-fill form with chatbot data
    initial_values = st.session_state.user_data
    
    # Create input form
    with st.form("patient_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=20, max_value=100, 
                                value=initial_values.get('Age', 50))
            sex = st.selectbox("Sex", options=[(1, "Male"), (0, "Female")], 
                             index=initial_values.get('Sex', 1),
                             format_func=lambda x: x[1])
            
            chest_pain_options = ["typical", "nontypical", "nonanginal", "asymptomatic"]
            chest_pain_index = 0
            if 'ChestPain' in initial_values:
                try:
                    chest_pain_index = chest_pain_options.index(initial_values['ChestPain'])
                except ValueError:
                    chest_pain_index = 0
            
            chest_pain = st.selectbox("Chest Pain Type", 
                                     options=chest_pain_options,
                                     index=chest_pain_index)
            
            rest_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=220, 
                                    value=initial_values.get('RestBP', 120))
            chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, 
                                 value=initial_values.get('Chol', 200))
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", options=[(0, "No"), (1, "Yes")], 
                             format_func=lambda x: x[1])[0]
        
        with col2:
            rest_ecg = st.selectbox("Resting ECG Results", 
                                  options=[(0, "Normal"), 
                                          (1, "ST-T Wave Abnormality"), 
                                          (2, "Left Ventricular Hypertrophy")], 
                                  format_func=lambda x: x[1])[0]
            max_hr = st.number_input("Maximum Heart Rate", min_value=50, max_value=220, 
                                   value=initial_values.get('MaxHR', 150))
            ex_ang = st.selectbox("Exercise Induced Angina", options=[(0, "No"), (1, "Yes")], 
                                format_func=lambda x: x[1])[0]
            oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=10.0, 
                                    value=0.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", 
                               options=[(1, "Upsloping"), (2, "Flat"), (3, "Downsloping")], 
                               format_func=lambda x: x[1])[0]
            ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)
            thal = st.selectbox("Thalassemia", options=["normal", "fixed", "reversable"])
        
        submit_button = st.form_submit_button("🔍 Predict Heart Disease Risk", type="primary")
    
    # Make prediction when form is submitted
    if submit_button:
        user_data = {
            'Age': age,
            'Sex': sex[0],
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
        
        if result and result['success']:
            # Display result with nice formatting
            st.subheader("📊 Comprehensive Risk Assessment Results")
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Prediction", result['prediction'])
            
            with col2:
                st.metric("Probability", f"{result['probability']:.2f}")
            
            with col3:
                st.metric("Risk Level", result['risk_level'])
            
            # Add detailed recommendations based on the prediction
            st.subheader("💡 Medical Recommendations")
            
            if result['prediction'] == 'Heart Disease':
                st.error("⚠️ The assessment indicates potential heart disease risk.")
                st.write("**Immediate medical follow-up is strongly recommended including:**")
                st.write("- 🏥 Consultation with a cardiologist")
                st.write("- 🔬 Further diagnostic tests (ECG, stress test, etc.)")
                st.write("- 💊 Review of current lifestyle and medications")
            else:
                st.success("✅ The assessment shows low risk of heart disease.")
                st.write("**Continue with regular health check-ups and healthy lifestyle including:**")
                st.write("- 🏃‍♂️ Regular exercise (150 minutes moderate activity/week)")
                st.write("- 🥗 Heart-healthy diet (Mediterranean-style)")
                st.write("- 📊 Regular monitoring of blood pressure and cholesterol")
                st.write("- 🚭 Avoiding smoking and limiting alcohol consumption")
        else:
            st.error(f"❌ Error in prediction: {result['error'] if result else 'Unknown error'}")
            st.write("Please check your inputs and try again.")

# Add footer with medical disclaimer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9em; padding: 20px;">
    <strong>⚠️ Medical Disclaimer:</strong> This tool is for educational and informational purposes only. 
    It does not constitute medical advice and should not replace consultation with qualified healthcare professionals. 
    Always seek professional medical advice for diagnosis and treatment decisions.
</div>
""", unsafe_allow_html=True)