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
    Extract medical information from user text input.
    
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
    
    # Extract age
    age_match = re.search(r'\b(\d{1,3})\s*(?:years?\s*old|yo|age)\b', text_lower)
    if age_match:
        age = int(age_match.group(1))
        if 20 <= age <= 100:
            info['Age'] = age
    
    # Extract gender
    if any(word in text_lower for word in ['male', 'man', 'men', 'boy']):
        info['Sex'] = 1
    elif any(word in text_lower for word in ['female', 'woman', 'women', 'girl']):
        info['Sex'] = 0
    
    # Extract chest pain type
    if any(word in text_lower for word in ['typical angina', 'typical chest pain']):
        info['ChestPain'] = 'typical'
    elif any(word in text_lower for word in ['atypical angina', 'atypical chest pain']):
        info['ChestPain'] = 'nontypical'
    elif any(word in text_lower for word in ['non-anginal', 'nonanginal']):
        info['ChestPain'] = 'nonanginal'
    elif any(word in text_lower for word in ['asymptomatic', 'no chest pain']):
        info['ChestPain'] = 'asymptomatic'
    elif 'chest pain' in text_lower:
        info['ChestPain'] = 'typical'  # default assumption
    
    # Extract blood pressure
    bp_match = re.search(r'\b(\d{2,3})\s*(?:mmhg|mm hg|blood pressure|bp)\b', text_lower)
    if bp_match:
        bp = int(bp_match.group(1))
        if 80 <= bp <= 220:
            info['RestBP'] = bp
    
    # Extract cholesterol
    chol_match = re.search(r'\b(\d{2,3})\s*(?:mg/dl|cholesterol|chol)\b', text_lower)
    if chol_match:
        chol = int(chol_match.group(1))
        if 100 <= chol <= 600:
            info['Chol'] = chol
    
    # Extract heart rate
    hr_match = re.search(r'\b(\d{2,3})\s*(?:bpm|heart rate|hr|max hr)\b', text_lower)
    if hr_match:
        hr = int(hr_match.group(1))
        if 50 <= hr <= 220:
            info['MaxHR'] = hr
    
    return info

def get_ai_response(user_message, context=""):
    """
    Get response from Hugging Face Mistral model for more intelligent conversations
    
    Parameters:
    -----------
    user_message : str
        User's message
    context : str
        Additional context about the conversation
        
    Returns:
    --------
    str
        AI-generated response
    """
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create a medical-focused prompt for Mistral
        system_prompt = """You are a helpful medical AI assistant specializing in heart disease risk assessment. 
        You provide educational information about cardiovascular health, risk factors, and symptoms. 
        Always remind users to consult healthcare professionals for medical advice.
        Be empathetic, informative, and encouraging. Keep responses concise and helpful."""
        
        # Format the prompt for Mistral
        if context:
            full_prompt = f"<s>[INST] {system_prompt}\n\nContext: {context}\n\nUser question: {user_message} [/INST]"
        else:
            full_prompt = f"<s>[INST] {system_prompt}\n\nUser question: {user_message} [/INST]"
        
        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '').strip()
                
                # Clean up the response
                if ai_response:
                    # Remove system prompt repetition if present
                    ai_response = ai_response.replace(system_prompt, "").strip()
                    # Add medical disclaimer if not present
                    if "consult" not in ai_response.lower() and "medical advice" not in ai_response.lower():
                        ai_response += "\n\n⚠️ Please consult healthcare professionals for medical advice."
                    
                    return ai_response
        
        # Fallback to rule-based response
        return generate_enhanced_fallback_response(user_message, context)
        
    except requests.exceptions.Timeout:
        st.warning("🤖 AI response is taking longer than usual. Using fallback response.")
        return generate_enhanced_fallback_response(user_message, context)
    except Exception as e:
        st.warning(f"🤖 AI service temporarily unavailable. Using enhanced fallback.")
        return generate_enhanced_fallback_response(user_message, context)

def get_medical_ai_response(user_message, context=""):
    """
    Get specialized medical response using Mistral model
    """
    try:
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        # Create medical prompt for Mistral
        medical_prompt = f"""<s>[INST] You are a medical AI assistant for heart disease risk assessment. 
        
        Context: {context}
        Patient Query: {user_message}

        Provide helpful, educational information about:
        - Heart disease risk factors
        - Symptoms to monitor
        - Lifestyle recommendations
        - When to seek medical care

        Keep the response concise, informative, and always remind to consult healthcare professionals. [/INST]"""
        
        payload = {
            "inputs": medical_prompt,
            "parameters": {
                "max_new_tokens": 150,
                "temperature": 0.6,
                "top_p": 0.8,
                "do_sample": True,
                "return_full_text": False
            }
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=15)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                ai_response = result[0].get('generated_text', '').strip()
                
                if ai_response and len(ai_response) > 20:
                    return ai_response
        
        # Fallback to general conversational AI
        return get_ai_response(user_message, context)
        
    except Exception as e:
        return get_ai_response(user_message, context)

def generate_enhanced_fallback_response(user_message, context=""):
    """
    Enhanced fallback response system with better medical knowledge
    """
    message_lower = user_message.lower()
    
    # Greeting responses
    if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'greetings']):
        return """👋 Hello! I'm your AI-powered Heart Disease Risk Assessment Assistant.

🩺 **I can help you with:**
• Understanding heart disease risk factors
• Analyzing symptoms and medical information
• Providing educational content about cardiovascular health
• Guiding you through risk assessment

💬 **Try asking me:**
• "What are the signs of heart disease?"
• "How does age affect heart disease risk?"
• Share your medical details for assessment

How can I assist you with your cardiovascular health today?"""
    
    # Help and capability responses
    elif any(word in message_lower for word in ['help', 'what can you do', 'how does this work', 'capabilities']):
        return """🤖 **AI Assistant Capabilities:**
        - Understands complex medical queries
        - Provides personalized health education
        - Remembers conversation context
        - Offers evidence-based insights
        
        **🩺 I can help with:**
        - **Risk Assessment:** Share your medical details for analysis
        - **Symptom Analysis:** Describe what you're experiencing
        - **Educational Info:** Learn about heart disease prevention
        - **Lifestyle Guidance:** Get personalized recommendations
        
        **Example:** "I'm 58 years old, male, with high blood pressure 160 mmHg and cholesterol 280. I get chest pain when exercising."

        What would you like to know about heart disease risk?"""
    
    # Symptoms and risk factors
    elif any(word in message_lower for word in ['symptoms', 'signs', 'risk factors', 'warning signs']):
        return """🚨 **Heart Disease Warning Signs & Risk Factors:**

**🔴 Immediate Warning Signs:**
• Chest pain, pressure, or tightness
• Pain radiating to arms, neck, jaw, back
• Shortness of breath
• Cold sweats, nausea, lightheadedness
• Fatigue or weakness

**⚠️ Major Risk Factors:**
• **Age:** Men ≥45, Women ≥55
• **High Blood Pressure:** ≥140/90 mmHg
• **High Cholesterol:** ≥240 mg/dl
• **Smoking & Diabetes**
• **Family History**
• **Obesity & Physical Inactivity**

**📊 Modifiable Risk Factors:**
• Diet high in saturated fats
• Lack of exercise
• Stress and poor sleep
• Excessive alcohol consumption

🆘 **Seek immediate medical attention** if experiencing severe chest pain, difficulty breathing, or signs of heart attack.

Would you like to share any symptoms or risk factors you're concerned about?"""
    
    # Chest pain specific responses
    elif 'chest pain' in message_lower or 'chest discomfort' in message_lower:
        return """💔 **Understanding Chest Pain Types:**

**🔴 Typical Angina (Classic):**
• Pressure, squeezing, or burning in chest
• Triggered by physical exertion or stress
• Relieved by rest or medication
• May radiate to arms, neck, jaw

**🟡 Atypical Angina:**
• Similar to typical but with unusual features
• May occur at rest
• Different triggers or relief patterns

**🟢 Non-Anginal Chest Pain:**
• Sharp, stabbing, or localized
• Not related to exertion
• Lasts seconds or hours
• Often musculoskeletal

**⚪ Asymptomatic:**
• No chest pain symptoms
• Silent heart disease possible
• Other symptoms may be present

**🚨 Emergency Signs:**
• Crushing chest pressure
• Pain with sweating, nausea
• Severe shortness of breath
• Pain lasting >20 minutes

Can you describe your chest pain in more detail? When does it occur?"""
    
    # Blood pressure information
    elif any(word in message_lower for word in ['blood pressure', 'bp', 'hypertension', 'pressure']):
        return """🩺 **Blood Pressure & Heart Disease:**

**📊 Blood Pressure Categories:**
• **Normal:** <120/80 mmHg
• **Elevated:** 120-129/<80 mmHg
• **High Stage 1:** 130-139/80-89 mmHg
• **High Stage 2:** ≥140/90 mmHg
• **Crisis:** >180/120 mmHg (Emergency!)

**💔 Impact on Heart Disease:**
High blood pressure makes your heart work harder and can damage arteries over time, significantly increasing heart disease risk.

**🎯 Management Strategies:**
• **Diet:** Reduce sodium, increase potassium
• **Exercise:** 150 mins moderate activity/week
• **Weight:** Maintain healthy BMI
• **Stress:** Practice relaxation techniques
• **Medication:** As prescribed by doctor

**📈 Monitoring Tips:**
• Check at same time daily
• Rest 5 minutes before measuring
• Use proper cuff size
• Keep a log for your doctor

What's your current blood pressure reading?"""
    
    # Cholesterol information
    elif any(word in message_lower for word in ['cholesterol', 'chol', 'lipids']):
        return """🧪 **Cholesterol & Heart Health:**

**📊 Optimal Levels:**
• **Total Cholesterol:** <200 mg/dL
• **LDL (Bad):** <100 mg/dL (optimal)
• **HDL (Good):** >40 mg/dL (men), >50 mg/dL (women)
• **Triglycerides:** <150 mg/dL

**⚠️ Risk Categories:**
• **Borderline High:** 200-239 mg/dL total
• **High:** ≥240 mg/dL total
• **Very High LDL:** ≥190 mg/dL

**🔄 How It Affects Heart:**
• LDL builds up in artery walls
• Creates plaques that narrow arteries
• Can rupture and cause clots
• Leads to heart attacks and strokes

**🥗 Natural Management:**
• **Eat:** Oats, fish, nuts, olive oil
• **Avoid:** Trans fats, processed foods
• **Exercise:** Raises HDL, lowers LDL
• **Weight Loss:** Improves all levels

Do you know your current cholesterol numbers?"""
    
    # Age and gender factors
    elif any(word in message_lower for word in ['age', 'gender', 'sex', 'older', 'aging']):
        return """👥 **Age & Gender in Heart Disease Risk:**

**📈 Age Factors:**
• **Men:** Risk increases after age 45
• **Women:** Risk increases after menopause (55+)
• **Elderly:** Highest risk group (65+)
• **Young Adults:** Lower risk but not immune

**♂️ Men vs ♀️ Women:**
• **Men:** Earlier onset, more obvious symptoms
• **Women:** Often atypical symptoms
• **Post-menopause:** Women's risk equals men's
• **Hormones:** Estrogen protective before menopause

**🔄 Age-Related Changes:**
• Arteries stiffen and narrow
• Heart muscle weakens
• Blood pressure tends to rise
• Cholesterol levels may increase

**💪 Prevention at Any Age:**
• Never too late to start healthy habits
• Exercise adapted to ability
• Regular medical check-ups
• Medication compliance

What's your age and gender? This helps assess your baseline risk."""
    
    # Lifestyle and prevention
    elif any(word in message_lower for word in ['lifestyle', 'prevention', 'exercise', 'diet', 'healthy']):
        return """💪 **Heart-Healthy Lifestyle:**

**🏃‍♂️ Exercise Guidelines:**
• **Aerobic:** 150 mins moderate OR 75 mins vigorous/week
• **Strength:** 2+ days/week muscle strengthening
• **Daily:** Any movement better than none
• **Examples:** Walking, swimming, cycling, dancing

**🥗 Heart-Healthy Diet:**
• **Mediterranean Style:** Fish, olive oil, nuts, fruits
• **Limit:** Saturated fats, trans fats, sodium
• **Increase:** Fiber, potassium, omega-3s
• **Portions:** Control serving sizes

**🚭 Lifestyle Modifications:**
• **No Smoking:** #1 preventable risk factor
• **Moderate Alcohol:** ≤1 drink/day (women), ≤2 (men)
• **Stress Management:** Yoga, meditation, hobbies
• **Sleep:** 7-9 hours quality sleep nightly

**📊 Regular Monitoring:**
• Blood pressure checks
• Cholesterol screening
• Blood sugar testing
• Weight management

Which aspect of heart-healthy living interests you most?"""
    
    # Default comprehensive response
    else:
        return f"""🤖 **AI Heart Health Assistant Ready!**

I understand you're asking about: "{user_message}"

🩺 **I can help with:**
• **Risk Assessment:** Share your medical details for analysis
• **Symptom Analysis:** Describe what you're experiencing
• **Educational Info:** Learn about heart disease prevention
• **Lifestyle Guidance:** Get personalized recommendations

📝 **For best results, try:**
• "I'm [age] years old, [gender], with [symptoms/conditions]"
• "What does blood pressure 150/90 mean for heart risk?"
• "I have chest pain when exercising, should I worry?"

💡 **Quick Links:**
• Heart disease symptoms and warning signs
• Blood pressure and cholesterol information
• Exercise and diet recommendations
• When to seek medical care

What specific aspect of heart health would you like to explore?

⚠️ **Remember:** This is educational information only. Always consult healthcare professionals for medical advice and diagnosis."""

def generate_smart_response(user_message, extracted_info=None):
    """
    Enhanced intelligent response combining Hugging Face AI and medical data extraction
    """
    # Build context from conversation history and extracted data
    context_parts = []
    
    if st.session_state.user_data:
        context_parts.append(f"Patient data: {st.session_state.user_data}")
    
    if st.session_state.chat_history:
        recent_history = st.session_state.chat_history[-3:]  # Last 3 exchanges
        context_parts.append("Recent conversation: " + " | ".join([f"{role}: {msg}" for role, msg, _ in recent_history]))
    
    context = " | ".join(context_parts)
    
    # If we extracted medical information, format it and update session
    if extracted_info:
        response_parts = ["🩺 **Medical Information Recorded:**\n"]
        
        for key, value in extracted_info.items():
            if key == 'Sex':
                response_parts.append(f"• **Gender:** {'Male' if value == 1 else 'Female'}")
            elif key == 'Age':
                response_parts.append(f"• **Age:** {value} years")
            elif key == 'ChestPain':
                response_parts.append(f"• **Chest Pain Type:** {value.title()}")
            elif key == 'RestBP':
                bp_status = "High" if value >= 140 else "Elevated" if value >= 120 else "Normal"
                response_parts.append(f"• **Blood Pressure:** {value} mmHg ({bp_status})")
            elif key == 'Chol':
                chol_status = "High" if value >= 240 else "Borderline" if value >= 200 else "Normal"
                response_parts.append(f"• **Cholesterol:** {value} mg/dl ({chol_status})")
            elif key == 'MaxHR':
                response_parts.append(f"• **Max Heart Rate:** {value} bpm")
        
        # Update session state
        st.session_state.user_data.update(extracted_info)
        
        # Check completeness
        required_fields = ['Age', 'Sex', 'ChestPain', 'RestBP', 'Chol', 'MaxHR']
        missing_info = [field for field in required_fields if field not in st.session_state.user_data]
        
        if missing_info:
            response_parts.append(f"\n📋 **Still needed:** {', '.join(missing_info)}")
            response_parts.append("Please provide more details or use the detailed form below.")
        else:
            response_parts.append("\n✅ **Complete information collected!** Ready for risk assessment.")
        
        # Add AI insight about the provided information
        try:
            ai_context = f"Patient provided: {extracted_info}. Existing data: {st.session_state.user_data}"
            ai_insight = get_medical_ai_response(
                f"Analyze this medical information and provide brief educational insights: {extracted_info}",
                ai_context
            )
            if ai_insight and len(ai_insight) > 20:
                response_parts.append(f"\n💡 **AI Insight:** {ai_insight}")
        except:
            pass
        
        return "\n".join(response_parts)
    
    # For general queries, use AI response
    try:
        return get_medical_ai_response(user_message, context)
    except:
        return generate_enhanced_fallback_response(user_message, context)

# Chat interface function (enhanced)
def chat_interface():
    """Display the enhanced AI-powered chatbot interface."""
    st.subheader("🤖 AI-Powered Heart Disease Risk Assessment")
    
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
    st.write("**🤖 AI-Powered Examples:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🩺 Heart Disease Symptoms", key="symptoms_btn"):
            user_input = "What are the key symptoms and warning signs of heart disease I should watch for?"
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    with col2:
        if st.button("📊 Risk Factors", key="risk_btn"):
            user_input = "What are the main risk factors for heart disease and how can I reduce my risk?"
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    with col3:
        if st.button("💪 Prevention Tips", key="prevention_btn"):
            user_input = "How can I prevent heart disease through lifestyle changes and what should I monitor?"
            timestamp = datetime.now().strftime("%H:%M")
            st.session_state.chat_history.append(("user", user_input, timestamp))
            extracted_info = extract_medical_info(user_input)
            bot_response = generate_smart_response(user_input, extracted_info)
            st.session_state.chat_history.append(("bot", bot_response, timestamp))
            st.rerun()
    
    # Enhanced chat input
    user_input = st.text_area(
        "🗣️ Ask your AI assistant about heart health:", 
        key="chat_input", 
        placeholder="e.g., 'I'm 58 years old, male, with high blood pressure 160/95 and cholesterol 280. I sometimes get chest pain during exercise. What's my risk?'",
        height=80
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