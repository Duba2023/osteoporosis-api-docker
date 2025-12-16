import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from groq import Groq
from dotenv import load_dotenv

# ===============================
# LOAD ENV VARIABLES
# ===============================
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Osteoporosis Risk Assessment",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# GROQ CLIENT
# ===============================
def get_groq_client():
    if not api_key:
        st.error(
            """
            ⚠️ Groq API key not found.  
            Set your environment variable GROQ_API_KEY.
            """
        )
        return None
    try:
        return Groq(api_key=api_key)
    except Exception as e:
        st.error(f"⚠️ Failed to initialize Groq client: {e}")
        return None

client = get_groq_client()

# ===============================
# LOAD MODEL & SCALER
# ===============================
@st.cache_resource
def load_model_and_scaler():
    model_path = "best_osteoporosis_model.joblib"
    scaler_path = "scaler.joblib"
    
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found!")
        st.stop()
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
    return model, scaler

model, scaler = load_model_and_scaler()

FEATURE_ORDER = [
    'Age', 'Hormonal Changes', 'Family History', 'Body Weight',
    'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking',
    'Prior Fractures', 'Gender_Female', 'Gender_Male',
    'Medications_Corticosteroids', 'Medications_Unknown',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medical Conditions_Unknown', 'Alcohol Consumption_Moderate',
    'Alcohol Consumption_Unknown'
]

# ===============================
# HEADER
# ===============================
st.markdown("<h1 style='text-align:center'>🦴 Osteoporosis Risk Assessment</h1>", unsafe_allow_html=True)

# ===============================
# TABS
# ===============================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 Input",
    "📊 Results",
    "📤 Batch CSV",
    "ℹ️ About",
    "🤖 AI Assistant (Chatbot)"
])

# ===============================
# USER INPUT
# ===============================
st.sidebar.header("📝 User Input")

def user_input_features():
    age = st.sidebar.slider("Age", 18, 90, 55)
    gender = st.sidebar.radio("Gender", ["Female", "Male"])
    hormonal = st.sidebar.radio("Hormonal Changes", ["Normal", "Postmenopausal"])
    family = st.sidebar.radio("Family History", ["No", "Yes"])
    fracture = st.sidebar.radio("Prior Fractures", ["No", "Yes"])
    smoking = st.sidebar.radio("Smoking", ["No", "Yes"])

    data = {
        'Age': age,
        'Hormonal Changes': 1 if hormonal == "Postmenopausal" else 0,
        'Family History': 1 if family == "Yes" else 0,
        'Body Weight': 0,
        'Calcium Intake': 0,
        'Vitamin D Intake': 0,
        'Physical Activity': 0,
        'Smoking': 1 if smoking == "Yes" else 0,
        'Prior Fractures': 1 if fracture == "Yes" else 0,
        'Gender_Female': 1 if gender == "Female" else 0,
        'Gender_Male': 1 if gender == "Male" else 0,
        'Medications_Corticosteroids': 0,
        'Medications_Unknown': 1,
        'Medical Conditions_Hyperthyroidism': 0,
        'Medical Conditions_Rheumatoid Arthritis': 0,
        'Medical Conditions_Unknown': 1,
        'Alcohol Consumption_Moderate': 0,
        'Alcohol Consumption_Unknown': 1
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()[FEATURE_ORDER]

# ===============================
# INPUT REVIEW
# ===============================
with tab1:
    st.dataframe(input_df, use_container_width=True)

# ===============================
# PREDICTION
# ===============================
predict_btn = st.sidebar.button("🔍 Analyze Risk")

if predict_btn:
    X = input_df.copy()
    if scaler:
        X = scaler.transform(X)
        X = pd.DataFrame(X, columns=FEATURE_ORDER)

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    st.session_state.prediction = pred
    st.session_state.proba = proba

# ===============================
# RESULTS TAB
# ===============================
with tab2:
    if st.session_state.get("prediction") is not None:
        risk = "High Risk" if st.session_state.prediction == 1 else "Low Risk"
        high_prob = st.session_state.proba[1] * 100
        st.success(f"Risk Level: **{risk}**")
        st.info(f"High Risk Probability: **{high_prob:.1f}%**")
    else:
        st.info("Click 'Analyze Risk' in the sidebar to see your results.")

# ===============================
# SHOW RESULTS IN INPUT TAB
# ===============================
with tab1:
    if st.session_state.get("prediction") is not None:
        risk = "High Risk" if st.session_state.prediction == 1 else "Low Risk"
        high_prob = st.session_state.proba[1] * 100
        st.success(f"Risk Level: **{risk}**")
        st.info(f"High Risk Probability: **{high_prob:.1f}%**")

# ===============================
# ABOUT TAB
# ===============================
with tab4:
    st.markdown("""
    **Disclaimer:**  
    This app provides educational information only and is not a medical diagnosis.
    """)

# ===============================
# AI ASSISTANT TAB - Auto-scroll version
# ===============================
with tab5:
    st.subheader("🤖 AI Assistant")
    st.caption("Educational use only. Not medical advice. Ask anything you like!")

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Context from prediction
    risk_context = ""
    if "prediction" in st.session_state:
        risk_level = "High Risk" if st.session_state.prediction == 1 else "Low Risk"
        high_prob = st.session_state.proba[1] * 100
        risk_context = f"Osteoporosis result: {risk_level}, Probability: {high_prob:.1f}%"

    system_prompt = f"""
    You are a helpful AI assistant embedded in a health assessment app.
    Answer general, educational, or medical-related questions clearly.
    Do NOT provide personalized medical advice.
    Context: {risk_context}
    """

    # Container for chat messages
    chat_container = st.container()
    # Placeholder for auto-scroll
    scroll_anchor = st.empty()

    # Display previous chat messages
    with chat_container:
        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # Chat input with unique key
    user_msg = st.chat_input("Ask me anything...", key="chat_input_unique")

    if user_msg:
        # Append user message
        st.session_state.chat.append({"role": "user", "content": user_msg})

        with chat_container:
            st.chat_message("user").markdown(user_msg)

            # Placeholder for assistant reply
            placeholder = st.empty()
            with st.chat_message("assistant"):
                placeholder.markdown("…thinking…")

            # Call Groq API
            if client:
                try:
                    response = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_msg}
                        ]
                    )
                    reply = response.choices[0].message.content
                except Exception as e:
                    reply = f"⚠️ AI assistant error: {str(e)}"
            else:
                reply = "⚠️ Groq API key is missing."

            # Update placeholder with reply
            placeholder.markdown(reply)

            # Save assistant reply
            st.session_state.chat.append({"role": "assistant", "content": reply})

            # Auto-scroll
            scroll_anchor.empty()
