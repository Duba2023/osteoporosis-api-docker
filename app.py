
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

# ===== PAGE CONFIGURATION =====
st.set_page_config(
    page_title="Osteoporosis Risk Assessment",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "### Osteoporosis Detection AI System\nPowered by Machine Learning | v1.0"
    }
)

# ===== CUSTOM STYLING =====
st.markdown("""
    <style>
    .main {
        padding-top: 2rem;
    }
    .header-title {
        font-size: 2.8rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    .risk-low {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        color: #0c5460;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ===== MODEL & SCALER LOADING WITH ERROR HANDLING =====
@st.cache_resource
def load_model_and_scaler():
    """Load the pre-trained osteoporosis model and scaler with error handling"""
    model_path = 'best_osteoporosis_model.joblib'
    scaler_path = 'scaler.joblib'
    
    # Load model
    if not os.path.exists(model_path):
        st.error(f"❌ Critical Error: Model file '{model_path}' not found!")
        st.info("Please ensure the model file is in the same directory as this application.")
        st.stop()
    
    try:
        model = joblib.load(model_path)
        st.sidebar.success("✅ Model loaded successfully")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Load scaler
    if not os.path.exists(scaler_path):
        st.warning(f"⚠️ Warning: Scaler file '{scaler_path}' not found. Using model without scaling.")
        scaler = None
    else:
        try:
            scaler = joblib.load(scaler_path)
            st.sidebar.success("✅ Scaler loaded successfully")
        except Exception as e:
            st.warning(f"⚠️ Warning: Failed to load scaler: {str(e)}")
            scaler = None
    
    return model, scaler

model, scaler = load_model_and_scaler()

# Define expected feature order (CRITICAL for XGBoost model)
FEATURE_ORDER = [
    'Age', 'Hormonal Changes', 'Family History', 'Body Weight',
    'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking',
    'Prior Fractures', 'Gender_Female', 'Gender_Male',
    'Medications_Corticosteroids', 'Medications_Unknown',
    'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
    'Medical Conditions_Unknown', 'Alcohol Consumption_Moderate',
    'Alcohol Consumption_Unknown'
]

# ===== HEADER SECTION =====
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<h1 class="header-title">🦴 Osteoporosis Risk Assessment</h1>', unsafe_allow_html=True)
    st.markdown('<p class="header-subtitle">AI-Powered Health Analysis System</p>', unsafe_allow_html=True)

# ===== INFORMATION TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["📋 Input Information", "📊 Results", "📤 Batch CSV Upload", "ℹ️ About & Guidelines"])



# ===== BATCH CSV UPLOAD SECTION =====
with tab3:
    st.subheader("📤 Batch Prediction with CSV Upload")
    
    st.markdown("""
    Upload a CSV file with patient records to get predictions for multiple patients at once.
    
    **Supported Features (optional):**
    The model can work with any subset of these 18 features. Missing features will use default values.
    """)
    
    # Display available features
    st.info("""
    **Available Features:**
    - Age, Hormonal Changes, Family History, Body Weight, Calcium Intake
    - Vitamin D Intake, Physical Activity, Smoking, Prior Fractures
    - Gender_Female, Gender_Male
    - Medications_Corticosteroids, Medications_Unknown
    - Medical Conditions_Hyperthyroidism, Medical Conditions_Rheumatoid Arthritis, Medical Conditions_Unknown
    - Alcohol Consumption_Moderate, Alcohol Consumption_Unknown
    """)
    
    # CSV upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="csv_upload")
    
    if uploaded_file is not None:
        try:
            # Read CSV with flexible settings
            try:
                df = pd.read_csv(uploaded_file)
            except Exception as read_error:
                st.error(f"❌ Error reading CSV: {str(read_error)}")
                st.info("Please ensure the file is a valid CSV format.")
                st.stop()
            
            # Handle empty dataframe
            if df.empty:
                st.error("❌ CSV file is empty. Please upload a file with data.")
                st.stop()
            
            st.success(f"✅ CSV uploaded successfully! Records: {len(df)}")
            
            # Show preview
            st.subheader("📋 Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Display data quality info
            st.info(f"📊 Dataset Info: {len(df)} rows, {len(df.columns)} columns")
            
            # All supported columns
            all_columns = [
                'Age', 'Hormonal Changes', 'Family History', 'Body Weight',
                'Calcium Intake', 'Vitamin D Intake', 'Physical Activity', 'Smoking',
                'Prior Fractures', 'Gender_Female', 'Gender_Male',
                'Medications_Corticosteroids', 'Medications_Unknown',
                'Medical Conditions_Hyperthyroidism', 'Medical Conditions_Rheumatoid Arthritis',
                'Medical Conditions_Unknown', 'Alcohol Consumption_Moderate',
                'Alcohol Consumption_Unknown'
            ]
            
            # Find available columns in the uploaded file
            available_cols = [col for col in all_columns if col in df.columns]
            missing_cols = [col for col in all_columns if col not in df.columns]
            
            if not available_cols:
                st.error("❌ No matching columns found in your CSV.")
                st.info("Please upload a CSV with at least one of the supported features.")
                st.stop()
            
            st.success(f"✅ Found {len(available_cols)}/{len(all_columns)} matching features")
            
            try:
                # Create a dataframe with all required columns, filling missing ones with 0
                df_prepared = df.copy()
                
                # Remove duplicate rows if any
                df_prepared = df_prepared.drop_duplicates()
                
                # Add missing columns with default value 0
                for col in missing_cols:
                    df_prepared[col] = 0
                
                # Convert all columns to numeric, handling any non-numeric values
                cleaned_rows = 0
                problematic_cols = []
                
                for col in all_columns:
                    original_non_null = df_prepared[col].notna().sum()
                    
                    # Convert to numeric with coercion
                    df_prepared[col] = pd.to_numeric(df_prepared[col], errors='coerce')
                    
                    # Count how many values couldn't be converted
                    nan_count = df_prepared[col].isna().sum()
                    
                    # Fill NaN values with 0
                    df_prepared[col] = df_prepared[col].fillna(0)
                    
                    if nan_count > 0:
                        problematic_cols.append(f"{col} ({nan_count} non-numeric values)")
                
                # Show data cleaning summary only if there are significant issues
                if problematic_cols and len(problematic_cols) <= 3:
                    with st.expander("📝 Data Cleaning Details", expanded=False):
                        for issue in problematic_cols:
                            st.write(f"  • {issue}")
                
                # Remove rows that are completely empty/zero
                df_prepared = df_prepared[(df_prepared != 0).any(axis=1)]
                
                if df_prepared.empty:
                    st.error("❌ No valid data found after cleaning.")
                    st.stop()
                
                st.success(f"✅ Data cleaned! Processing {len(df_prepared)} valid records")
                
                # Ensure correct column order
                df_ordered = df_prepared[all_columns]
                
                # Validate data ranges (optional warning)
                age_col_idx = all_columns.index('Age')
                if (df_ordered['Age'] < 0).any() or (df_ordered['Age'] > 150).any():
                    st.warning("⚠️ Some Age values are outside normal range (0-150). They will be used as-is.")
                
                # Apply scaler if available
                if scaler is not None:
                    try:
                        df_scaled = scaler.transform(df_ordered)
                        df_scaled = pd.DataFrame(df_scaled, columns=all_columns)
                    except Exception as scale_error:
                        st.warning(f"⚠️ Scaling not applied: {str(scale_error)[:100]}. Using unscaled data.")
                        df_scaled = df_ordered
                else:
                    df_scaled = df_ordered
                
                # Make predictions
                try:
                    predictions = model.predict(df_scaled)
                    probabilities = model.predict_proba(df_scaled)
                    
                    # Create results dataframe
                    results_df = df_prepared.copy()
                    results_df['Risk_Level'] = ['High Risk' if pred == 1 else 'Low Risk' for pred in predictions]
                    results_df['Low_Risk_Probability'] = probabilities[:, 0]
                    results_df['High_Risk_Probability'] = probabilities[:, 1]
                    results_df['Confidence_%'] = np.max(probabilities, axis=1) * 100
                    
                    st.subheader("🎯 Batch Prediction Results")
                    
                    # Summary statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    total_records = len(results_df)
                    high_risk = len(results_df[results_df['Risk_Level'] == 'High Risk'])
                    low_risk = len(results_df[results_df['Risk_Level'] == 'Low Risk'])
                    avg_confidence = results_df['Confidence_%'].mean()
                    
                    col1.metric("Total Records", total_records)
                    col2.metric("High Risk", high_risk, f"{(high_risk/total_records*100):.1f}%")
                    col3.metric("Low Risk", low_risk, f"{(low_risk/total_records*100):.1f}%")
                    col4.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                    
                    st.divider()
                    
                    # Results table
                    st.subheader("📊 Detailed Results")
                    result_cols = ['Risk_Level', 'Low_Risk_Probability', 'High_Risk_Probability', 'Confidence_%']
                    st.dataframe(results_df[result_cols], use_container_width=True)
                    
                    # Risk distribution chart
                    st.subheader("📈 Risk Distribution")
                    risk_counts = results_df['Risk_Level'].value_counts()
                    st.bar_chart(risk_counts)
                    
                    # Download results
                    st.subheader("💾 Export Results")
                    csv_results = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_results,
                        file_name=f"osteoporosis_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Detailed statistics
                    st.subheader("📊 Detailed Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Probability Statistics:**")
                        st.dataframe(results_df[['Low_Risk_Probability', 'High_Risk_Probability', 'Confidence_%']].describe())
                    
                    with col2:
                        st.write("**Risk Stratification:**")
                        stratification = pd.DataFrame({
                            'Risk Level': ['High Risk', 'Low Risk'],
                            'Count': [high_risk, low_risk],
                            'Percentage': [f"{(high_risk/total_records*100):.1f}%", f"{(low_risk/total_records*100):.1f}%"]
                        })
                        st.dataframe(stratification, use_container_width=True, hide_index=True)
                
                except Exception as pred_error:
                    st.error(f"❌ Prediction error: {str(pred_error)}")
                    st.info("🔧 Troubleshooting tips:")
                    st.write("  • Ensure numeric values in all columns")
                    st.write("  • Check for infinity or NaN values")
                    st.write("  • Try uploading a smaller sample first")
            
            except Exception as prep_error:
                st.error(f"❌ Data preparation error: {str(prep_error)}")
                st.info("Please check your CSV format and try again.")
        
        except Exception as e:
            st.error(f"❌ Error reading CSV file: {str(e)}")
            st.info("Please ensure the file is a valid CSV format.")

# ===== ABOUT & GUIDELINES SECTION =====
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏥 About Osteoporosis")
        st.markdown("""
        **Osteoporosis** is a medical condition characterized by weakened bones that become fragile and more susceptible to fractures.
        
        **Key Facts:**
        - Affects over 200 million people worldwide
        - Women are at higher risk, especially postmenopausal women
        - Often called a "silent disease" - no symptoms until fracture occurs
        - Most common fractures: hip, spine, and wrist
        """)
    
    with col2:
        st.subheader("⚠️ Risk Factors")
        st.markdown("""
        - **Age & Gender**: Risk increases with age, women at higher risk
        - **Hormonal Changes**: Postmenopausal status increases risk
        - **Lifestyle**: Low calcium, vitamin D, and physical activity
        - **Family History**: Genetic predisposition
        - **Habits**: Smoking and excessive alcohol consumption
        """)
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Prevention Tips")
        st.markdown("""
        1. **Nutrition**: Consume adequate calcium (1000-1200 mg/day)
        2. **Vitamin D**: Maintain sufficient vitamin D levels
        3. **Exercise**: Engage in weight-bearing activities
        4. **Lifestyle**: Avoid smoking and limit alcohol
        5. **Screening**: Regular bone density tests if at risk
        """)
    
    with col2:
        st.subheader("⚕️ Important Disclaimer")
        st.markdown("""
        <div class="info-box">
        <strong>🔔 Disclaimer:</strong>
        This application is for educational and informational purposes only. 
        It is NOT a substitute for professional medical diagnosis or advice. 
        Always consult qualified healthcare professionals for proper evaluation and treatment.
        </div>
        """, unsafe_allow_html=True)

# ===== USER INPUT SECTION =====
st.sidebar.header('📝 User Input Features')
st.sidebar.markdown("Fill in your health information below:")

def user_input_features():
    """Collect comprehensive user input from sidebar"""
    
    # Demographics Section
    st.sidebar.subheader("👤 Demographics")
    age = st.sidebar.slider('Age (years)', 18, 90, 55, 
                            help='Your current age in years')
    gender = st.sidebar.radio('Gender', ('Female', 'Male'),
                             help='Biological gender')
    
    # Hormonal Status Section
    st.sidebar.subheader("🔬 Hormonal Status")
    hormonal_changes = st.sidebar.radio('Hormonal Changes', 
                                       ('Normal', 'Postmenopausal'),
                                       help='Women: Select if postmenopausal')
    
    # Medical History Section
    st.sidebar.subheader("📋 Medical History")
    family_history = st.sidebar.radio('Family History of Osteoporosis', 
                                     ('No', 'Yes'),
                                     help='Anyone in your family diagnosed with osteoporosis?')
    prior_fractures = st.sidebar.radio('Prior Fractures', 
                                      ('No', 'Yes'),
                                      help='History of bone fractures or breaks?')
    
    medical_conditions = st.sidebar.selectbox('Medical Conditions', 
                                             ('Unknown', 'Hyperthyroidism', 'Rheumatoid Arthritis'),
                                             help='Select if applicable')
    
    medications = st.sidebar.selectbox('Current Medications', 
                                      ('Unknown', 'Corticosteroids'),
                                      help='Long-term corticosteroid use increases osteoporosis risk')
    
    # Lifestyle Section
    st.sidebar.subheader("🏃 Lifestyle Factors")
    body_weight = st.sidebar.radio('Body Weight Status', 
                                  ('Normal', 'Underweight'),
                                  help='Underweight increases fracture risk')
    
    physical_activity = st.sidebar.radio('Physical Activity Level', 
                                        ('Active', 'Sedentary'),
                                        help='Regular weight-bearing exercise protects bones')
    
    # Nutrition Section
    st.sidebar.subheader("🥗 Nutrition")
    calcium_intake = st.sidebar.radio('Calcium Intake', 
                                     ('Adequate', 'Low'),
                                     help='Daily calcium is crucial for bone health')
    
    vitamin_d_intake = st.sidebar.radio('Vitamin D Intake', 
                                       ('Sufficient', 'Insufficient'),
                                       help='Vitamin D helps calcium absorption')
    
    # Habits Section
    st.sidebar.subheader("⛔ Harmful Habits")
    smoking = st.sidebar.radio('Smoking Status', 
                              ('No', 'Yes'),
                              help='Smoking impairs bone formation')
    
    alcohol_consumption = st.sidebar.selectbox('Alcohol Consumption', 
                                              ('Unknown', 'Moderate'),
                                              help='Excessive alcohol affects bone density')

    # Map categorical inputs to numerical values
    data = {
        'Age': age,
        'Hormonal Changes': 1 if hormonal_changes == 'Postmenopausal' else 0,
        'Family History': 1 if family_history == 'Yes' else 0,
        'Body Weight': 1 if body_weight == 'Underweight' else 0,
        'Calcium Intake': 1 if calcium_intake == 'Low' else 0,
        'Vitamin D Intake': 1 if vitamin_d_intake == 'Insufficient' else 0,
        'Physical Activity': 1 if physical_activity == 'Sedentary' else 0,
        'Smoking': 1 if smoking == 'Yes' else 0,
        'Prior Fractures': 1 if prior_fractures == 'Yes' else 0,
        'Gender_Female': 1 if gender == 'Female' else 0,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Medications_Corticosteroids': 1 if medications == 'Corticosteroids' else 0,
        'Medications_Unknown': 1 if medications == 'Unknown' else 0,
        'Medical Conditions_Hyperthyroidism': 1 if medical_conditions == 'Hyperthyroidism' else 0,
        'Medical Conditions_Rheumatoid Arthritis': 1 if medical_conditions == 'Rheumatoid Arthritis' else 0,
        'Medical Conditions_Unknown': 1 if medical_conditions == 'Unknown' else 0,
        'Alcohol Consumption_Moderate': 1 if alcohol_consumption == 'Moderate' else 0,
        'Alcohol Consumption_Unknown': 1 if alcohol_consumption == 'Unknown' else 0
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Reorder features to match model's expected order
input_df = input_df[FEATURE_ORDER]

# ===== DISPLAY USER INPUT =====
with tab1:
    st.subheader('👁️ Review Your Input Information')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Demographics**")
        demo_data = input_df[['Age', 'Gender_Female', 'Gender_Male', 'Hormonal Changes']]
        st.dataframe(demo_data, use_container_width=True)
    
    with col2:
        st.markdown("**Medical History**")
        medical_data = input_df[['Family History', 'Prior Fractures', 'Medications_Corticosteroids']]
        st.dataframe(medical_data, use_container_width=True)
    
    with col3:
        st.markdown("**Lifestyle**")
        lifestyle_data = input_df[['Smoking', 'Physical Activity', 'Calcium Intake', 'Vitamin D Intake']]
        st.dataframe(lifestyle_data, use_container_width=True)
    
    st.divider()
    
    # ===== PREDICTION BUTTON =====
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        predict_button = st.button('🔍 Analyze Risk Level', 
                                   use_container_width=True, 
                                   type="primary",
                                   key="predict_btn")

# ===== PREDICTIONS AND RESULTS =====
if predict_button or 'prediction_made' in st.session_state:
    try:
        # Prepare data for prediction
        prediction_data = input_df.copy()
        
        # Apply scaler if available
        if scaler is not None:
            try:
                prediction_data = scaler.transform(prediction_data)
                prediction_data = pd.DataFrame(prediction_data, columns=FEATURE_ORDER)
                st.sidebar.info("ℹ️ Data scaled using StandardScaler")
            except Exception as e:
                st.sidebar.warning(f"⚠️ Scaling failed: {str(e)}. Using unscaled data.")
        
        # Generate predictions
        prediction = model.predict(prediction_data)
        prediction_proba = model.predict_proba(prediction_data)
        
        st.session_state.prediction_made = True
        
        with tab2:
            st.subheader('📊 Osteoporosis Risk Assessment Results')
            
            # Main Prediction Cards
            col1, col2, col3 = st.columns(3)
            
            with col1:
                risk_status = 'High Risk' if prediction[0] == 1 else 'Low Risk'
                color = '🔴' if prediction[0] == 1 else '🟢'
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{color} {risk_status}</h3>
                    <p>Overall Osteoporosis Risk</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                low_risk_prob = prediction_proba[0][0] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🟢 {low_risk_prob:.1f}%</h3>
                    <p>Low Risk Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                high_risk_prob = prediction_proba[0][1] * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>🔴 {high_risk_prob:.1f}%</h3>
                    <p>High Risk Probability</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.divider()
            
            # Detailed Analysis
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📈 Probability Distribution")
                prob_chart_data = pd.DataFrame({
                    'Risk Level': ['Low Risk', 'High Risk'],
                    'Probability (%)': [low_risk_prob, high_risk_prob]
                })
                st.bar_chart(prob_chart_data.set_index('Risk Level'), use_container_width=True)
            
            with col2:
                st.subheader("📋 Risk Assessment Summary")
                assessment_df = pd.DataFrame({
                    'Metric': ['Low Risk Probability', 'High Risk Probability', 'Risk Level'],
                    'Value': [f'{low_risk_prob:.2f}%', f'{high_risk_prob:.2f}%', risk_status]
                })
                st.dataframe(assessment_df, use_container_width=True, hide_index=True)
            
            st.divider()
            
            # Recommendations based on risk level
            if prediction[0] == 1:
                st.markdown("""
                <div class="risk-high">
                    <h3>⚠️ High Osteoporosis Risk Detected</h3>
                    <p>Based on your health profile, you have a significant likelihood of osteoporosis. 
                    Immediate medical consultation is strongly recommended.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🏥 Recommended Actions:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Medical Interventions:**
                    - 📅 Schedule a DEXA scan (bone density test) immediately
                    - 👨‍⚕️ Consult an endocrinologist or rheumatologist
                    - 🩺 Request comprehensive metabolic panel
                    - 💊 Discuss medication options (bisphosphonates, etc.)
                    """)
                
                with col2:
                    st.markdown("""
                    **Lifestyle Changes:**
                    - 🥛 Increase calcium intake to 1,200 mg/day
                    - ☀️ Ensure adequate vitamin D (600-800 IU/day)
                    - 🏃 Engage in weight-bearing exercises (30 min daily)
                    - 🚫 Quit smoking if applicable
                    - 🍺 Limit alcohol consumption
                    """)
            else:
                st.markdown("""
                <div class="risk-low">
                    <h3>✅ Low Osteoporosis Risk</h3>
                    <p>Based on your health profile, you have a low likelihood of osteoporosis. 
                    Continue with preventive measures to maintain bone health.</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.subheader("💪 Preventive Measures:")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    **Nutrition:**
                    - 🥛 Maintain adequate calcium intake (1,000 mg/day)
                    - ☀️ Get sufficient vitamin D exposure
                    - 🥗 Eat bone-healthy foods (leafy greens, dairy, nuts)
                    - 💧 Stay well-hydrated
                    """)
                
                with col2:
                    st.markdown("""
                    **Lifestyle:**
                    - 🏃 Stay physically active with regular exercise
                    - 🤸 Include resistance and balance training
                    - 🚫 Avoid smoking and excessive alcohol
                    - 🏥 Regular health check-ups and screening
                    """)
            
            st.divider()
            
            # Risk Factor Analysis
            st.subheader("🔍 Your Risk Factors Analysis")
            
            risk_factors = []
            if input_df['Age'][0] > 65:
                risk_factors.append(("Age > 65", "⚠️ High age increases osteoporosis risk"))
            if input_df['Gender_Female'][0] == 1:
                risk_factors.append(("Female Gender", "⚠️ Women have higher osteoporosis risk"))
            if input_df['Hormonal Changes'][0] == 1:
                risk_factors.append(("Postmenopausal", "⚠️ Hormonal changes increase risk"))
            if input_df['Family History'][0] == 1:
                risk_factors.append(("Family History", "⚠️ Genetic predisposition detected"))
            if input_df['Smoking'][0] == 1:
                risk_factors.append(("Smoking", "⚠️ Smoking impairs bone formation"))
            if input_df['Body Weight'][0] == 1:
                risk_factors.append(("Underweight", "⚠️ Low body weight increases risk"))
            if input_df['Calcium Intake'][0] == 1:
                risk_factors.append(("Low Calcium", "⚠️ Insufficient calcium intake"))
            if input_df['Vitamin D Intake'][0] == 1:
                risk_factors.append(("Low Vitamin D", "⚠️ Insufficient vitamin D levels"))
            if input_df['Physical Activity'][0] == 1:
                risk_factors.append(("Sedentary", "⚠️ Lack of weight-bearing exercise"))
            if input_df['Prior Fractures'][0] == 1:
                risk_factors.append(("Prior Fractures", "⚠️ History of fractures"))
            
            if risk_factors:
                for factor, description in risk_factors:
                    st.info(f"**{factor}**: {description}")
            else:
                st.success("✅ No significant risk factors detected in your profile!")
            
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
        st.info("Please ensure all input fields are filled correctly and try again.")

# ===== FOOTER =====
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("""
    <div style='text-align: center; color: #999; font-size: 0.85rem; margin-top: 2rem;'>
        <p>🏥 <strong>Medical Disclaimer:</strong> This application is for educational purposes only. 
        It is not a substitute for professional medical advice, diagnosis, or treatment.
        Always consult with qualified healthcare professionals for medical guidance.</p>
        <p style='color: #ccc;'>© 2025 Osteoporosis Detection AI System | v1.0</p>
    </div>
    """, unsafe_allow_html=True)
