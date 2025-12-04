# Osteoporosis Detection AI System 🏥

A machine learning-powered web application for predicting osteoporosis risk using XGBoost and Streamlit.

## 🎯 Features

- **Single Patient Predictions**: Real-time risk assessment with probability scores and confidence levels
- **Batch CSV Upload**: Process multiple patient records simultaneously with automatic data cleaning
- **Data Preprocessing**: Automatic scaling and normalization using scikit-learn StandardScaler
- **Professional UI**: Clean Streamlit interface with 4 tabs:
  - Tab 1: Input Information (single patient)
  - Tab 2: Results & Analysis
  - Tab 3: Batch CSV Upload
  - Tab 4: About & Guidelines
- **Export Results**: Download predictions as timestamped CSV files
- **Robust Error Handling**: Handles dirty data, missing values, and non-numeric inputs

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/osteoporosis-detection.git
cd osteoporosis-detection
```

2. Create a virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
streamlit run app.py
```

The app will be available at `http://localhost:8502`

## 📊 How to Use

### Single Patient Prediction
1. Fill in the patient health information in the sidebar
2. Click "Predict Risk Level"
3. View results with:
   - Risk classification (High/Low)
   - Probability scores
   - Confidence percentage
   - Risk factors identification

### Batch CSV Upload
1. Go to the "Batch CSV Upload" tab
2. Upload a CSV file with patient data
3. The app will automatically:
   - Clean and validate data
   - Handle missing features
   - Convert non-numeric values
   - Generate predictions
4. Download results as CSV

## 📁 Project Structure

```
osteoporosis-detection/
├── app.py                              # Main Streamlit application
├── best_osteoporosis_model.joblib     # Trained XGBoost model (18 features)
├── scaler.joblib                      # StandardScaler for feature normalization
├── requirements.txt                    # Python dependencies
├── sample_predictions.csv             # Sample data for testing
├── README.md                          # This file
├── START_HERE.txt                     # Quick start guide
└── documentation/
    ├── CSV_UPLOAD_GUIDE.md
    ├── DEPLOYMENT_GUIDE.md
    └── MODEL_INFO.md
```

## 🧠 Model Details

- **Algorithm**: XGBoost (Gradient Boosting)
- **Estimators**: 100
- **Task**: Binary Classification
- **Features**: 18 health indicators
- **Output**: Risk Level (High/Low) with probability scores

### Model Features (18 total)

**Demographics & Hormonal:**
- Age
- Hormonal Changes (Postmenopausal status)
- Gender_Female, Gender_Male

**Medical History:**
- Family History
- Prior Fractures
- Medical Conditions (Hyperthyroidism, Rheumatoid Arthritis)
- Medications (Corticosteroids)

**Lifestyle & Nutrition:**
- Body Weight Status
- Physical Activity Level
- Calcium Intake
- Vitamin D Intake
- Smoking Status
- Alcohol Consumption

## 🔄 Data Preprocessing

All input data is automatically preprocessed using StandardScaler:
- Features are normalized to have mean=0 and std=1
- Ensures consistent model predictions
- Handles missing columns by filling with 0
- Converts non-numeric values to 0

## 📋 CSV Format Requirements (Flexible)

The app accepts CSV files with:
- ✅ Any subset of the 18 features (missing features are auto-filled)
- ✅ Case-sensitive column names matching the feature list
- ✅ Numeric values (or will be converted to 0)
- ✅ Any number of patient records

### Example CSV Structure:
```
Age,Family History,Physical Activity,Smoking,...
45,0,0,1,...
55,1,1,0,...
```

## ⚠️ Medical Disclaimer

This application is **for educational and informational purposes only**. It is **NOT a substitute** for professional medical advice, diagnosis, or treatment. 

**Always consult with qualified healthcare professionals for proper evaluation and medical guidance.**

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Python web framework)
- **ML Model**: XGBoost (Gradient Boosting)
- **Data Processing**: Pandas, NumPy
- **Preprocessing**: scikit-learn (StandardScaler)
- **Model Serialization**: Joblib

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:
```
streamlit
xgboost
scikit-learn
pandas
numpy
joblib
```

## 🔐 Security & Privacy

- No data is stored on the server
- All processing happens in-memory
- Results are downloaded locally
- No external API calls for predictions

## 🐛 Troubleshooting

### Issue: Model not found
- Ensure `best_osteoporosis_model.joblib` is in the same directory as `app.py`

### Issue: Scaler not found
- Ensure `scaler.joblib` is in the same directory as `app.py`

### Issue: CSV upload fails
- Check that CSV is valid and readable
- Ensure columns are properly named (case-sensitive)
- Try with sample_predictions.csv first

### Issue: Streamlit not starting
- Verify Python version: `python --version` (should be 3.10+)
- Reinstall dependencies: `pip install -r requirements.txt --force-reinstall`

## 📈 Future Enhancements

- [ ] Multi-model ensemble predictions
- [ ] Feature importance visualization
- [ ] Risk trend analysis over time
- [ ] Mobile-friendly interface
- [ ] API endpoint for integrations
- [ ] Advanced analytics dashboard

## 📄 License

MIT License - See LICENSE file for details

## 👤 Author

Your Name

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For issues and questions, please open an issue on GitHub.

---

**Note**: This project was developed as an educational demonstration of machine learning and Streamlit web applications.
