# CSV Batch Upload Feature - Complete Guide

## 📤 Overview

The CSV upload feature allows you to upload a spreadsheet with multiple patient records and get osteoporosis risk predictions for all of them at once. This is perfect for:
- Healthcare facilities analyzing patient cohorts
- Research studies with multiple subjects
- Batch screening programs
- Data analysis and population health studies

---

## 📋 CSV File Requirements

### Format Specifications
- **File Type**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8 (standard)
- **Delimiter**: Comma (,)
- **Header**: Yes, must include column names in first row

### Required Columns

Your CSV must contain exactly these 18 columns (case-sensitive):

1. **Age** (numeric, 18-90)
2. **Hormonal Changes** (binary, 0 or 1)
3. **Family History** (binary, 0 or 1)
4. **Body Weight** (binary, 0 or 1)
5. **Calcium Intake** (binary, 0 or 1)
6. **Vitamin D Intake** (binary, 0 or 1)
7. **Physical Activity** (binary, 0 or 1)
8. **Smoking** (binary, 0 or 1)
9. **Prior Fractures** (binary, 0 or 1)
10. **Gender_Female** (binary, 0 or 1)
11. **Gender_Male** (binary, 0 or 1)
12. **Medications_Corticosteroids** (binary, 0 or 1)
13. **Medications_Unknown** (binary, 0 or 1)
14. **Medical Conditions_Hyperthyroidism** (binary, 0 or 1)
15. **Medical Conditions_Rheumatoid Arthritis** (binary, 0 or 1)
16. **Medical Conditions_Unknown** (binary, 0 or 1)
17. **Alcohol Consumption_Moderate** (binary, 0 or 1)
18. **Alcohol Consumption_Unknown** (binary, 0 or 1)

---

## 🔄 Feature Mapping

### Binary Features (0 = No/Normal, 1 = Yes/Abnormal)

```
Hormonal Changes:
  0 = Normal
  1 = Postmenopausal

Family History:
  0 = No
  1 = Yes

Body Weight:
  0 = Normal
  1 = Underweight

Calcium Intake:
  0 = Adequate
  1 = Low

Vitamin D Intake:
  0 = Sufficient
  1 = Insufficient

Physical Activity:
  0 = Active
  1 = Sedentary

Smoking:
  0 = No
  1 = Yes

Prior Fractures:
  0 = No
  1 = Yes

Gender (One-hot encoded):
  Gender_Female=1, Gender_Male=0 (for females)
  Gender_Female=0, Gender_Male=1 (for males)
  
Medications (One-hot encoded):
  Medications_Corticosteroids=1, Medications_Unknown=0 (if using corticosteroids)
  Medications_Corticosteroids=0, Medications_Unknown=1 (if unknown)
  
Medical Conditions (One-hot encoded):
  Medical Conditions_Hyperthyroidism=1, others=0 (if has hyperthyroidism)
  Medical Conditions_Rheumatoid Arthritis=1, others=0 (if has RA)
  Medical Conditions_Unknown=1, others=0 (if unknown)
  
Alcohol Consumption (One-hot encoded):
  Alcohol Consumption_Moderate=1, Alcohol_Unknown=0 (if moderate consumption)
  Alcohol Consumption_Moderate=0, Alcohol_Unknown=1 (if unknown)
```

---

## 📝 Example CSV Structure

```csv
Age,Hormonal Changes,Family History,Body Weight,Calcium Intake,Vitamin D Intake,Physical Activity,Smoking,Prior Fractures,Gender_Female,Gender_Male,Medications_Corticosteroids,Medications_Unknown,Medical Conditions_Hyperthyroidism,Medical Conditions_Rheumatoid Arthritis,Medical Conditions_Unknown,Alcohol Consumption_Moderate,Alcohol Consumption_Unknown
45,0,0,0,0,0,0,0,0,1,0,0,1,0,0,1,0,1
65,1,1,0,1,1,0,1,1,1,0,1,0,0,1,0,0,1
52,0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1
```

---

## 🎯 How to Create a CSV File

### Using Excel/Google Sheets

1. Open Excel or Google Sheets
2. Create column headers matching the required columns
3. Fill in patient data row by row
4. Export as CSV:
   - **Excel**: File → Save As → CSV (Comma Delimited)
   - **Google Sheets**: File → Download → CSV

### Using Python

```python
import pandas as pd

# Create sample data
data = {
    'Age': [45, 65, 52],
    'Hormonal Changes': [0, 1, 0],
    'Family History': [0, 1, 0],
    'Body Weight': [0, 0, 0],
    'Calcium Intake': [0, 1, 0],
    'Vitamin D Intake': [0, 1, 0],
    'Physical Activity': [0, 0, 0],
    'Smoking': [0, 1, 0],
    'Prior Fractures': [0, 1, 0],
    'Gender_Female': [1, 1, 0],
    'Gender_Male': [0, 0, 1],
    'Medications_Corticosteroids': [0, 1, 0],
    'Medications_Unknown': [1, 0, 1],
    'Medical Conditions_Hyperthyroidism': [0, 0, 0],
    'Medical Conditions_Rheumatoid Arthritis': [0, 1, 0],
    'Medical Conditions_Unknown': [1, 0, 1],
    'Alcohol Consumption_Moderate': [0, 0, 0],
    'Alcohol Consumption_Unknown': [1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv('patient_data.csv', index=False)
```

---

## ✅ Data Validation

Before uploading, ensure:

### Column Names
- ✅ All 18 required columns present
- ✅ Exact spelling and capitalization
- ✅ No extra spaces in headers

### Data Values
- ✅ Age is numeric (18-90)
- ✅ Binary features are 0 or 1 only
- ✅ No missing values (NA, blank, or N/A)
- ✅ No text values in numeric columns

### Row Checks
- ✅ At least one data row (excluding header)
- ✅ All rows have same number of columns
- ✅ No duplicate rows (unless intentional)

---

## 🚀 Upload Process

### Step-by-Step

1. **Navigate to "📤 Batch CSV Upload" tab** in the app
2. **Click "Browse files"** or drag-drop your CSV
3. **Review the data preview** shown in the app
4. **Wait for validation** - app will check all columns
5. **View results** in the batch prediction results section
6. **Download results** using the download button

### What Happens

1. App validates column names
2. App validates data types and ranges
3. Features are scaled using the StandardScaler
4. XGBoost model makes predictions
5. Results are displayed and available for download

---

## 📊 Output Information

### Prediction Results Include

For each patient record:
- **Risk_Level**: High Risk or Low Risk
- **Low_Risk_Probability**: Probability of low risk (0-1)
- **High_Risk_Probability**: Probability of high risk (0-1)
- **Confidence_%**: Maximum probability × 100

### Summary Statistics

- Total Records: Number of patients analyzed
- High Risk: Count and percentage with high risk
- Low Risk: Count and percentage with low risk
- Average Confidence: Mean confidence across all predictions

### Visualizations

- **Risk Distribution Chart**: Bar chart showing High vs Low risk counts
- **Probability Statistics**: Min, Max, Mean, Std Dev of probabilities
- **Risk Stratification Table**: Summary of risk categories

---

## 💾 Exporting Results

### Download Options

1. **CSV Format**: Complete results with all predictions
2. **Filename**: Automatically timestamped
   - Format: `osteoporosis_predictions_YYYYMMDD_HHMMSS.csv`

### Exported Columns

Original data columns + prediction columns:
- All original 18 feature columns
- Risk_Level
- Low_Risk_Probability
- High_Risk_Probability
- Confidence_%

---

## 🔍 Troubleshooting

### Error: "Missing required columns"

**Solution**: 
- Check column names match exactly (case-sensitive)
- Ensure no extra spaces before/after names
- Verify all 18 columns are present

### Error: "Invalid data type"

**Solution**:
- Age must be numeric (not text)
- Binary features must be 0 or 1 (not Yes/No)
- No empty cells allowed

### Error: "Scaling failed"

**Solution**:
- Check for invalid values (infinity, NaN)
- Ensure all numeric fields have valid numbers
- Age should be realistic (18-90)

### Predictions don't make sense

**Solution**:
- Verify input data accuracy
- Check binary features are correctly coded (0/1)
- Ensure one Gender flag is 1, others are 0
- Review Medical Conditions coding

---

## 📈 Use Cases

### Use Case 1: Clinical Screening
```
Upload: 100 patient records from clinic database
Get: Risk assessment for all patients
Action: Identify high-risk patients for DEXA screening
```

### Use Case 2: Research Study
```
Upload: Baseline data from 500 study participants
Get: Batch predictions for all
Export: Results for analysis and publication
```

### Use Case 3: Population Health
```
Upload: Demographic data from health system
Get: Population-level risk assessment
Use: Guide public health intervention programs
```

---

## ⚠️ Important Notes

### Data Privacy
- ⚠️ App does NOT store uploaded files
- ⚠️ Predictions are generated locally
- ⚠️ No data sent to external servers
- ⚠️ Each session is independent

### Clinical Use
- ⚠️ Results are screening tool only
- ⚠️ NOT diagnostic
- ⚠️ Always validate with DEXA scan
- ⚠️ Consult healthcare professionals

### Accuracy
- ⚠️ Depends on data accuracy
- ⚠️ Model trained on historical data
- ⚠️ Individual variation may differ
- ⚠️ Regular model updates recommended

---

## 📝 Sample CSV File

A sample file `sample_predictions.csv` is included with the app.

**To use**:
1. Download `sample_predictions.csv` from project folder
2. Open in the "📤 Batch CSV Upload" tab
3. See how predictions work
4. Modify as needed for your data

---

## 🎓 Best Practices

1. **Validate Input Data**
   - Check for data quality before upload
   - Remove obviously incorrect entries
   - Ensure consistent data entry

2. **Organize Your Data**
   - Use consistent feature encoding
   - Keep original identifiers (patient ID, etc.)
   - Document data collection procedures

3. **Review Results**
   - Check for unexpected patterns
   - Validate with clinical knowledge
   - Flag outliers for further review

4. **Document Usage**
   - Keep upload dates/times
   - Track model versions used
   - Document any data cleaning
   - Archive results for audit trail

---

## 📞 Support

For issues or questions:
1. Check this guide first
2. Review sample CSV file
3. Verify data format requirements
4. Check app documentation
5. Ensure scaler.joblib is present

---

**Last Updated**: December 3, 2025
**Feature**: Batch CSV Upload & Prediction
**Version**: 2.0 (with CSV Support)
