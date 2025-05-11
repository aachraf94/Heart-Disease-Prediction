# Heart Disease Prediction

## Overview

This project is a machine learning application that predicts the risk of heart disease based on patient health information. The application uses a logistic regression model trained on clinical data to provide personalized risk assessments and recommendations.

## Features

- Interactive web interface built with Streamlit
- Real-time prediction of heart disease risk
- Detailed probability assessment and risk level categorization
- Personalized recommendations based on prediction results
- Visualization of key health metrics and their impact on heart disease risk

## Dataset

The model is trained on a heart disease dataset with 303 patient records including various health metrics such as:

- Demographic information (age, sex)
- Clinical measurements (blood pressure, cholesterol)
- ECG results (resting ECG, maximum heart rate)
- Exercise-related information (exercise-induced angina, ST depression)
- Chest pain characteristics

The target variable "AHD" indicates the presence (Yes) or absence (No) of heart disease.

### Dataset Features

| Feature   | Description                                                        |
| --------- | ------------------------------------------------------------------ |
| Age       | Age in years                                                       |
| Sex       | Gender (1 = male; 0 = female)                                      |
| ChestPain | Type of chest pain (typical, nontypical, nonanginal, asymptomatic) |
| RestBP    | Resting blood pressure in mm Hg                                    |
| Chol      | Serum cholesterol in mg/dl                                         |
| Fbs       | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)              |
| RestECG   | Resting electrocardiographic results (0, 1, 2)                     |
| MaxHR     | Maximum heart rate achieved                                        |
| ExAng     | Exercise induced angina (1 = yes; 0 = no)                          |
| Oldpeak   | ST depression induced by exercise relative to rest                 |
| Slope     | Slope of the peak exercise ST segment                              |
| Ca        | Number of major vessels colored by fluoroscopy (0-3)               |
| Thal      | Thalassemia (normal, fixed defect, reversible defect)              |
| AHD       | Presence of heart disease (Yes, No) - Target variable              |

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. Clone this repository:

   ```
   git clone https://github.com/aachraf94/Heart-Disease-Prediction.git
   ```

2. Install required dependencies:

   ```
   pip install -r requirements.txt
   ```

3. Make sure you have the trained model and associated files in the `saved_models` directory:
   - `logistic_regression_model.joblib`
   - `scaler.joblib`
   - `preprocessing_info.joblib`

## Usage

1. Run the Streamlit application:

   ```
   streamlit run heart_disease_app.py
   ```

2. Access the application in your web browser (typically at http://localhost:8501)

3. Enter patient information in the form:

   - Age
   - Sex
   - Chest pain type
   - Resting blood pressure
   - Serum cholesterol
   - Fasting blood sugar
   - Resting ECG results
   - Maximum heart rate
   - Exercise-induced angina
   - ST depression
   - Slope of peak exercise ST segment
   - Number of major vessels
   - Thalassemia

4. Click "Predict" to get the heart disease risk assessment

## Model Details

- Algorithm: Logistic Regression
- Features: The model uses both raw features and derived features (like BP per Age and HR per Age)
- Preprocessing: Includes categorical encoding, feature scaling, and age grouping
- Performance: The model achieves approximately 85% accuracy on the test dataset
- Evaluation: The model performance metrics are available in the Jupyter notebook used for training

### Model Evaluation Metrics

- Accuracy: ~85%
- Precision: ~84%
- Recall: ~86%
- F1 Score: ~85%
- AUC-ROC: ~0.90

## Project Structure

```
Heart Disease Prediction /
├── heart_disease_app.py      # Streamlit application
├── Heart.csv                 # Dataset with heart disease information
├── saved_models/             # Directory containing trained models
│   ├── logistic_regression_model.joblib  # Trained logistic regression model
│   ├── scaler.joblib         # Feature scaler
│   └── preprocessing_info.joblib         # Preprocessing information
├── requirements.txt          # Package dependencies
└── README.md                 # Project documentation
```

## Technologies Used

- Python
- Pandas for data manipulation
- Scikit-learn for machine learning
- Streamlit for the web interface
- Joblib for model serialization

## How It Works

1. **Data Preprocessing**: The raw patient data is processed to handle categorical variables, create derived features, and scale numerical values.
2. **Feature Engineering**: The application creates age groups and calculates ratios like BP per Age to improve predictive power.
3. **Model Prediction**: The preprocessed data is fed into the trained logistic regression model.
4. **Risk Assessment**: Based on the prediction probability, a risk level (Low, Medium, High) is assigned.
5. **Recommendations**: The system provides tailored medical recommendations based on the prediction.

## Future Improvements

- Integration of additional machine learning models
- More detailed visualizations of feature importance
- Enhanced user interface with patient history tracking
- Support for uploading patient data in batch

## Author

- **Abdelkebir Achraf**


## Acknowledgments

- UCI Machine Learning Repository for the Heart Disease dataset
- The Streamlit team for the awesome web framework
