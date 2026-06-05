# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project builds fraud detection models for two transaction types at Adey Innovations Inc.:
- **E-commerce transactions** (Fraud_Data.csv) - rich behavioral and device features
- **Bank credit card transactions** (creditcard.csv) - PCA-anonymized features

## Project Structure

    fraud-detection/
    ├── .github/workflows/    # CI/CD pipeline
    ├── data/
    │   ├── raw/              # Original datasets (not tracked by git)
    │   └── processed/        # Cleaned, engineered, resampled data
    ├── notebooks/
    │   ├── eda-fraud-data.ipynb        # EDA for e-commerce data
    │   ├── eda-creditcard.ipynb        # EDA + preprocessing for credit card
    │   ├── feature-engineering.ipynb   # Feature engineering + SMOTE
    │   ├── modeling.ipynb              # Model training and evaluation
    │   └── shap-explainability.ipynb   # SHAP analysis and recommendations
    ├── models/               # Saved model artifacts
    ├── tests/                # Unit tests
    ├── requirements.txt
    └── README.md

## Setup Instructions

### 1. Clone the repository

    git clone https://github.com/Benareyo/fraud-detection.git
    cd fraud-detection

### 2. Create and activate virtual environment

    python -m venv venv
    source venv/Scripts/activate

### 3. Install dependencies

    pip install -r requirements.txt

### 4. Add datasets to data/raw/
- Fraud_Data.csv
- IpAddress_to_Country.csv
- creditcard.csv

### 5. Run notebooks in order
1. notebooks/eda-fraud-data.ipynb
2. notebooks/eda-creditcard.ipynb
3. notebooks/feature-engineering.ipynb
4. notebooks/modeling.ipynb
5. notebooks/shap-explainability.ipynb

## Key Tasks

### Task 1 - Data Analysis and Preprocessing
- Cleaned and merged Fraud_Data.csv with IP-to-country mapping
- Engineered features: time_since_signup, hour_of_day, day_of_week, transaction velocity
- Handled class imbalance using SMOTE on training set only
- Class ratio before SMOTE: ~9% fraud / 91% legitimate

### Task 2 - Model Building
- Baseline: Logistic Regression
- Ensemble: XGBoost with hyperparameter tuning
- Evaluation: AUC-PR, F1-Score, Confusion Matrix
- Cross-validation: Stratified K-Fold (k=5)
- XGBoost outperformed Logistic Regression on both datasets

### Task 3 - SHAP Explainability
- Global feature importance via SHAP summary plots
- Force plots for True Positive, False Positive, False Negative
- Top fraud drivers: time_since_signup, user_transaction_count, hour_of_day

## Business Recommendations
1. Transactions within 24 hours of signup require extra verification
2. Accounts with more than 3 transactions per hour should be auto-flagged
3. Late night transactions (midnight-4am) require step-up authentication
4. Purchases exceeding 3x user average should be flagged for review
5. Apply risk scoring based on country geolocation

## Tech Stack
- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- XGBoost, SHAP
- GitHub Actions CI/CD
