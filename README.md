# Fraud Detection for E-commerce and Bank Transactions

## Project Overview
This project builds fraud detection models for two transaction types:
- E-commerce transactions (Fraud_Data.csv)
- Bank credit card transactions (creditcard.csv)

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/Benareyo/fraud-detection.git
cd fraud-detection
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add datasets
Place these files in `data/raw/`:
- Fraud_Data.csv
- IpAddress_to_Country.csv
- creditcard.csv

### 5. Run notebooks in order
1. notebooks/eda-fraud-data.ipynb
2. notebooks/eda-creditcard.ipynb
3. notebooks/feature-engineering.ipynb
4. notebooks/modeling.ipynb
5. notebooks/shap-explainability.ipynb

## Project Structure
fraud-detection/
├── data/raw/          # Raw datasets (not tracked by git)
├── data/processed/    # Cleaned datasets
├── notebooks/         # Jupyter notebooks
├── src/               # Source modules
├── tests/             # Unit tests
├── models/            # Saved model files
└── scripts/           # Utility scripts
