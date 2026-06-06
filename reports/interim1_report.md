# Fraud Detection Project — Interim-1 Report
**Student:** Betel Yohannes  
**GitHub:** https://github.com/Benareyo/fraud-detection  
**Date:** June 7, 2026  
**Program:** 10 Academy — Week 5 & 6 Challenge

---

## 1. Project Overview

This project builds a fraud detection system for **Adey Innovations Inc.**, a FinTech company serving e-commerce and banking clients. The goal is to detect fraudulent transactions across two very different datasets:

- **Fraud_Data.csv** — E-commerce transactions with rich behavioral, device, and user context
- **creditcard.csv** — Bank credit card transactions with PCA-anonymized features

This report covers **Task 1**: data understanding, cleaning, preprocessing, feature engineering, and class imbalance handling.

---

## 2. Dataset Overview

### 2.1 Fraud_Data.csv (E-commerce)
| Property | Value |
|----------|-------|
| Total records | 151,112 |
| Features | 11 (user_id, signup_time, purchase_time, purchase_value, device_id, source, browser, sex, age, ip_address, class) |
| Target column | `class` (1 = fraud, 0 = legitimate) |
| Missing values | None |
| Duplicate rows | None |

### 2.2 IpAddress_to_Country.csv
| Property | Value |
|----------|-------|
| Total records | 138,846 |
| Features | 3 (lower_bound_ip_address, upper_bound_ip_address, country) |
| Purpose | Maps IP ranges to countries for geolocation enrichment |

### 2.3 creditcard.csv (Bank Transactions)
| Property | Value |
|----------|-------|
| Total records | 284,807 |
| Features | 31 (Time, V1-V28, Amount, Class) |
| Target column | `Class` (1 = fraud, 0 = legitimate) |
| Missing values | None |
| Duplicate rows | 1,081 (removed) |

---

## 3. Data Cleaning

### 3.1 Fraud_Data.csv
- **Missing values:** No missing values found across all columns.
- **Duplicates:** No duplicate rows found.
- **Data type corrections:** `signup_time` and `purchase_time` were converted from string to `datetime64` format to enable time-based feature engineering.
- **IP address:** The `ip_address` column was stored as a float (e.g., `732758368.79`) and required conversion to integer format for range-based lookup.

### 3.2 creditcard.csv
- **Missing values:** No missing values found.
- **Duplicates:** 1,081 duplicate rows were identified and removed. Justification: duplicate transactions are likely data entry errors and could bias the model.
- **Data types:** All features (V1-V28, Time, Amount) were already in numeric format. No additional corrections needed.

---

## 4. Exploratory Data Analysis

### 4.1 Class Imbalance

Class imbalance is the most critical characteristic of both datasets. Fraud is a rare event, and models trained without addressing this will be biased toward predicting the majority class.

**Fraud_Data.csv:**
| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate (0) | 136,961 | 90.63% |
| Fraud (1) | 14,151 | 9.37% |

**creditcard.csv:**
| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate (0) | 283,253 | 99.83% |
| Fraud (1) | 492 | 0.17% |

The credit card dataset is extremely imbalanced — only 0.17% of transactions are fraudulent. This makes accuracy a misleading metric; a model that predicts everything as legitimate would achieve 99.83% accuracy while catching zero fraud cases. This is why we use **AUC-PR and F1-Score** as primary evaluation metrics.

### 4.2 Key EDA Findings — Fraud_Data.csv

**Age Distribution:**
- Fraudulent transactions are slightly more concentrated in younger age groups (20-35).
- Legitimate transactions are more evenly spread across age groups.

**Purchase Value:**
- Both fraud and legitimate transactions show similar purchase value distributions.
- This suggests purchase value alone is not a strong fraud signal.

**Source Channel:**
- Fraud rates vary slightly by traffic source (SEO, Ads, Direct).
- No single source dramatically dominates fraud.

**Browser:**
- Chrome and Firefox users show similar fraud rates.
- This feature alone is not highly discriminative.

**Sex:**
- Fraud rate is similar between male and female users.
- Gender is not a strong standalone predictor.

### 4.3 Key EDA Findings — creditcard.csv

**Amount:**
- Fraudulent transactions tend to have lower amounts on average compared to legitimate ones.
- This is counterintuitive but may reflect fraudsters testing cards with small amounts.

**Time:**
- Fraud transactions are more evenly distributed across time.
- Legitimate transactions show two peaks (likely two business days in the dataset).

**V Features (PCA):**
- Several V features (V4, V11, V12, V14, V17) show strong separation between fraud and legitimate classes in the correlation heatmap.

---

## 5. Geolocation Integration

### 5.1 IP Address to Integer Conversion
IP addresses in `Fraud_Data.csv` were stored as floating point numbers representing the numeric form of the IP. We converted them to proper integers using the formula:

```
IP_int = part1 × 16,777,216 + part2 × 65,536 + part3 × 256 + part4
```

### 5.2 Range-Based Merge
The `IpAddress_to_Country.csv` file maps IP integer ranges to countries. We used `pandas.merge_asof` for an efficient range-based lookup — this merges each transaction's IP integer with the closest lower bound in the IP range table, then validates that the IP falls within the upper bound.

Key steps:
- Both dataframes were sorted by IP address before merging
- Data types were aligned to `int64` to avoid merge errors
- IPs that fell outside all ranges were labeled `Unknown`

### 5.3 Fraud Patterns by Country
After merging, we analyzed fraud rates by country (minimum 50 transactions):
- Some countries showed significantly higher fraud rates than others
- This geographic signal adds meaningful predictive power to the model
- Countries with high fraud rates should trigger additional verification steps

---

## 6. Feature Engineering

The following new features were created from `Fraud_Data.csv`:

### 6.1 time_since_signup
```python
df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
```
**Why:** Fraudsters often sign up and immediately make purchases. A very short time between signup and purchase is a strong fraud signal. This turned out to be the most important feature in our model.

### 6.2 hour_of_day
```python
df['hour_of_day'] = df['purchase_time'].dt.hour
```
**Why:** Fraud is more likely at certain times of day, particularly late night hours when monitoring is lower.

### 6.3 day_of_week
```python
df['day_of_week'] = df['purchase_time'].dt.dayofweek
```
**Why:** Fraud patterns may differ on weekdays vs weekends.

### 6.4 user_transaction_count
```python
df['user_transaction_count'] = df.groupby('user_id')['user_id'].transform('count')
```
**Why:** Users with unusually high total transaction counts may indicate account takeover or bot activity.

### 6.5 user_daily_transactions
```python
df['user_daily_transactions'] = df.groupby(['user_id', 'purchase_date'])['user_id'].transform('count')
```
**Why:** High transaction velocity within a single day is a strong fraud signal.

---

## 7. Data Transformation

### 7.1 Encoding Categorical Features
One-hot encoding was applied to: `source`, `browser`, `sex`, `country`.  
`drop_first=True` was used to avoid multicollinearity.

### 7.2 Scaling Numerical Features
`StandardScaler` was applied to numerical columns:
`purchase_value`, `age`, `time_since_signup`, `hour_of_day`, `day_of_week`, `user_transaction_count`, `user_daily_transactions`

**Important:** The scaler was **fit only on the training set** and then applied to the test set. This prevents data leakage.

---

## 8. Handling Class Imbalance

### 8.1 Strategy: SMOTE (Synthetic Minority Over-sampling Technique)

We chose **SMOTE** over undersampling for the following reasons:
- Undersampling discards legitimate transaction data — we lose valuable information
- SMOTE generates synthetic fraud samples by interpolating between existing fraud examples
- This preserves all legitimate transaction data while balancing the classes

### 8.2 Results

**Fraud_Data.csv — Training Set:**
| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| Legitimate (0) | 109,568 | 109,568 |
| Fraud (1) | 11,321 | 109,568 |

**creditcard.csv — Training Set:**
| Class | Before SMOTE | After SMOTE |
|-------|-------------|-------------|
| Legitimate (0) | 226,601 | 226,601 |
| Fraud (1) | 394 | 226,601 |

### 8.3 Critical Note
SMOTE was applied **only to the training set**. The test set was kept with its original imbalanced distribution to simulate real-world evaluation conditions. Applying SMOTE to the test set would produce artificially optimistic metrics.

---

## 9. Processed Data Outputs

All processed files were saved to `data/processed/`:

| File | Description |
|------|-------------|
| fraud_data_with_country.csv | Fraud data merged with country |
| X_train_fraud.csv | Training features (after SMOTE) |
| y_train_fraud.csv | Training labels (after SMOTE) |
| X_test_fraud.csv | Test features |
| y_test_fraud.csv | Test labels |
| X_train_creditcard.csv | Credit card training features |
| y_train_creditcard.csv | Credit card training labels |
| X_test_creditcard.csv | Credit card test features |
| y_test_creditcard.csv | Credit card test labels |
| scaler_fraud.pkl | Fitted StandardScaler for fraud data |
| scaler_creditcard.pkl | Fitted StandardScaler for credit card |

---

## 10. Summary

| Step | Fraud_Data | creditcard |
|------|-----------|------------|
| Missing values | None | None |
| Duplicates removed | 0 | 1,081 |
| Geolocation merge | Yes (IP to country) | No |
| Features engineered | 5 new features | None needed |
| Encoding | One-hot (4 columns) | Not needed |
| Scaling | StandardScaler | StandardScaler |
| Imbalance handling | SMOTE | SMOTE |
| Train/test split | 80/20 stratified | 80/20 stratified |

---

## 11. GitHub Repository

**Repository:** https://github.com/Benareyo/fraud-detection  
**Branch structure:**
- `main` — stable, merged code
- `task-1-data-preprocessing` — Task 1 work (merged)
- `task-2-modeling` — Task 2 work (merged)
- `task-3-shap-explainability` — Task 3 work (merged)

**CI/CD:** GitHub Actions runs unit tests automatically on every push. All checks passing ✅

