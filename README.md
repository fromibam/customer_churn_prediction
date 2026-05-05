# customer_churn_prediction
Predicting customer churn using XGBoost Classification — 75% Accuracy | 84% Recall | ROC-AUC: 0.8456

## Overview
A machine learning classification project that predicts whether a telecom 
customer will churn (cancel their subscription) based on their service 
usage and contract data. This project simulates a real-world use case for 
companies like Telkom Indonesia to proactively retain at-risk customers.

## Problem Statement
Customer churn is a major revenue loss challenge for telecom companies — 
once a customer leaves, that monthly revenue is gone permanently. This 
project builds an XGBoost classification model trained on customer 
subscription data, using SMOTE to handle class imbalance, enabling Telkom 
to identify at-risk customers before they leave and take proactive 
retention action.

## Dataset
- Source: [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Total records: 7,043 customers
- Target variable: Churn (Yes = churned, No = stayed)
- Class distribution: 73.5% Stayed, 26.5% Churned

## Tools & Technologies
- Python
- Google Colab
- Pandas, NumPy, Scikit-learn, XGBoost, Imbalanced-learn, 
  Matplotlib, Seaborn

## Methodology
1. Exploratory Data Analysis (EDA)
   - Target variable distribution analysis
   - Identified class imbalance (73.5% vs 26.5%)
2. Data Preprocessing
   - Fixed hidden missing values in TotalCharges (stored as object, 
     11 empty rows converted and filled with median)
   - Label encoding for categorical features
   - Dropped CustomerID (non-predictive identifier)
   - Train/test split (80/20)
3. Handling Class Imbalance
   - Applied SMOTE to training data only
   - Balanced both classes to 4,138 samples each
4. Feature Scaling — StandardScaler
5. Model Implementation — XGBoost Classifier
6. Hyperparameter Tuning
   - Increased estimators to 200
   - Reduced learning rate to 0.05
   - Adjusted max_depth, subsample, colsample_bytree
   - Applied scale_pos_weight=2 to improve churn recall
7. Model Evaluation — Classification Report, Confusion Matrix, 
   ROC Curve, Feature Importance

## Results

| Metric | Initial Model | Tuned Model |
|--------|--------------|-------------|
| Accuracy | 79% | 75% |
| Recall (Churn) | 0.68 | 0.84 |
| Precision (Churn) | 0.58 | 0.52 |
| F1-Score (Churn) | 0.63 | 0.64 |
| ROC-AUC | 0.8448 | 0.8456 |

## Key Findings — Feature Importance
Top features influencing customer churn:
1. **Contract** (0.40) — strongest predictor by far; 
   month-to-month customers churn significantly more
2. **OnlineSecurity** — customers without security add-ons 
   are more likely to leave
3. **Dependents** — customers with dependents are less 
   likely to churn
4. **TechSupport** — customers without tech support 
   show higher churn rates
5. **InternetService** — type of internet service 
   affects churn likelihood

## Business Insight
- **Contract type is the #1 churn driver** — Telkom's most effective 
  retention strategy is offering customers longer contracts early
- The tuned model reduced missed churners from 116 to 60 — 
  significantly improving revenue protection
- Customers without OnlineSecurity and TechSupport add-ons are 
  high-risk segments worth targeting with bundle offers

## How to Run
1. Download the dataset from the Kaggle link above
2. Open `Customer_Churn_Telkom.ipynb` in Google Colab
3. Update the file path to match where you saved the dataset
4. Run all cells
