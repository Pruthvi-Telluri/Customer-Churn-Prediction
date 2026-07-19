# Customer Churn Prediction

A machine learning project that predicts whether a telecom customer is likely to churn, using account details, billing information, and service usage from the IBM Telco Customer Churn dataset (7,043 customers).

## Problem Statement

Customer retention is far cheaper than acquisition. This project identifies customers at risk of churning and highlights *why*, so a business can proactively intervene.

## Tech Stack

- Python, pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (Logistic Regression, Random Forest)
- Jupyter Notebook

## Workflow

1. Data cleaning (handled hidden blank values in `TotalCharges` for new customers)
2. Exploratory Data Analysis
3. Feature encoding and scaling
4. Model training and comparison (Logistic Regression vs Random Forest)
5. Class imbalance handling (`class_weight="balanced"`)
6. Hyperparameter tuning (GridSearchCV)
7. Feature engineering experiments
8. Final evaluation

## Results

| Metric | Score |
|---|---|
| Accuracy | 73.9% |
| Precision | 50.5% |
| Recall | 79.7% |
| F1 Score | 0.618 |
| ROC-AUC | 0.839 |

**Model chosen:** Logistic Regression with balanced class weights — prioritizes catching actual churners (79.7% recall) over raw accuracy, since missing a churning customer is costlier to a business than a false alarm.

### Key insights from EDA:
- Month-to-month contract customers churn far more than annual/two-year contracts
- New customers (low tenure) are highest risk
- Higher monthly charges correlate with higher churn

## How to Run

\`\`\`bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/customer_churn_prediction.ipynb
\`\`\`

## Author

Pruthviraj Sundar