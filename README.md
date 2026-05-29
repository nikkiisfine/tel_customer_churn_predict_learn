# Telecom Customer Churn Prediction

Exploratory data analysis and multi-model classification on 7,032 telecom customer records to predict whether a customer will churn.

## Workflow

| Step | Description |
|---|---|
| Data Import | Load dataset, inspect shape, types, and summary statistics |
| Data Cleaning | Handle missing values, type conversion, drop low-relevance features |
| Visualization | Churn rate distribution, feature vs. churn bar charts, correlation heatmap |
| Preprocessing | Standardization, value replacement, LabelEncoder encoding |
| Modeling | Compare 10 classifiers using Recall / Precision / F1 |
| Prediction | Apply best model to predict churn for new customers |

## Dataset

**Telco Customer Churn** — originally published by IBM Watson Analytics as a sample dataset for customer retention analysis. The data tracks a fictional California-based telecom company's customers (7,043 records, 19 features) and whether they left within the last month.

- Rows: 7,043 (7,032 after cleaning)
- Features: 21 columns covering demographics, subscribed services, and billing
- Target: `Churn` (Yes/No → 1/0); positive-to-negative ratio ≈ 1:2.77 (imbalanced)

| Feature Group | Examples |
|---|---|
| Demographics | gender, SeniorCitizen, Partner, Dependents |
| Services | InternetService, OnlineSecurity, StreamingTV, Contract |
| Billing | MonthlyCharges, TotalCharges, PaymentMethod |

**Sources:**
- IBM (original): https://www.ibm.com/docs/en/cognos-analytics/12.0.x?topic=samples-telco-customer-churn
- Kaggle mirror: https://www.kaggle.com/datasets/blastchar/telco-customer-churn (file: `WA_Fn-UseC_-Telco-Customer-Churn.csv`)

## Key Findings

- **Senior citizens** and young users who are unmarried or financially dependent show higher churn rates
- **Contract length** is the strongest retention lever: month-to-month > one year > two year churn rate
- **Electronic check** payers churn more than other payment methods
- Add-on internet services (online security, device protection, tech support) significantly reduce churn
- **Gender** and **phone service** have near-zero correlation with churn and were removed as features

## Model Comparison

Data split via `StratifiedShuffleSplit` (20% train / 80% test):

| Model | Recall | Precision | F1 Score |
|---|---|---|---|
| **Naive Bayes** | **0.7311** | **0.5387** | **0.6203** |
| AdaBoost | 0.5438 | 0.6307 | 0.5841 |
| CatBoost | 0.5251 | 0.6315 | 0.5734 |
| Gradient Boosting | 0.5177 | 0.6376 | 0.5714 |
| Logistic Regression | 0.5224 | 0.6283 | 0.5705 |
| SVM | 0.4903 | 0.6407 | 0.5555 |
| Random Forest | 0.4816 | 0.6332 | 0.5471 |
| KNN | 0.5231 | 0.5622 | 0.5419 |
| XGBoost | 0.4890 | 0.5610 | 0.5225 |
| Decision Tree | 0.5070 | 0.4732 | 0.4895 |

**Naive Bayes achieves the best F1 score (62.03%) and is selected as the final model.**

> Because the dataset is class-imbalanced (only 26.5% churned), Recall is the more business-critical metric. Naive Bayes also leads on Recall (0.731), making it the most effective model for catching at-risk customers.

## Preprocessing Notes

- `TotalCharges` converted to float; 11 rows with invalid values dropped
- `No internet service` and `No phone service` unified to `No`
- Numeric features (`tenure`, `MonthlyCharges`, `TotalCharges`) scaled with `StandardScaler`
- Categorical features encoded with `LabelEncoder`
- Dropped features with negligible churn correlation: `customerID`, `gender`, `PhoneService`

## Setup

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost catboost
```

## Usage

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from either source above
2. Place the CSV in the same directory as the notebook
3. Open `电信用户流失模型分析.ipynb` and run all cells in order
