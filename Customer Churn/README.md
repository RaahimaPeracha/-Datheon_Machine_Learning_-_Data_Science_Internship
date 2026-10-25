# Customer Churn Prediction (Random Forest & XGBoost) + Streamlit Dashboard

## What is this project?
This project predicts which telecom customers are likely to **churn** (leave the company). It trains two models, Random Forest and XGBoost, and then shows the results in a cute pastel **Streamlit dashboard**.

## Dataset
- File: `Telco-Customer-Churn.csv`
- 7,043 customers, 21 columns (contract type, tenure, services, monthly charges, payment method, etc.)
- Target: `Churn` (Yes / No)
- Only about **26.5%** of customers churn, so the data is **imbalanced**.

## What the notebook does (`churn_classifier.ipynb`)
1. **Load and clean the data.** `TotalCharges` was stored as text. The 11 blank values belong to brand-new customers (tenure 0), so they are filled with 0.
2. **Explore** the class imbalance, churn by contract type, and tenure vs churn.
3. **Split** the data 80% train and 20% test (stratified).
4. **Preprocess** with `OneHotEncoder` (text columns) and `StandardScaler` (numbers).
5. **Balance the classes with SMOTE.** It is placed inside the pipeline, so it only touches training data and the test scores stay honest.
6. **Cross-validate** (5-fold) and train Random Forest and XGBoost.
7. **Evaluate** with ROC-AUC, accuracy, precision, recall and F1, plus the ROC curve and confusion matrices.
8. **Plot the top 10 feature importances** for both models.
9. **Save the trained models** as `.joblib` files for the dashboard.

## Results
| Model | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| Random Forest | 0.841 | 0.836 | 0.764 | 0.546 | 0.663 | 0.599 |
| XGBoost | 0.843 | 0.839 | 0.781 | 0.582 | 0.618 | 0.599 |

- Both models pass the **ROC-AUC > 0.80** goal. They are almost tied.
- Random Forest catches more churners; XGBoost raises fewer false alarms.
- **Top churn drivers:** month-to-month contract, short tenure, no online security, no tech support, fiber optic internet, and electronic check payment. Two-year contracts keep customers.

## The dashboard (`app.py`)
A soft pastel Streamlit app with four tabs:
- 🌸 **Overview:** key numbers and charts (churn by contract, tenure, payment method), with a contract filter.
- 🎀 **Model Performance:** ROC curves, confusion matrix and metrics, with a decision-threshold slider.
- 🍓 **Feature Importance:** the most important features for the chosen model.
- 🔮 **Predict a Customer:** enter one customer's details and get a churn-risk gauge plus friendly retention tips.

You can switch between Random Forest and XGBoost in the sidebar.

> The models were trained on SMOTE-balanced data, so the dashboard score is best read as a **relative risk**, not an exact probability.

## Tools used
Python, Pandas, Scikit-Learn, imbalanced-learn (SMOTE), XGBoost, Seaborn, Matplotlib, Plotly, Streamlit, Joblib.

## How to run
1. Put all files in one folder (see below).
2. Install the libraries:
   ```
   pip install pandas numpy scikit-learn imbalanced-learn xgboost seaborn matplotlib plotly streamlit joblib
   ```
3. Open `churn_classifier.ipynb` in VS Code, select a Python kernel and click **Run All**. This also creates the two `.joblib` model files.
4. Start the dashboard from the terminal:
   ```
   streamlit run app.py
   ```

## Files
```
Telco-Customer-Churn.csv     the dataset
churn_classifier.ipynb       notebook: code, outputs and plots
rf_churn_model.joblib        saved Random Forest (created by the notebook)
xgb_churn_model.joblib       saved XGBoost (created by the notebook)
app.py                       the Streamlit dashboard
.streamlit/config.toml       pastel theme for the dashboard
README.md                    this file
```