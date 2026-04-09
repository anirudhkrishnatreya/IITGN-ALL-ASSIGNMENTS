# Day 32: Decision Trees & Random Forests

## 🎯 Overview
This project explores the power of tree-based models for classification tasks. It covers both the interpretability of **Decision Trees** and the robust predictive performance of **Random Forests**, applied to synthetic datasets for Loan Approval and Fraud Detection.

## 📋 Task Description
The assignment was divided into two main parts:
1.  **AM Session: Loan Approval Prediction** — Building a model to predict if a loan application should be approved based on financial metrics.
2.  **PM Session: Insurance Fraud Detection** — Identifying potentially fraudulent insurance claims using historical behavior and claim details.

## 🛠️ Solution Implementation

### 1. Loan Approval (AM Assignment)
*   **Data Generation:** Created a synthetic dataset including features like `annual_income`, `credit_score`, `loan_amount`, and `debt_to_income`.
*   **Decision Tree:** Trained a `DecisionTreeClassifier` with `max_depth=4` and visualized the rules using `export_text`.
*   **Random Forest:** Implemented a `RandomForestClassifier` and used `RandomizedSearchCV` to find the best hyperparameters for `n_estimators` and `max_depth`.
*   **Comparison:** Evaluated and compared the accuracy of both models on a test set.

### 2. Fraud Detection (PM Assignment)
*   **Data Generation:** Generated synthetic insurance data with features like `claim_amount`, `num_claims`, and `fraud_history`.
*   **Strategic Metric Selection:** Since missing a fraud case is more costly than a false alarm, we optimized the model for **Recall**.
*   **Decision Tree:** Built a depth-limited tree to understand key fraud indicators.
*   **Hyperparameter Tuning:** Used `RandomizedSearchCV` with a `scoring='recall'` parameter to tune the Random Forest, ensuring high sensitivity to fraudulent claims.

## 💡 Key Takeaways
*   **Interpretability:** Decision trees provide clear, human-readable logic for decisions (e.g., "If credit score > 650 then Approve").
*   **Ensemble Power:** Random Forests significantly improve performance by reducing variance through bagging.
*   **Metric Choice:** Accuracy isn't always the best metric—especially in fraud detection where **Recall** is critical.
