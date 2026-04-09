# Day 30: Logistic Regression — SUV Purchase Dataset

## 🎯 Overview
This project implements an end-to-end Machine Learning pipeline to predict whether a user will purchase an SUV based on their age and estimated salary. This is a classic binary classification problem solved using **Logistic Regression**.

## 📋 Task Description
The primary goal was to take a raw dataset (`suv_data.csv`) and move through all the standard stages of a supervised learning project:
1.  **Exploratory Data Analysis (EDA):** Understanding the distribution and types of features.
2.  **Data Preprocessing:** Handling categorical variables and selecting relevant features.
3.  **Model Building:** Developing a predictive model using `sklearn`.
4.  **Performance Evaluation:** Assessing the model's reliability using standard metrics.

## 🛠️ Solution Implementation

### 1. Data Loading & Exploration
*   Loaded the dataset using `pandas`.
*   Inspected the shape and column types.
*   Checked for missing values (none found) to ensure data quality.

### 2. Preprocessing & Feature Engineering
*   **Categorical Encoding:** Mapped 'Gender' to binary values (Male: 0, Female: 1).
*   **Feature Selection:** Focused on `Age` and `EstimatedSalary` as the primary predictors for the target `Purchased`.

### 3. Pipeline Construction
*   **Train-Test Split:** Partitioned the data into training (80%) and testing (20%) sets.
*   **Feature Scaling:** Applied `StandardScaler` to normalize the input features, which is crucial for Logistic Regression performance.

### 4. Model Training & Evaluation
*   Initialized and trained a `LogisticRegression` model.
*   **Accuracy:** Achieved a high classification accuracy on the test set.
*   **Confusion Matrix:** Visualized True Positives, True Negatives, False Positives, and False Negatives to confirm the model's precision and recall.

### 5. Robustness Testing
*   Experimented with different test sizes (30%, 25%) to verify the consistency of the model's performance across different data splits.

## 💡 Key Takeaways
*   Normalization of features is essential for distance-based or gradient-based algorithms like Logistic Regression.
*   Evaluation metrics beyond simple accuracy (like the Confusion Matrix) provide a deeper understanding of classification errors.
