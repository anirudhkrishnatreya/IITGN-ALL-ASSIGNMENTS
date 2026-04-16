# Day 45: Deep Learning + Data Cleaning

## 🎯 Overview
This assignment combines practical data cleaning with a from-scratch deep learning implementation. The notebook (`imeteo_assignment.ipynb`) begins by auditing and cleaning a hospital records dataset, then demonstrates a 3-layer neural network built purely with NumPy, followed by accuracy evaluation and a simple cost-based decision analysis.

## 📋 Task Description
The assignment is organised into five sub-steps:

1. **Data Audit** — Load the dataset, inspect missing values, data types, and inconsistencies.
2. **Data Cleaning** — Handle null values, fix data types where needed, and remove duplicate rows.
3. **Neural Network (NumPy)** — Build a 3-layer neural network with manual forward propagation and backpropagation.
4. **Evaluation** — Convert predictions into class labels and measure classification accuracy.
5. **Cost-Based Decision** — Assign different costs to false negatives and false positives to estimate total decision cost.

## 🛠️ Solution Implementation

### 1. Data Audit
- Loads `hospital_records.csv` using pandas.
- Inspects dataset structure with `head()`, `info()`, and `describe()`.
- Checks column-wise missing values to identify data quality issues before modeling.

### 2. Data Cleaning
- Removes duplicate records using `drop_duplicates()`.
- Fills missing numerical values with the column median.
- Fills missing categorical values with the column mode.
- Prints the cleaned dataset shape to confirm the result of preprocessing.

### 3. Neural Network from Scratch
- Implements the sigmoid activation function and its derivative using NumPy.
- Initializes weights for a 3-layer feedforward neural network.
- Runs manual forward propagation through hidden layers and the output layer.
- Performs backpropagation and gradient-based weight updates over multiple epochs.
- Tracks loss during training to monitor whether the network is learning.

### 4. Evaluation
- Converts final output probabilities into binary predictions using a `0.5` threshold.
- Measures model performance with `accuracy_score` from scikit-learn.

### 5. Cost-Based Decision
- Defines separate penalties for:
  - **False Negatives (FN):** missing a patient who should be flagged
  - **False Positives (FP):** raising an unnecessary alert
- Computes the overall cost using the prediction outcomes, illustrating why business or healthcare costs can matter more than accuracy alone.

## 💡 Key Takeaways
- **Data cleaning is foundational:** Even a simple model pipeline becomes unreliable if duplicates and missing values are ignored.
- **Neural networks can be understood mathematically:** Building one from scratch with NumPy makes forward and backward propagation much easier to grasp.
- **Accuracy is not always enough:** In healthcare-style problems, missing a true positive can be far more expensive than a false alarm.
- **Cost-sensitive thinking improves decisions:** Translating prediction errors into real-world cost helps align ML systems with domain priorities.

## ▶️ Notes
- The notebook expects `hospital_records.csv` to be available in the same folder when running the data audit and cleaning steps.
- The neural network section currently demonstrates the training workflow using generated sample data, making it useful for understanding the mechanics of backpropagation.
