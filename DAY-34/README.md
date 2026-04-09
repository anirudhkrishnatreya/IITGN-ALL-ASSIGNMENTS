# Day 34: PCA, Clustering & Model Comparison

## 🎯 Overview
This project dives into Unsupervised Learning techniques, specifically **Feature Dimensionality Reduction** (PCA) and **Clustering** (KMeans, DBSCAN). It also features an automated framework for comparing multiple classification models across standard datasets.

## 📋 Task Description
The assignment covered several key areas of Machine Learning:
1.  **Clustering:** Segmenting the Iris dataset and evaluating cluster quality.
2.  **PCA for Compression:** Using Principal Component Analysis to compress images.
3.  **Benchmarking:** Creating a reusable pipeline to compare performance across different supervised learning algorithms.

## 🛠️ Solution Implementation

### 1. Clustering on Iris Dataset
*   **Algorithms:** Implemented both **KMeans** (partition-based) and **DBSCAN** (density-based).
*   **Evaluation:** Used sophisticated metrics like **Adjusted Rand Index (ARI)** and **Normalized Mutual Info (NMI)** to measure how well the clusters matched the true labels.
*   **Visualization:** Analyzed the results through Confusion Matrices to understand where labels were mixed.

### 2. PCA Image Compression
*   Applied **PCA** to the classic "china.jpg" dataset from `sklearn`.
*   **Dimension Reduction:** Retained the top 50 principal components to represent the image.
*   **Reconstruction:** Successfully reconstructed the image from the compressed data, demonstrating the tradeoff between storage and visual fidelity.

### 3. Weekly Model Comparison
*   Developed a `weekly_model_comparison` function.
*   **Algorithms Compared:** Logistic Regression (LR), Random Forest (RF), Support Vector Machine (SVM), and K-Nearest Neighbors (KNN).
*   **Methodology:** Used **5-fold Cross-Validation** on the Wine dataset to ensure robust and unbiased performance scores.

### 4. Theoretical Analysis
*   Analyzed why accuracy might drop after PCA, noting that PCA maximizes variance but may discard low-variance features that are critical for prediction.

## 💡 Key Takeaways
*   **PCA** is a powerful tool for noise reduction and compression but must be used carefully to avoid losing predictive information.
*   **Clustering** performance is highly dependent on the algorithm and the underlying density of the data.
*   **Automated Benchmarking** is essential for selecting the best model for a specific dataset.
