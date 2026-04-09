# Day 37: TF-IDF and BM25 from Scratch

## 🎯 Overview
This project focuses on the fundamentals of Information Retrieval and Natural Language Processing. Instead of relying on high-level libraries, we implement the core logic of **TF-IDF (Term Frequency-Inverse Document Frequency)** and **BM25** from scratch to deeply understand how search and retrieval systems work.

## 📋 Task Description
Using a synthetic "ShopSense" E-Commerce reviews dataset (10,000 entries), the goals were:
1.  **Manual Implementation:** Build a complete TF-IDF vectorization engine without using `sklearn`.
2.  **Retrieval Engine:** Develop a search system using Cosine Similarity to find relevant reviews for a query.
3.  **Benchmarking:** Validate the scratch implementation against `sklearn.feature_extraction.text.TfidfVectorizer`.
4.  **Advanced Ranking:** Implement the BM25 algorithm and compare its results with traditional TF-IDF.

## 🛠️ Solution Implementation

### 1. The TF-IDF Pipeline (From Scratch)
*   **Preprocessing:** Implemented text cleaning (lowercasing, punctuation removal, tokenization) using `re`.
*   **Vocabulary Building:** Created a global vocabulary with `min_df` filtering to remove rare tokens.
*   **IDF Calculation:** Manually computed smoothed IDF using the formula: `log((1+N)/(1+df)) + 1`.
*   **Matrix Construction:** Built a sparse L2-normalized TF-IDF matrix.

### 2. Search & Retrieval
*   **Vectorization:** Implemented a function to convert raw queries into the same TF-IDF space as the corpus.
*   **Cosine Similarity:** Used dot products between query vectors and the document matrix to rank reviews.
*   **Evaluation:** Extracted the top-5 most relevant reviews for a sample query ("wireless earbuds battery life poor").

### 3. Validation & Comparison
*   **Metric:** Calculated the **Average L2 Difference** between the scratch matrix and the `sklearn` matrix.
*   **Result:** Achieved a near-zero difference, proving the mathematical correctness of the custom implementation.
*   **Category Analysis:** Automated the discovery of domain-specific terms (e.g., Identifying "fabric" as a high-scoring term in 'Clothing').

### 4. BM25 Implementation (Bonus)
*   Implemented **BM25** with parameters `k1=1.5` and `b=0.75`.
*   Analyzed how BM25 improves search quality by applying TF saturation and length normalization, preventing long documents from unfairly dominating search results.

## 💡 Key Takeaways
*   **Mathematics of Search:** Understanding the linear algebra behind document ranking.
*   **Sparse Computing:** Handling large text datasets efficiently using sparse matrices.
*   **Algorithm Evolution:** Seeing how BM25 addresses the limitations of standard TF-IDF for real-world search engines.
