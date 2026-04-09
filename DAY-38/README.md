# Day 38: Word2Vec & Semantic Similarity

## 🎯 Overview
This project explores the evolution of word embeddings and the transition from surface-level lexical matching to deep semantic understanding. We implement and analyze **Word2Vec (Skip-Gram)** models and compare them against modern transformer-based architectures like **Sentence-BERT (SBERT)**.

## 📋 Task Description
The assignment focused on the "Semantic Gap"—the discrepancy between what words say and what they mean. Key objectives included:
1.  **Skip-Gram Training:** Building Word2Vec models with varying window sizes to capture syntactic vs. semantic relationships.
2.  **Polysemy Analysis:** Demonstrating the "One Vector Problem" where a static embedding averages conflicting meanings of the same word.
3.  **Disambiguation:** Building a sense-detection system using context windows and anchor vectors.
4.  **Multi-Method Benchmarking:** Evaluating Similarity across BOW, TF-IDF, Word2Vec, and SBERT.

## 🛠️ Solution Implementation

### 1. Word2Vec Dynamics
*   **Window Size Impact:** Trained two models:
    *   `window=2`: Captured **Syntactic** relationships (e.g., POS patterns, nearby antonyms like *fast/slow*).
    *   `window=10`: Captured **Semantic** relationships (e.g., thematic coherence like *camera/photos*, *battery/charging*).
*   **Polysemy Demo:** Showed that the word "**cheap**" resulted in a vector that was semi-similar to both "affordable" and "flimsy," effectively landing in a "no-man's land" between the two poles.

### 2. Disambiguation System
*   **Methodology:** Developed a custom sense detector that:
    1.  Extracts the context vector around a target word.
    2.  Compares it to "Anchor Prototypes" (average vectors of sense-specific words like *budget/economical* vs *shoddy/fragile*).
    3.  Successfully predicts the intended sense of polysemous words based on neighboring tokens.

### 3. Closing the Semantic Gap
*   **The Benchmarking Test:** Compared two reviews expressing the same sentiment but using different words:
    *   *Review A:* "incredible camera but terrible battery life"
    *   *Review B:* "Battery drains fast, although photos are stunning"
*   **Results:**
    *   **BOW/TF-IDF:** Failed (~0.12 similarity) because there was zero word overlap beyond "battery".
    *   **Word2Vec avg:** Partially succeeded (~0.52) by capturing synonym relationships (camera $\approx$ photos).
    *   **Sentence-BERT:** Superior performance (~0.76), correctly identifying the high semantic equivalence through attention-based context.

## 💡 Key Takeaways
*   **Context is King:** Standard Word2Vec is limited by its static nature; context averaging or Transformers are required to handle complex meanings.
*   **The Semantic Gap:** Lexical methods (BOW/TF-IDF) are insufficient for modern search or sentiment tasks where paraphrasing is common.
*   **Vector Geometry:** Large windows help models "behave" semantically, while small windows help them "behave" grammatically.
