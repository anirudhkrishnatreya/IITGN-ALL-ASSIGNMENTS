# Day 40: NLP Evaluation, Model Selection & Production Readiness

## 🎯 Overview
This assignment addresses a real-world production failure scenario at **ShopSense**: a classifier reporting **94% accuracy** but predicting "positive" for nearly *every* review. The notebook (`week07_friday_evaluation.ipynb`) builds a rigorous evaluation framework covering class imbalance, multi-metric assessment, hard production constraints, cost modelling, and a formal technical brief — culminating in diagnosing and fixing the broken pipeline.

## 📋 Scenario
> Priya Menon, Head of Product, needs a defensible production recommendation before greenlighting the ShopSense review intelligence feature. The dataset is imbalanced: **Positive=65%, Neutral=20%, Negative=15%**.

## 🛠️ Solution Implementation

### Sub-step 1 (Easy) — Class Distribution & Why Accuracy Fails
- **`analyse_class_distribution(df)`**: Computes class frequencies and percentages.
- **Key Insight:** With 65% positive class, a model predicting *all positive* achieves 65% accuracy — or even 94% on a skewed test split. Accuracy is a **misleading** metric for imbalanced data.
- **Correct metric:** Macro F1 — averages F1 across all classes equally, penalising models that ignore the minority class.

### Sub-step 2 (Easy) — Evaluate Classifier with Appropriate Metrics
- **Baseline Model:** TF-IDF + Logistic Regression (`class_weight='balanced'`).
- **Metrics reported:** Accuracy, Macro F1, Per-class Precision/Recall/F1 via `classification_report`.
- **Why `class_weight='balanced'`?** Automatically adjusts loss weights inversely proportional to class frequency — no oversampling needed for sparse TF-IDF vectors.

### Sub-step 3 (Medium) — Two Approaches vs. 3 Hard Production Constraints
Two models compared: **TF-IDF + Logistic Regression** vs **BOW + Multinomial Naive Bayes**

| Constraint | Description | Winner |
|---|---|---|
| **New categories** | F1 drop on held-out product categories (zero-shot generalisation) | TF-IDF + LR |
| **Code-mixed reviews** | Hindi-English transliteration pipeline then re-evaluate | TF-IDF + LR |
| **Inference speed** | Must be < 20ms per review (200-run average latency) | Both pass |

### Sub-step 4 (Medium) — Cost Model & Production Recommendation
- **`compute_daily_cost(y_test, y_pred)`**: Translates errors into real business cost.
  - **False Negative (missed negative review):** Higher cost — unhappy customer churns silently.
  - **False Positive (flagged positive as negative):** Lower cost — unnecessary moderation effort.
- Daily cost = `FN_count × cost_FN + FP_count × cost_FP`, scaled to `DAILY_VOLUME` reviews.
- **Recommendation:** TF-IDF + LR wins on both macro F1 and daily cost.

### Sub-step 5 (Medium) — One-Page Technical Brief for Priya Menon
- Structured brief covering:
  - Why the 94% accuracy figure was misleading
  - Correct evaluation methodology (Macro F1)
  - Model comparison summary
  - Cost impact analysis
  - Production recommendation with monitoring guidance (PSI for distribution drift)

### Sub-step 6 (Hard) — Reproduce & Fix the 94% Accuracy Failure
- **`build_broken_pipeline(df)`**: Reproduces the failure — no `class_weight`, accuracy-only evaluation, majority-class bias.
- **Root Cause:** Without class balancing, LR minimises loss by predicting the dominant class (`positive`) for ambiguous reviews. Accuracy stays high because 65% of test labels are already positive.
- **Fix:** Add `class_weight='balanced'` + evaluate on Macro F1 → model correctly learns minority class boundaries.

### Sub-step 7 (Hard) — Cost of the Production Failure
- Runs the **cost model on the broken pipeline** vs the fixed one.
- **Result:** The broken pipeline's high FN rate (missing most negative reviews) translates to significantly higher daily cost compared to the balanced, fixed pipeline — making the business case for the fix concrete and defensible.

## 💡 Key Takeaways
- **Accuracy is not enough:** On imbalanced data, it can be actively misleading — always report Macro F1 and per-class metrics.
- **`class_weight='balanced'` > SMOTE** for sparse TF-IDF vectors: simpler, no synthetic data, equivalent effect.
- **Production constraints matter:** A model can be statistically better but fail latency or generalisation requirements.
- **Cost models make the case:** Translating FN/FP counts into monetary impact is far more persuasive to stakeholders than abstract F1 scores.
- **Monitor distribution shift:** Track Population Stability Index (PSI) between training and live data to catch future silent failures early.
