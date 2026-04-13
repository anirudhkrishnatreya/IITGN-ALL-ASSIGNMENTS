# Day 39: Hard NLP Patterns & Aspect-Level Sentiment Analysis

## 🎯 Overview
This assignment tackles the most challenging real-world NLP phenomena encountered in Indian e-commerce reviews. Using the **ShopSense E-Commerce Reviews (10K)** dataset, we build rule-based and feature-engineering pipelines to handle linguistic patterns that break naive sentiment classifiers, culminating in a full **Aspect-Based Sentiment Analysis (ABSA)** system.

## 📋 Task Description
The notebook (`week07_wednesday_nlp_patterns.ipynb`) addresses five hard NLP patterns and a full ABSA module:

1. **Negation** – "not bad at all" → Positive
2. **Sarcasm** – "Wow great! Broke on day 1" → Negative
3. **Code-Mixing** – Hindi-English transliteration ("bahut accha" → "very good")
4. **Implicit Sentiment** – "Returned it within 2 hours" → Negative (no opinion words)
5. **Comparative Sentiment** – "Way better than my previous Samsung"
6. **Aspect-Based Sentiment Analysis (ABSA)** – Extracting per-aspect (camera, battery, etc.) sentiment pairs

## 🛠️ Solution Implementation

### Q1(a) — Negation Detection
- **`detect_negation(text)`**: Tags every token within a configurable window after negation cues (`not`, `never`, `hardly`, etc.) with a `NEG_` prefix.
- **`negation_aware_features(text)`**: Converts raw text into negation-aware unigram + bigram features.
- **Baseline Failure:** BOW sees `"bad"` → predicts NEGATIVE. ✗
- **Fixed:** `"NEG_bad"` is learned as a distinct positive feature → predicts POSITIVE. ✓

### Q1(b) — Sarcasm Detection
- **`detect_sarcasm_signals(text)`**: Multi-signal rule-based detector using:
  - Hyperbolic opener (wow, great, amazing in first 5 tokens)
  - Positive + Negative co-occurrence
  - Excessive punctuation (`!!!`, `???`)
  - Clause contrast (positive first clause, negative second clause)
- Score = fraction of signals triggered; threshold ≥ 0.5 → SARCASTIC.
- **Key Insight:** Surface polarity is *inverted* in sarcasm — positive words carry negative intent.

### Q1(c) — Code-Mixed (Hindi-English) Reviews
- **`transliterate_hindi_tokens(tokens)`**: Maps Roman-script Hindi tokens (`bahut`, `accha`, `bekar`, `kharab`) to English equivalents via a domain-specific sentiment lexicon.
- **`preprocess_code_mixed(text)`**: Full pipeline — lowercase → transliterate → negation-aware featurisation.
- **Baseline Failure:** English-only model drops all Hindi tokens, misses sentiment entirely.
- **Fixed:** After transliteration, "bahut accha" → "very good" is correctly recognised.

### Q1(d) — Implicit Sentiment
- **`detect_implicit_sentiment(text)`**: Detects behavioural signals of dissatisfaction without opinion words:
  - Return/refund signals (`returned`, `sent back`, `refund`)
  - Negative actions (`broke`, `stopped working`, `damaged`)
  - Time-based failure (`within X hours/days`)
- Returns confidence score and implicit sentiment label.

### Q1(e) — Comparative Sentiment
- Detects comparative markers (`better than`, `worse than`, `superior to`, `way better`, etc.)
- Extracts **subject entity** (what is being praised/criticised) and **reference entity** (what it is compared against).
- Assigns directional sentiment: positive toward subject vs. negative toward reference.

### Q2 — Aspect-Based Sentiment Analysis (ABSA)
- **Q2(a) — Why harder than review-level?**
  - Review-level F1 = 88%: one label per review.
  - Aspect-level F1 = 71%: one label *per aspect* — a single review can contain conflicting sentiments ("great camera, terrible battery").
  - Challenges: implicit aspects, aspect-opinion misalignment, aspect boundary detection.

- **Q2(b) — How to improve from 71% → 80%+?**
  - Span-based aspect extraction instead of keyword matching.
  - Dependency parsing to link opinion words to aspect spans.
  - Fine-tuned BERT/RoBERTa with aspect-specific attention.

- **Q2(c) — Aspect-Sentiment Pair Extraction**
  - `ASPECT_LEXICON`: maps synonyms to canonical aspect names (`camera`, `battery`, `delivery`, `price`, `build`).
  - Pipeline: tokenise → detect aspects → score sentiment in local context window → return `(aspect, sentiment, confidence)` triples.

## 💡 Key Takeaways
- **One label per review is not enough:** Real reviews contain mixed, conflicting, and aspect-specific sentiments.
- **Surface polarity ≠ Intended polarity:** Negation and sarcasm invert the apparent sentiment.
- **Multilingual reality:** Indian e-commerce NLP must handle code-mixing by default, not as an edge case.
- **Implicit signals matter:** Behavioural facts ("returned it") carry strong sentiment without a single opinion word.
- **Context windows are critical:** Aspect sentiment should be scored locally around the aspect mention, not globally across the review.
