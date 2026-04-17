# Day 46: CNNs + Text Classification

## 🎯 Overview
This assignment introduces two practical deep learning workflows inside `wednesday_assignment.ipynb`: image classification with a Convolutional Neural Network (CNN) using the MNIST handwritten digits dataset, and text classification on social media posts using a TF-IDF representation with Logistic Regression. Together, these sections show how different data modalities require different preprocessing and modeling choices.

## 📋 Task Description
The notebook is organised into four main stages:

1. **Social Media Dataset Inspection** — Load the text dataset and inspect the label distribution for `hate_speech`.
2. **Image Data Preparation** — Load MNIST, normalize pixel values, and reshape images for CNN input.
3. **CNN Architecture Setup** — Define and compile a convolutional neural network for digit classification.
4. **Text Classification Pipeline** — Convert post text into TF-IDF features and train a Logistic Regression classifier.

## 🛠️ Solution Implementation

### 1. Social Media Dataset Inspection
- Loads `social_media_posts.csv` with pandas.
- Displays sample rows using `head()` to inspect the dataset structure.
- Checks the class balance with `value_counts()` on the `hate_speech` column.

### 2. MNIST Preprocessing
- Imports the MNIST dataset from `tensorflow.keras.datasets`.
- Scales image pixel values from `0-255` to `0-1` for stable training.
- Reshapes images to `(28, 28, 1)` so they can be processed by convolutional layers.

### 3. CNN Model Definition
- Builds a sequential CNN using TensorFlow/Keras.
- Uses two convolution stages:
  - `Conv2D(32, (3,3))` followed by max pooling
  - `Conv2D(64, (3,3))` for deeper feature extraction
- Flattens learned image features and feeds them through dense layers.
- Produces 10-class predictions with a `softmax` output layer.
- Compiles the model with the Adam optimizer and sparse categorical cross-entropy loss.

### 4. Text Classification Pipeline
- Vectorizes post text using `TfidfVectorizer(max_features=5000)`.
- Converts raw text into a sparse numerical representation.
- Trains a `LogisticRegression` classifier on the transformed features to predict `hate_speech`.

## 💡 Key Takeaways
- **CNNs are well-suited for images:** Convolution and pooling layers help the model learn local spatial patterns such as strokes and shapes in handwritten digits.
- **Preprocessing depends on data type:** Image tensors and text features need very different preparation before modeling.
- **Simple baselines still matter:** TF-IDF with Logistic Regression is a strong baseline for many text classification tasks.
- **Model design should match the problem:** Image classification benefits from deep feature extraction, while classical ML methods can still perform effectively on vectorized text data.

## ▶️ Notes
- The notebook expects `social_media_posts.csv` to be available in the same folder for the text classification section.
- The current notebook defines and compiles the CNN but does not yet show full model training or evaluation metrics.
- Despite the notebook heading mentioning “Embeddings,” the implemented text pipeline currently uses TF-IDF features rather than learned word embeddings.
