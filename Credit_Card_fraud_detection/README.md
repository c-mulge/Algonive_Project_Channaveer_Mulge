# Credit Card Fraud Detection Using Machine Learning

**Project Report**

## 1. Introduction

Credit card fraud is one of the most common financial crimes in digital transactions. Detecting fraudulent transactions early helps financial institutions reduce losses and protect users. This project develops a machine learning–based classification system to identify fraudulent transactions using anonymized real-world transaction data.

The model analyzes transaction behavior patterns and assigns a fraud probability score to each transaction.

---

## 2. Objective 🎯

The main objectives of this project are:

* Detect fraudulent transactions using machine learning techniques
* Analyze transaction patterns through exploratory data analysis
* Handle class imbalance in fraud datasets
* Train classification models for fraud prediction
* Evaluate model performance using appropriate metrics
* Generate fraud probability scores for decision-making

---

## 3. Dataset Description 📊

The dataset contains anonymized credit card transactions made by European cardholders.

Dataset characteristics:

* Total transactions: 284,807
* Fraudulent transactions: 492
* Features:

  * **Time**: seconds elapsed between transactions
  * **Amount**: transaction value
  * **V1–V28**: PCA-transformed anonymized features
  * **Class**:

    * 0 → Normal transaction
    * 1 → Fraudulent transaction

The dataset is highly imbalanced, with fraud cases representing only about **0.17%** of total transactions.

---

## 4. Exploratory Data Analysis (EDA) 🔍

Exploratory analysis was performed to understand dataset structure and transaction patterns.

Key observations:

* No missing values were present in the dataset
* Fraud transactions formed a very small percentage of total data
* Transaction amount distribution showed significant outliers
* PCA-transformed variables captured hidden behavioral patterns
* Strong imbalance between normal and fraudulent transactions required correction before model training

Visualization techniques used:

* Class distribution plots
* Transaction amount histograms
* Fraud vs amount comparison
* Feature correlation heatmap

---

## 5. Data Preprocessing 🧹

Data preprocessing steps included:

### Feature Scaling

Since PCA features were already normalized, only the following variables were scaled:

* Time
* Amount

StandardScaler was applied to normalize these features.

### Feature Selection

Input variables:

* V1–V28
* Scaled Time
* Scaled Amount

Target variable:

* Class

---

## 6. Handling Imbalanced Data ⚖️

The dataset contained very few fraud cases compared to normal transactions. To address this issue:

**SMOTE (Synthetic Minority Oversampling Technique)** was applied to the training dataset.

Benefits of SMOTE:

* Generates synthetic fraud samples
* Improves model learning
* Reduces prediction bias toward majority class
* Enhances recall performance

---

## 7. Model Development 🤖

Two classification models were implemented:

### Logistic Regression

Used as a baseline classifier to establish reference performance.

Advantages:

* Fast training
* Works well with scaled features
* Provides interpretable results

### Decision Tree Classifier

Used to capture nonlinear relationships between features.

Advantages:

* Handles complex fraud behavior patterns
* Automatically ranks feature importance
* Improces fraud detection capability

---

## 8. Model Evaluation 📈

Model performance was evaluated using:

* Precision
* Recall
* F1-score
* Confusion Matrix
* ROC-AUC Score

Special emphasis was placed on **recall**, as detecting fraudulent transactions is more important than minimizing false alarms.

Confusion matrix analysis helped identify:

* Correct fraud detections
* Missed fraud cases
* False positive alerts

---

## 9. Feature Importance Analysis 📊

Decision Tree feature importance scores identified the most influential predictors of fraud.

Top contributing features included:

* V14
* V12
* V10
* V17
* V4

These PCA-derived variables represent hidden transaction behavior anomalies strongly associated with fraudulent activity.

---

## 10. Fraud Probability Prediction 🔢

Instead of only classifying transactions as fraud or non-fraud, the model assigns a probability score between **0 and 1**.

Example:

* Probability near 0 → normal transaction
* Probability near 1 → high fraud risk

A threshold-based decision rule was implemented:

Transactions with probability ≥ 0.30 are classified as fraudulent.

This allows flexible risk-based fraud detection similar to real banking systems.

---

## 11. Key Results ✅

From this project:

* Dataset successfully analyzed and preprocessed
* Class imbalance handled using SMOTE
* Logistic Regression baseline model trained
* Decision Tree classifier implemented
* Fraud probability scoring introduced
* Important predictive features identified
* Model evaluation performed using multiple performance metrics

The system can now detect suspicious transactions and estimate fraud likelihood effectively.

---

## 12. Conclusion 🚀

This project demonstrates how machine learning techniques can be applied to detect fraudulent financial transactions using anonymized behavioral data.

By combining preprocessing, imbalance handling, classification modeling, and probability-based decision thresholds, the developed system provides a practical framework for fraud detection.

The approach can be extended further using advanced ensemble methods, hyperparameter tuning, and deployment pipelines for real-time fraud monitoring applications.
