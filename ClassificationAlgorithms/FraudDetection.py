import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Example dataset
data = {
    "Amount": [100, 200, 50000, 150, 300000, 250],
    "Fraud":  [0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["Amount"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))

# 💳 Fraud Detection using Machine Learning

# Used in:

# Credit card transactions

# UPI payments

# Online banking

# Insurance claims

# Big companies like Visa, Mastercard, PayPal use advanced ML for this.

# 🟢 1️⃣ VERY EASY EXPLANATION

# Imagine:

# A person normally spends ₹500–₹2000 in Rajkot.

# Suddenly:

# ₹2,00,000 transaction from another country.

# System thinks:

# ⚠ This looks unusual → Maybe Fraud.

# Fraud detection = Detect unusual patterns.

# 🟡 2️⃣ WHAT TYPE OF PROBLEM IS THIS?

# It is:

# 👉 Binary Classification

# 0 → Genuine
# 1 → Fraud

# But here is the big challenge:

# Fraud cases are very rare.

# Example:

# 100,000 transactions
# Only 200 fraud

# That means:

# Highly Imbalanced Dataset ⚠

# 🔵 3️⃣ COMPLETE ML PIPELINE

# Step 1 → Collect transaction data

# Features may include:

# Transaction amount

# Time

# Location

# Merchant type

# Device ID

# Previous behavior

# Step 2 → Data Cleaning

# Step 3 → Handle Imbalanced Data

# Step 4 → Train Model

# Step 5 → Evaluate carefully

# 🟣 4️⃣ BIG CHALLENGE: IMBALANCED DATA

# If 99% are genuine:

# Model can predict:

# Always 0

# Accuracy = 99%

# But it is useless ❌

# So we use:

# ✔ Precision
# ✔ Recall
# ✔ F1-score
# ✔ ROC-AUC

# NOT just accuracy.

# 🔴 5️⃣ IMPORTANT METRICS

# For Fraud Detection:

# Recall is VERY IMPORTANT.

# Why?

# If fraud is missed → Money loss.

# Recall formula:
# Recall = TP / (TP + FN)

# High recall = Detect most fraud cases.

# 🟤 6️⃣ BEST ALGORITHMS FOR FRAUD DETECTION

# ✔ Logistic Regression
# ✔ Random Forest
# ✔ Gradient Boosting
# ✔ XGBoost
# ✔ Neural Networks

# Random Forest & XGBoost are very popular.

# 8️⃣ HOW TO HANDLE IMBALANCE?

# Very important 🔥

# Methods:

# 1️⃣ Oversampling (SMOTE)
# 2️⃣ Undersampling
# 3️⃣ Class weights

# Example in sklearn:

# RandomForestClassifier(class_weight="balanced")

# 9️⃣ REAL-WORLD FRAUD FEATURES

# In real systems:

# ✔ Time between transactions
# ✔ IP address change
# ✔ Device fingerprint
# ✔ Amount deviation
# ✔ Location mismatch

# Model learns abnormal behavior patterns.

# 🧠 10️⃣ INTERVIEW QUESTIONS

# Q1: Why is accuracy not good metric in fraud detection?
# Because data is imbalanced.

# Q2: Which metric is more important?
# Recall (to catch fraud).

# Q3: What is imbalanced dataset?
# One class is much larger than other.

# Q4: How to handle imbalance?
# SMOTE, class weights, resampling.

# Q5: Is fraud detection supervised or unsupervised?
# Mostly supervised, but anomaly detection can be unsupervised.

# 🟣 11️⃣ ADVANCED CONCEPT

# Sometimes fraud is unknown pattern.

# Then we use:

# ✔ Isolation Forest
# ✔ One-Class SVM
# ✔ Autoencoders

# This is called:

# Anomaly Detection.

# 🟢 12️⃣ HOW YOU CAN USE THIS IN PROJECT

# Since you're building MERN + ML projects:

# You can build:

# 💳 Transaction Fraud Detection API

# Frontend → React
# Backend → Node + Express
# ML Model → Python Flask
# Database → MongoDB

# This will look very strong on resume 🔥

# 🔥 13️⃣ SUMMARY

# Fraud Detection:

# ✔ Binary classification
# ✔ Highly imbalanced
# ✔ Recall very important
# ✔ RandomForest / XGBoost common
# ✔ Real-world impact is huge