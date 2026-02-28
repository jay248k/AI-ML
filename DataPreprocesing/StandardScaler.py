

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [35, 40, 50, 55, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)

# 1️⃣ VERY EASY EXPLANATION

# Imagine you have 2 features:

# Hours studied → 1 to 10

# Salary → 10,000 to 1,00,000

# Problem ❌
# Salary numbers are very big.
# Hours numbers are very small.

# Machine learning models get confused because:

# Big numbers dominate small numbers.

# So we make all numbers balanced.

# StandardScaler does this:

# It makes:

# Mean = 0

# Standard Deviation = 1

# So data looks like:

# -1.2
# 0.5
# 0.8
# -0.3

# Now everything is on same scale ✅

# 🔎 What happens internally?

# Suppose Hours column:

# 1, 2, 3, 4, 5

# Mean = 3
# Standard deviation ≈ 1.41

# Formula used:

# 𝑧
# =
# (
# 𝑥
# −
# 𝑚
# 𝑒
# 𝑎
# 𝑛
# )
# /
# 𝑠
# 𝑡
# 𝑑
# z=(x−mean)/std

# For value 1:

# (
# 1
# −
# 3
# )
# /
# 1.41
# ≈
# −
# 1.41
# (1−3)/1.41≈−1.41

# For value 5:

# (
# 5
# −
# 3
# )
# /
# 1.41
# ≈
# 1.41
# (5−3)/1.41≈1.41

# So data becomes centered around 0.

# 🔵 3️⃣ WHY WE USE StandardScaler?

# Because some ML algorithms depend on distance.

# For example:

# KNN

# SVM

# Logistic Regression

# KMeans

# They calculate distance using formula like:

# Distance=√((x1−x2)2+(y1−y2)2)

# If one feature is huge and another is small:

# Large feature dominates distance.

# So scaling is necessary.

# 4️⃣ MATHEMATICAL EXPLANATION (Advanced)

# StandardScaler transforms feature X as:

# X_scaled = (X - μ) / σ

# Where:

# μ = mean

# σ = standard deviation

# After scaling:

# Mean becomes 0
# Variance becomes 1

# This helps in:

# Faster gradient descent convergence

# Better optimization

# Improved numerical stability

# 🟣 5️⃣ INTERVIEW QUESTIONS

# Q1: When should we use StandardScaler?
# Answer: When algorithm is distance-based or gradient-based.

# Q2: Difference between StandardScaler and MinMaxScaler?

# StandardScaler:

# Mean = 0

# Std = 1

# Can have negative values

# MinMaxScaler:

# Range between 0 and 1

# No negative values

# Q3: Should we scale before or after train_test_split?

# Correct answer:
# After splitting.

# Why?

# Because we must not leak test data information into training.

# Correct way:

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Notice:

# fit only on training data

# transform on test data

# 🟤 6️⃣ REAL-WORLD CONNECTION (For You)

# In resume ranking:

# Features:

# Experience (0–20 years)

# Skills score (0–100)

# Salary expectation (2L–20L)

# Different ranges.

# Without scaling:

# Salary dominates model.

# With StandardScaler:

# All features treated equally.

# Model becomes fair.

# 🟢 7️⃣ WHEN NOT TO USE StandardScaler?

# ❌ Tree-based models:

# DecisionTree

# RandomForest

# XGBoost

# Trees do not depend on distance.

# So scaling not required there.

# 🧠 VERY IMPORTANT CONCEPT

# StandardScaler is used mostly in:

# Logistic Regression

# SVM

# KNN

# Neural Networks

# PCA

# KMeans

# Not required for:

# Decision Trees

# Random Forest