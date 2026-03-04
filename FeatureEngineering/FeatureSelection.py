# 1️⃣ What is Feature Selection?

# Feature Selection means:

# 👉 Selecting only important input features
# 👉 Removing unnecessary features

# Example:

# If predicting salary:

# Features:

# Age ✅

# Experience ✅

# Education ✅

# Favorite color ❌

# Mobile wallpaper ❌

# We remove useless features.

# 🟡 2️⃣ Why Feature Selection is Important?

# ✔ Improves model performance
# ✔ Reduces overfitting
# ✔ Faster training
# ✔ Better interpretability
# ✔ Reduces noise

# Very common interview topic 🔥

# 🔵 3️⃣ Types of Feature Selection

# There are 3 main types:

# 1️⃣ Filter Methods
# 2️⃣ Wrapper Methods
# 3️⃣ Embedded Methods
# 🟣 4️⃣ 1️⃣ Filter Methods (Simple & Fast)

# Based on statistical tests.

# Examples:

# ✔ Correlation
# ✔ Chi-Square Test
# ✔ ANOVA
# ✔ Mutual Information

# Example (Correlation):

# import pandas as pd

# corr = df.corr()
# print(corr["Target"])

# Remove features with low correlation.

# 🟤 5️⃣ 2️⃣ Wrapper Methods (Powerful but Slow)

# Use model performance to select features.

# Example:

# ✔ Forward Selection
# ✔ Backward Elimination
# ✔ Recursive Feature Elimination (RFE)

# Example:

# from sklearn.feature_selection import RFE
# from sklearn.linear_model import LogisticRegression

# model = LogisticRegression()
# rfe = RFE(model, n_features_to_select=2)
# X_selected = rfe.fit_transform(X, y)
# 🔴 6️⃣ 3️⃣ Embedded Methods (Best Practical)

# Feature selection happens during model training.

# Examples:

# ✔ Lasso Regression (L1 penalty)
# ✔ Ridge Regression (L2 penalty)
# ✔ Random Forest (feature importance)

# Example:

# from sklearn.ensemble import RandomForestClassifier

# model = RandomForestClassifier()
# model.fit(X, y)

# print(model.feature_importances_)
# 🟠 7️⃣ Real-World Example

# In Fraud Detection:

# Features:

# Amount

# Time

# Location

# Device ID

# 200 more variables

# We select only most important features.

# Companies like PayPal and Visa heavily use feature engineering & selection.

# 🧠 8️⃣ Interview Questions

# Q1: What is feature selection?
# → Selecting important features for model training.

# Q2: Why is it important?
# → Reduce overfitting and improve performance.

# Q3: Difference between feature selection and feature extraction?

# Feature Selection → Choose existing features
# Feature Extraction → Create new features (PCA)

# Q4: Which regularization method does feature selection?
# → Lasso (L1)

# +---------------------------+---------------------------+
# | Feature Selection         | Feature Extraction        |
# +---------------------------+---------------------------+
# | Select existing features  | Create new features       |
# | Example: RFE              | Example: PCA              |
# | Keeps original meaning    | Changes feature meaning   |
# +---------------------------+---------------------------+

# 10️⃣ Summary

# Feature Selection:

# ✔ Removes irrelevant features
# ✔ Improves accuracy
# ✔ Reduces overfitting
# ✔ Speeds up training
# ✔ Important for large datasets