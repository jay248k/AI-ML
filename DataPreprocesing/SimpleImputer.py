# Hours	Marks
# 2	40
# 3	❌
# 4	55

# One mark is missing.

# Machine learning models cannot handle empty values.

# So we must fill them.

# That process is called Imputation.
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    "Hours": [2, 3, None, 5, 6],
    "Marks": [40, None, 50, 60, 65]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

imputer = SimpleImputer(strategy="mean")

df_filled = imputer.fit_transform(df)

print("After Imputation:")
print(df_filled)

# 1️⃣ VERY EASY EXPLANATION

# Imagine student data:

# +-------+-------+
# | Hours | Marks |
# +-------+-------+
# | 2     | 40    |
# | 3     | ❌    |
# | 4     | 55    |
# +-------+-------+

# One mark is missing.

# Machine learning models cannot handle empty values.

# So we must fill them.

# That process is called Imputation.

# SimpleImputer helps us fill missing values automatically.

# 🟡 2️⃣ WHY MISSING VALUES ARE PROBLEM?

# If dataset contains:

# None

# NaN

# Blank values

# Model will throw error.

# Example error:

# ValueError: Input contains NaN

# So we must clean data before training.

# 4️⃣ DIFFERENT STRATEGIES

# SimpleImputer has 4 main strategies:

# ✅ 1. Mean (For Numeric Data)

# Replace missing with average value.

# Example:

# Marks = [40, 50, 60]

# Mean = 50

# Missing → 50

# Used for:

# Continuous numerical features

# ✅ 2. Median

# Replace missing with middle value.

# Better when:

# Data has outliers

# Example:

# [10, 15, 20, 1000]

# Mean = 261 ❌ (affected by 1000)
# Median = 17.5 ✅ (better)

# ✅ 3. Most Frequent

# Used for categorical data.

# Example:

# City:
# Mumbai
# Delhi
# Mumbai
# None

# Most frequent = Mumbai
# Missing → Mumbai

# ✅ 4. Constant

# Fill with custom value.

# SimpleImputer(strategy="constant", fill_value=0)

# 5️⃣ CORRECT WAY (Very Important ⚠)

# Always split first.

# Then:

# imputer = SimpleImputer(strategy="mean")

# X_train = imputer.fit_transform(X_train)
# X_test = imputer.transform(X_test)

# Never fit on test data.

# Why?

# Because it causes data leakage.

# 🟤 6️⃣ INTERVIEW QUESTIONS

# Q1: Why is imputation necessary?
# Answer:
# Because ML models cannot handle missing values directly.

# Q2: Mean vs Median?
# Answer:
# Use median when outliers exist.

# Q3: What is data leakage?
# Answer:
# Using test data information during training.

# Q4: Can we drop missing rows instead?
# Answer:
# Yes, but only if missing data is very small.

# ⚫ 7️⃣ ADVANCED UNDERSTANDING

# SimpleImputer assumes:

# Missing values are random.

# If missing values have pattern,
# simple imputation may reduce model accuracy.

# In advanced ML, we use:

# KNN Imputer

# Iterative Imputer

# Model-based imputation

# 🟢 8️⃣ REAL-WORLD EXAMPLE (Your Project)

# In resume ranking:

# Some users may not fill:

# Expected salary

# Experience years

# CGPA

# You cannot delete their resume.

# So you fill:

# Salary → median salary

# Experience → 0 (if fresher)

# CGPA → mean CGPA

# This keeps model stable.

# 🟡 9️⃣ BIG ML PIPELINE VIEW

# Real ML pipeline:

# Handle missing values (SimpleImputer)

# Encode categorical values

# Scale numeric values

# Train model

# Missing value handling is first step.