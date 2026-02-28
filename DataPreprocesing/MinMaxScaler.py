
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [35, 40, 50, 55, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)

# 1️⃣ VERY EASY EXPLANATION

# Imagine marks are:

# 10, 20, 30, 40, 50

# We want to convert them between:

# 0 and 1

# So:

# 10 → 0
# 50 → 1
# 30 → 0.5

# That’s what MinMaxScaler does.

# It shrinks data into a fixed range.

# Default range = 0 to 1.

# 🟡 2️⃣ FORMULA (Important)

# X_scaled = (X - Xmin) / (Xmax - Xmin)

# Where:

# Xmin = smallest value

# Xmax = largest value

# It rescales data proportionally.

# 🔎 What happens internally?

# For Hours:

# Min = 1
# Max = 5

# For value 3:


# (3−1)/(5−1)=2/4=0.5

# So:

# 1 → 0
# 3 → 0.5
# 5 → 1

# Everything now between 0 and 1.

# 🔴 4️⃣ WHY WE USE MinMaxScaler?

# When we want:

# Data in fixed range

# Neural networks

# Image processing

# Deep learning

# When algorithm expects 0–1 input

# Very common in:

# KNN

# SVM

# KMeans

# Neural Networks

# |----------------------|---------------------------|---------------------------|
# | Feature              | StandardScaler            | MinMaxScaler              |
# |----------------------|---------------------------|---------------------------|
# | Range                | Mean = 0, Std = 1         | 0 to 1                    |
# | Negative Values      | Yes                       | No                        |
# | Affected by Outliers | Less                      | More                      |
# | Distribution Shape   | Makes data Gaussian-like  | Preserves original shape  |
# | Used In              | Most ML models            | Deep Learning mostly      |
# | Formula              | (x - mean) / std          | (x - min) / (max - min)   |
# | Best For             | SVM, Logistic Regression, | Neural Networks           |
# |                      | KNN, PCA                  |                           |
# |----------------------|---------------------------|---------------------------|

# Important:

# MinMaxScaler is sensitive to outliers.

# If max value is very large,
# everything else becomes very small.

# 🟤 6️⃣ IMPORTANT INTERVIEW QUESTIONS

# Q1: When should we use MinMaxScaler?
# Answer:
# When we want fixed range data, especially for neural networks.

# Q2: What is problem with MinMaxScaler?
# Answer:
# If new data has value greater than training max, it may go beyond 1.

# Q3: Should scaling be done before or after split?
# Answer:
# After split.

# Correct method:

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

# Never do fit on test data.

# 🧠 7️⃣ REAL-WORLD EXAMPLE (Resume Ranking)

# Suppose features:

# Experience → 0 to 20
# Skills score → 0 to 100
# Salary → 2L to 20L

# Ranges are different.

# MinMaxScaler makes all between 0–1.

# Model treats them equally.

# ⚫ WHEN NOT TO USE MinMaxScaler?

# Not necessary for:

# DecisionTree

# RandomForest

# XGBoost

# Trees don’t care about scale.

# 🟢 SIMPLE SUMMARY

# StandardScaler → Center around 0
# MinMaxScaler → Shrink between 0 and 1

# Both are feature scaling methods.
