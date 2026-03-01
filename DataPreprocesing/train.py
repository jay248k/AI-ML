import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Create sample student data
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 50, 55, 60, 65, 75, 80, 85, 95]
}

df = pd.DataFrame(data)

# print(df)
X = df[["Hours"]]   # Independent variable
y = df["Marks"]     # Target variable


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2 #, random_state=3
)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Marks:", list(y_test))
print("Predicted Marks:", y_pred)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# 1️⃣ What is train_test_split?

# It is used to:

# 👉 Split your dataset into

# Training Data

# Testing Data

# 🟡 2️⃣ Why Do We Need It?

# If you train and test on same data:

# Model will just memorize 😅
# This is called Overfitting

# So we:

# Train model on one part

# Test model on unseen data

# Just like exam:

# Study from book 📘

# Give exam without book 📝

# 4️⃣ Meaning of Parameters
# 🔹 X

# Features (input data)

# 🔹 y

# Target (output labels)

# 🔹 test_size=0.2

# 20% data for testing
# 80% for training

# 🔹 random_state=42

# Fixes randomness
# So result is same every time

# Very common interview question 🔥

# 6️⃣ What Happens Internally?

# If dataset has 100 rows:

# test_size=0.2

# → 80 rows → Training
# → 20 rows → Testing

# Randomly selected.

# 🟠 7️⃣ Important Parameter: stratify

# Used for classification problems.

# Example:

# train_test_split(X, y, test_size=0.2, stratify=y)

# Why?

# If dataset is imbalanced (Fraud Detection):

# Stratify keeps same class ratio in train and test.

# VERY IMPORTANT 🔥

# 🧠 8️⃣ Interview Questions

# Q1: Why use train_test_split?
# → To evaluate model on unseen data.

# Q2: What does random_state do?
# → Makes splitting reproducible.

# Q3: What is good test_size?
# → Usually 0.2 or 0.3.

# Q4: What is stratify used for?
# → Maintain class distribution.

# 🔥 9️⃣ Best Practice

# For small datasets:

# Use:

# 👉 Cross Validation

# Instead of single split.

# We will learn that next 😉

# 🚀 Final Summary

# train_test_split:

# ✔ Prevents overfitting
# ✔ Separates training & testing data
# ✔ Essential before model training
# ✔ Use stratify for classification