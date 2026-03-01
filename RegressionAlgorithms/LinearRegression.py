# Linear Regression → uses everything
# Ridge → reduces weight but keeps everything
# Lasso → removes useless features (weight = 0)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Create dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Marks": [35,40,50,55,60,65,75,80]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Predictions:", y_pred)

# 1️⃣ VERY EASY EXPLANATION

# Imagine this:

# More hours of study → More marks.

# If we draw a straight line through points, it may look like:

# Marks = 5 × Hours + 30

# That straight-line relationship is called:

# 👉 Linear Regression

# It predicts a number using a straight line.

# Simple meaning:

# It finds the BEST straight line that fits the data.

# 3️⃣ MATHEMATICAL FORMULA

# Simple Linear Regression:

# y = mx + b

# Where:

# y = predicted output

# x = input feature

# m = slope (weight)

# c = intercept

# 🔴 4️⃣ HOW MODEL LEARNS (Important)

# Model tries to minimize error.

# Error = Actual − Predicted

# Loss Function used:
# Mean Squared Error (MSE) = (1/n) Σ (yᵢ − ŷᵢ)²

# This is called:

# 👉 Least Squares Method

# It finds values of m and c that make error smallest.

# 🟣 5️⃣ MULTIPLE LINEAR REGRESSION

# If more than one feature:

# Experience
# Skills
# Education

# Then formula becomes:
# y = m₁x₁ + m₂x₂ + ... + mₙxₙ + c

# In matrix form:
# Y = XW + C

# Where:
# Where:

# X = feature matrix

# β = weights

# 🟤 6️⃣ ASSUMPTIONS (Interview Important)

# Linear regression assumes:

# Linear relationship

# No multicollinearity

# Homoscedasticity (constant variance)

# Errors are normally distributed

# If these break 

# → performance drops.

# ⚫ 7️⃣ WHEN TO USE?

# Use when:

# ✔ Output is numeric
# ✔ Relationship is roughly linear

# Examples:

# Salary prediction

# House price prediction

# Marks prediction

# Revenue prediction

# 🔵 WHEN NOT TO USE?

# ❌ If relationship is non-linear
# ❌ Complex patterns
# ❌ Classification problems

# Then use:

# Polynomial Regression

# Decision Trees

# Random Forest

# 🟡 8️⃣ IMPORTANT ATTRIBUTES

# After training:

# model.coef_

# Gives slope (m)

# model.intercept_

# Gives intercept (c)

# So final equation becomes:

# Marks = m × Hours + c

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: Why is it called linear?
# Because output is linear combination of inputs.

# Q2: What is overfitting in linear regression?
# When model fits noise instead of pattern.

# Q3: Difference between Linear and Logistic Regression?
# Linear → continuous output
# Logistic → classification (0 or 1)

# Q4: What if features are correlated?
# It causes multicollinearity problem.

# 🟢 10️⃣ REAL-WORLD (Your Resume Project)

# Suppose resume score depends on:

# Experience
# Skill score
# CGPA

# Model learns:

# Score = 2×Experience + 0.5×Skill + 1.2×CGPA + 10

# Then predicts ranking score.

# 🟣 11️⃣ BIG PICTURE

# Machine Learning Flow:

# Data → Clean → Split → Train → Predict → Evaluate → Improve

# Linear Regression is foundation of ML.

# Understanding this deeply makes everything easier.