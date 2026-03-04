import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 4, 9, 16, 25])  # y = x^2

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Train model
model = LinearRegression()
model.fit(X_poly, y)

# Predict
print(model.predict(poly.transform([[6]])))

# 1️⃣ What is PolynomialFeatures?

# It is used to:

# 👉 Convert linear features into polynomial features.

# It helps when:

# Your data is NOT linear
# But curved (non-linear relationship)

# 🟡 2️⃣ Why Do We Need It?

# Imagine:

# Hours studied vs Marks.

# If relation is straight line → Linear Regression works.

# But if relation is curve:

# Marks = 2(Hours)² + 3(Hours) + 5

# Linear regression alone cannot model this curve.

# So we convert:

# X → X² → X³ → etc.

# Then apply Linear Regression.

# This is called:

# 👉 Polynomial Regression

# 🔵 3️⃣ Example Without Polynomial

# Data:

# Hours → Marks

# If graph looks curved, linear line cannot fit properly.

# 5️⃣ What Happens Internally?

# If:

# X = [2]

# degree = 2

# PolynomialFeatures converts:

# [2] → [1, 2, 4]

# Because:

# 1 (bias term)
# x
# x²

# If degree = 3:

# [1, x, x², x³]

# 🔴 6️⃣ Important Parameters
# PolynomialFeatures(degree=2, include_bias=True)
# degree

# Maximum power of features.

# include_bias

# If True → adds column of 1.

# 🟠 7️⃣ When To Use PolynomialFeatures?

# ✔ When data is non-linear
# ✔ When linear regression underfits
# ✔ When relationship is curved

# Not needed for:

# Decision Trees

# Random Forest

# Neural Networks

# Because they already handle non-linearity.

# 🧠 8️⃣ Interview Questions

# Q1: Is polynomial regression different algorithm?
# → No, it is Linear Regression on transformed features.

# Q2: Why can Linear Regression model curves with polynomial features?
# → Because features are transformed.

# Q3: What is risk of high degree polynomial?
# → Overfitting.

# 🔥 9️⃣ Overfitting Danger

# If degree = 10 or 20:

# Model may perfectly fit training data
# But fail on test data ❌

# So always check:

# R² score

# Cross-validation