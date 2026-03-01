from sklearn.metrics import r2_score

y_true = [3, -0.5, 2, 7]
y_pred = [2.5, 0.0, 2, 8]

r2 = r2_score(y_true, y_pred)
print("R2 Score:", r2)

#  1️⃣ What is R² Score?

# R² (R-squared) measures:

# 👉 How well your regression model explains the variance in the data.

# In simple words:

# How good is your model at predicting continuous values?

# Used in:

# House price prediction

# Salary prediction

# Stock prediction

# Sales forecasting

# 🟡 2️⃣ Simple Meaning

# If:

# R² = 1 → Perfect prediction ✅
# R² = 0 → Model is useless (same as predicting mean)
# R² < 0 → Very bad model ❌

# 🔵 3️⃣ Formula (Important for Interview)

# R²=1-SSres /SStot


# Where:

# SSres  = Sum of squared errors (actual - predicted)²

# 𝑆Stot= Total variance in actual values

# Interpretation:

# How much variance your model explained.

# 🟣 4️⃣ Easy Example

# Suppose:

# You predict house prices.

# Actual prices:

# [100, 200, 300]

# Predicted prices:

# [110, 190, 290]

# Small error → High R².

# 6️⃣ Interpretation Table

# +------------+---------------------+
# | R² Value   | Meaning             |
# +------------+---------------------+
# | 1.0        | Perfect model       |
# | 0.8+       | Very good           |
# | 0.6–0.8    | Good                |
# | 0.4–0.6    | Moderate            |
# | 0          | No predictive power |
# | < 0        | Worse than mean     |
# +------------+---------------------+

# 7️⃣ Important Difference

# Classification metrics:

# accuracy

# precision

# recall

# F1

# Regression metrics:

# R²

# MAE

# MSE

# RMSE

# Very common interview question 🔥

# 🧠 8️⃣ When R² is Misleading?

# R² always increases when you add more features.

# So for multiple regression we use:

# 👉 Adjusted R² (advanced concept)

# 🟢 9️⃣ Interview Questions

# Q1: What is R²?
# → Proportion of variance explained by model.

# Q2: Can R² be negative?
# → Yes.

# Q3: Is higher R² always better?
# → Generally yes, but overfitting possible.

# Q4: Difference between R² and Adjusted R²?
# → Adjusted R² penalizes extra features.

# 🔥 Final Understanding

# R² tells:

# How much better your model is compared to predicting the average.

# If R² = 0.85
# → Model explains 85% of data variance.