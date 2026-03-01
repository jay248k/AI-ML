# Linear Regression → uses everything
# Ridge → reduces weight but keeps everything
# Lasso → removes useless features (weight = 0)


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

# Sample dataset
data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Marks": [35,40,50,55,60,65,75,80]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Ridge(alpha=1.0)  # alpha = lambda
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)
print("Weights:", model.coef_)

# 1️⃣ VERY EASY EXPLANATION

# Linear Regression tries to fit:

# y = w1*x1 + w2*x2 + ... + wn*xn + b

# But sometimes:

# 👉 Model gives VERY BIG weight values
# 👉 Model overfits
# 👉 Features are highly correlated (multicollinearity)

# So we control weights.

# Ridge Regression = Linear Regression + Penalty for large weights.

# It says:

# “Okay… you can learn weights… but don’t make them too big.”

# 🟡 2️⃣ WHY WE NEED RIDGE?

# Problem 1: Overfitting
# Problem 2: Multicollinearity
# Problem 3: Large variance

# Example:

# Experience and Age are highly correlated.

# Linear regression becomes unstable.

# Ridge stabilizes it.

# 🔵 3️⃣ MATHEMATICAL FORMULA (Important)

# Normal Linear Regression Loss:
# Loss = ∑(yᵢ - ŷ)²

# Ridge Regression adds penalty:
# Loss = ∑(yᵢ - ŷ)² + α∑wⱼ²

# Where:

# Where:

# λ (lambda) = regularization parameter

# w² = square of weights

# This is called:

# 👉 L2 Regularization

# 🟣 4️⃣ WHAT λ (Lambda) DOES

# If:

# λ = 0 → Same as Linear Regression

# λ small → Small penalty

# λ big → Strong penalty → weights shrink

# Important:

# Ridge makes weights SMALL
# BUT never exactly ZERO

# In sklearn:

# alpha = λ

# 🔴 6️⃣ DIFFERENCE: LINEAR vs RIDGE

# +--------------------+----------------------+
# | Linear Regression  | Ridge                |
# +--------------------+----------------------+
# | No penalty         | Has penalty          |
# | Can overfit        | Reduces overfitting  |
# | Weights large      | Weights shrink       |
# | No regularization  | L2 regularization    |
# +--------------------+----------------------+

# 7️⃣ GEOMETRIC INTUITION (Advanced)

# Linear Regression:

# Minimizes error only.

# Ridge:

# Minimizes error inside a circle constraint.

# That constraint forces weights to stay small.

# 🟠 8️⃣ WHEN TO USE RIDGE?

# Use Ridge when:

# ✔ Many features
# ✔ Multicollinearity exists
# ✔ Linear regression overfits
# ✔ All features are important

# 🟡 9️⃣ WHEN NOT TO USE RIDGE?

# If you want:

# Feature selection

# Because Ridge never makes weight zero.

# For feature selection use:

# 👉 Lasso Regression

# 🧠 10️⃣ INTERVIEW QUESTIONS

# Q1: What type of regularization does Ridge use?
# Answer: L2 Regularization

# Q2: What happens if alpha increases?
# Weights shrink more.

# Q3: Does Ridge perform feature selection?
# No.

# Q4: What problem does Ridge solve?
# Multicollinearity and overfitting.

# 🟢 11️⃣ REAL WORLD EXAMPLE (Resume Ranking)

# Suppose you build ML model in your Job Portal:

# Features:

# Experience
# Skill Score
# Projects
# Certifications

# Linear regression might give:

# Experience = 100
# Skill Score = 0.0001

# Unstable.

# Ridge balances them properly.

# 🔥 12️⃣ SIMPLE SUMMARY

# Linear Regression = Fit best line

# Ridge = Fit best line + Don’t allow crazy weights

# It makes model more stable.