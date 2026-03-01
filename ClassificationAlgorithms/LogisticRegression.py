import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Pass":  [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print("Predictions:", predictions)
print("Probabilities:", probabilities)

# 1️⃣ VERY EASY EXPLANATION

# Linear Regression → Predicts numbers (like salary)

# Logistic Regression → Predicts categories (like Pass/Fail)

# Example:

# If student studies 5 hours
# Will he PASS or FAIL?

# Output:

# 0 = Fail
# 1 = Pass

# So Logistic Regression predicts probability.

# 🟡 2️⃣ WHY CAN’T WE USE LINEAR REGRESSION?

# Linear Regression gives output like:

# -2
# 1.7
# 10

# But probability must be between:

# 0 and 1

# So we use a special function:

# 👉 Sigmoid Function

# 🔵 3️⃣ SIGMOID FUNCTION (Heart of Logistic Regression)

# Formula:

# σ(z)=1/(1+e^(-z))

# Where:

# z = (w1*x1 + w2*x2 + ... + wn*xn) + b

# Sigmoid converts any number into:

# 0 to 1 range

# Graph shape: S-shaped curve

# If output > 0.5 → Class 1
# If output < 0.5 → Class 0

# 🟣 4️⃣ HOW MODEL LEARNS?

# Linear regression uses MSE.

# Logistic regression uses:

# 👉 Log Loss (Binary Cross Entropy)

# Formula:

# Log Loss = -(y*log(p) + (1-y)*log(1-p))

# Why?

# Because we are predicting probabilities.

# 6️⃣ LINEAR vs LOGISTIC REGRESSION
# +----------------------+-------------------+
# | Linear               | Logistic          |
# +----------------------+-------------------+
# | Predict numbers      | Predict classes   |
# | Uses MSE             | Uses Log Loss     |
# | Straight line        | S-shaped curve    |
# | No probability output| Gives probability |
# +----------------------+-------------------+

# 7️⃣ TYPES OF LOGISTIC REGRESSION

# Binary → 0 or 1
# Multinomial → 3+ classes
# One-vs-Rest strategy

# Example:

# Low / Medium / High salary category

# 🟠 8️⃣ DOES IT NEED SCALING?

# Yes, usually ✔

# Logistic Regression performs better when features are scaled.

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: Why is it called Logistic Regression?
# Because it uses logistic (sigmoid) function.

# Q2: What loss function is used?
# Log Loss / Cross Entropy.

# Q3: Output range?
# Between 0 and 1.

# Q4: Difference between Softmax and Sigmoid?
# Sigmoid → Binary classification
# Softmax → Multi-class classification

# Q5: Can Logistic Regression overfit?
# Yes. Use regularization (L1, L2).

# 🟢 10️⃣ REAL PROJECT CONNECTION (Your Job Portal)

# Example:

# Predict:

# Will candidate get selected?

# Features:

# Experience
# Skill Score
# Projects
# CGPA

# Output:

# 0 → Not Selected
# 1 → Selected

# Model gives probability:

# 0.82 → 82% chance of selection

# That’s powerful 🔥

# 🔥 11️⃣ REGULARIZATION IN LOGISTIC REGRESSION

# It supports:

# L1 (Lasso)
# L2 (Ridge)

# In sklearn:

# LogisticRegression(penalty='l2')

# 12️⃣ FINAL SUMMARY

# Linear Regression → Predict number

# Logistic Regression → Predict probability → Convert to class

# Uses Sigmoid
# Uses Log Loss
# Used for classification
