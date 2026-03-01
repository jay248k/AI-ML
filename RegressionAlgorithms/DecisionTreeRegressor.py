import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

# Dataset
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

model = DecisionTreeRegressor(max_depth=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 1️⃣ VERY EASY EXPLANATION

# Linear Regression draws a straight line.

# But what if data is NOT straight?

# Example:

# Hours studied → Marks

# Maybe:

# 0–2 hours → 30 marks
# 3–5 hours → 50 marks
# 6–8 hours → 80 marks

# This is not a straight line.

# Decision Tree says:

# 👉 “I will split the data into groups.”

# It creates rules like:

# If Hours ≤ 2 → Predict 30
# If 2 < Hours ≤ 5 → Predict 50
# If Hours > 5 → Predict 80

# So:

# Decision Tree = Series of IF-ELSE rules 🌳

# 🟡 2️⃣ HOW IT WORKS (STEP-BY-STEP)

# Step 1: Choose best feature to split
# Step 2: Divide data into two parts
# Step 3: Repeat splitting
# Step 4: Stop when condition met

# Final structure looks like:

#           Hours ≤ 5?
#            /      \
#         Yes        No
#       Marks=50   Marks=80

# Each end node = prediction value.

# 🔵 3️⃣ HOW DOES IT CHOOSE BEST SPLIT?

# For regression, it uses:

# 👉 MSE (Mean Squared Error)

# It tries to split data such that:

# Variance inside each group is minimum.

# Formula:

# MSE = (1/n) * ∑(yᵢ - ŷ)²

# Tree chooses split that reduces MSE most.

# 5️⃣ IMPORTANT PARAMETERS

# max_depth → Controls tree height
# min_samples_split → Minimum samples to split
# min_samples_leaf → Minimum samples per leaf
# max_features → Features to consider

# If you don’t control these → Overfitting.

# 🔴 6️⃣ BIG PROBLEM: OVERFITTING

# Decision Trees are powerful.

# Too powerful sometimes.

# If tree grows too deep:

# It memorizes training data.

# Training accuracy = 100%
# Test accuracy = Poor

# Solution:

# ✔ Set max_depth
# ✔ Use pruning
# ✔ Use RandomForest

# 7️⃣ LINEAR REGRESSION vs DECISION TREE

# +-------------------------------+--------------------------+
# | Linear Regression             | Decision Tree            |
# +-------------------------------+--------------------------+
# | Straight line                 | Tree rules               |
# | Assumes linear                | No assumption            |
# | Simple                        | Flexible                 |
# | Low variance                  | High variance            |
# | Cannot handle complex patterns| Handles complex patterns |
# +-------------------------------+--------------------------+

# 8️⃣ WHEN TO USE?

# Use Decision Tree when:

# ✔ Relationship is non-linear
# ✔ Data is complex
# ✔ You want interpretability (rules)
# ✔ Mixed features (numeric + categorical)

# Example:

# Job candidate score prediction.

# If:

# Experience > 3 years AND Skill score > 80
# → High score

# Tree handles this naturally.

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: What metric does DecisionTreeRegressor use?
# Answer: MSE / Variance reduction

# Q2: Why does Decision Tree overfit?
# Because it keeps splitting until pure nodes.

# Q3: How to prevent overfitting?
# Limit depth, pruning, RandomForest.

# Q4: Is scaling required?
# No. Trees don’t need scaling.

# Important point 🔥

# 🟢 10️⃣ REAL PROJECT CONNECTION (Your Job Portal)

# Suppose you predict resume score.

# Tree might create rules:

# If Experience > 5
# If Projects > 3
# Score = High
# Else
# Score = Medium

# Very practical.

# 🔥 11️⃣ SUMMARY

# Linear Regression → Draws line

# Decision Tree → Creates rule-based structure

# No scaling needed
# Handles non-linear patterns
# Can overfit easily