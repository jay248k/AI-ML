import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 1️⃣ VERY EASY EXPLANATION

# Imagine you are selecting a candidate.

# You create rules:

# If Experience > 3
#  If Skill Score > 80 → Selected
#  Else → Not Selected
# Else → Not Selected

# That rule-based system is a Decision Tree.

# So:

# DecisionTreeClassifier = IF-ELSE rule model 🌳

# 🟡 2️⃣ HOW IT WORKS

# Step 1: Choose best feature to split
# Step 2: Split dataset into groups
# Step 3: Repeat splitting
# Step 4: Stop when data becomes pure

# “Pure” means:

# All samples in node belong to same class.

# 🔵 3️⃣ HOW DOES IT CHOOSE BEST SPLIT?

# For classification, it uses:

# 👉 Gini Index (default)
# 👉 Entropy (Information Gain)

# 🔷 Gini Index

# Formula:

# Gini=1-∑p²

# If node is pure → Gini = 0

# Lower Gini = Better split

# 🔷 Entropy

# Formula:

# Gini = 1-∑p₂

# If node is pure → Gini = 0

# Lower Gini = Better split

# 🔷 Entropy

# Formula:

# Entropy=-∑plog(p)

# Entropy measures randomness.

# Lower entropy = Better split

# Difference:

# Gini → Faster
# Entropy → Slightly more precise

# In sklearn:

# criterion="gini"
# criterion="entropy"

# 5️⃣ IMPORTANT PARAMETERS

# max_depth → Limits tree height

# min_samples_split → Minimum samples to split

# min_samples_leaf → Minimum samples in leaf

# criterion → gini or entropy

# If not controlled → Overfitting

# ⚫ 6️⃣ OVERFITTING PROBLEM

# Decision Trees:

# Very powerful
# Very flexible

# But:

# If tree grows fully → Memorizes training data

# Training accuracy = 100%
# Test accuracy = Low

# Solution:

# ✔ Set max_depth
# ✔ Use pruning
# ✔ Use RandomForest

# 🟠 7️⃣ DOES IT NEED SCALING?

# No ❌

# Trees don’t use distance or gradient.

# Very important interview point 🔥

# +----------------------+-----------------+
# | Logistic Regression  | Decision Tree   |
# +----------------------+-----------------+
# | Linear boundary      | Non-linear      |
# | Uses sigmoid         | Uses splits     |
# | Needs scaling        | No scaling      |
# | Stable               | Can overfit     |
# +----------------------+-----------------+

# 9️⃣ INTERVIEW QUESTIONS

# Q1: What is Gini Index?
# Measure of impurity.

# Q2: What is entropy?
# Measure of randomness.

# Q3: Why does Decision Tree overfit?
# Because it keeps splitting until pure.

# Q4: Is Decision Tree parametric?
# No.

# Q5: Does it handle categorical data?
# Yes.

# 🟢 10️⃣ REAL PROJECT (Your Job Portal)

# Predict:

# Will candidate be selected?

# Tree might create rules:

# If Experience > 2
#  If Projects > 3 → Selected
# Else → Not Selected

# Very interpretable.

# HR people like this model because they can see rules.

# 🔥 11️⃣ SUMMARY

# DecisionTreeClassifier:

# ✔ Rule-based model
# ✔ Uses Gini or Entropy
# ✔ No scaling needed
# ✔ Handles non-linear data
# ❌ Can overfit

# Now you have learned:

# Regression models
# Classification models
# Tree models
# Ensemble model
# Distance-based model