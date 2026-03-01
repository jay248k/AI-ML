import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

model = RandomForestRegressor(
    n_estimators=100, 
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 1️⃣ VERY EASY EXPLANATION

# Decision Tree = One smart student

# Random Forest = 100 smart students voting together 😎

# Instead of building ONE tree, it builds MANY trees.

# Final prediction = Average of all tree predictions.

# So:

# Random Forest = Many Decision Trees + Averaging

# 🟡 2️⃣ WHY WE NEED IT?

# Problem with Decision Tree:

# ❌ Overfitting
# ❌ High variance

# Random Forest fixes this by:

# ✔ Using multiple trees
# ✔ Using different random samples
# ✔ Using different random features

# Result:

# More stable
# Better generalization
# Less overfitting

# 🔵 3️⃣ HOW IT WORKS (STEP-BY-STEP)

# Step 1: Take random sample from dataset (Bootstrap sampling)

# Step 2: Build a decision tree on that sample

# Step 3: At each split, choose random subset of features

# Step 4: Repeat this process many times

# Step 5: Final prediction = Average of all trees

# Formula:

# Prediction=1/N∑Tree_i(X)

# 5️⃣ IMPORTANT PARAMETERS

# n_estimators → Number of trees (default 100)

# max_depth → Controls tree size

# min_samples_split → Minimum samples to split

# min_samples_leaf → Minimum samples per leaf

# max_features → Number of features considered per split

# 🔴 6️⃣ WHY RANDOM FOREST IS POWERFUL

# Because of:

# ✔ Bagging (Bootstrap Aggregation)
# ✔ Random feature selection
# ✔ Averaging reduces variance

# This makes it:

# More accurate than single tree
# Less overfitting
# Robust

# 7️⃣ DECISION TREE vs RANDOM FOREST

# +----------------+-----------------+
# | Decision Tree  | Random Forest   |
# +----------------+-----------------+
# | One tree       | Many trees      |
# | High variance  | Low variance    |
# | Overfits easily| More stable     |
# | Fast           | Slower          |
# | Less accurate  | More accurate   |
# +----------------+-----------------+

# 8️⃣ DOES IT NEED SCALING?

# No ❌

# Trees do not require feature scaling.

# Very important interview point 🔥

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: What is bagging?
# Answer: Training multiple models on random subsets and averaging results.

# Q2: Why does Random Forest reduce overfitting?
# Because averaging multiple trees reduces variance.

# Q3: What happens if n_estimators increases?
# Better performance (usually), but slower.

# Q4: Does Random Forest perform feature selection?
# Yes, indirectly via feature importance.

# 🟢 10️⃣ FEATURE IMPORTANCE (Very Important)

# Random Forest can tell:

# Which feature is most important.

# Example:

# model.feature_importances_

# In your Job Portal:

# Experience → 0.40
# Skills → 0.35
# Projects → 0.20
# CGPA → 0.05

# This is powerful for analysis 🔥

# 🟣 11️⃣ WHEN TO USE?

# Use Random Forest when:

# ✔ Complex data
# ✔ Non-linear relationships
# ✔ Many features
# ✔ You want high accuracy
# ✔ Decision Tree overfits

# It works very well in:

# Salary prediction

# Resume ranking

# Credit scoring

# House price prediction

# 🔥 12️⃣ SUMMARY

# Decision Tree = One brain

# Random Forest = Many brains voting

# Less overfitting
# Better performance
# No scaling needed
# Handles non-linearity