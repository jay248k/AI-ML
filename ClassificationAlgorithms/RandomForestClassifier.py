import pandas as pd
from sklearn.ensemble import RandomForestClassifier
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

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 🌲🌲 RandomForestClassifier

# It is an advanced version of DecisionTreeClassifier.

# Very commonly used in real-world ML.

# 🟢 1️⃣ VERY EASY EXPLANATION

# Decision Tree = One interviewer deciding

# Random Forest = 100 interviewers voting together 😎

# Each tree gives prediction:

# Selected / Not Selected

# Final answer = Majority vote

# So:

# RandomForestClassifier = Many decision trees + Voting

# 🟡 2️⃣ WHY WE NEED IT?

# Problem with single Decision Tree:

# ❌ Overfits
# ❌ High variance
# ❌ Unstable

# Random Forest fixes this by:

# ✔ Using many trees
# ✔ Using random data samples
# ✔ Using random features

# Result:

# More accurate
# More stable
# Less overfitting

# 🔵 3️⃣ HOW IT WORKS

# Step 1: Take random sample from dataset (Bootstrap)

# Step 2: Build a decision tree

# Step 3: At each split, choose random subset of features

# Step 4: Repeat many times (100+ trees)

# Step 5: Final prediction = Majority vote

# For example:

# Tree 1 → Selected
# Tree 2 → Selected
# Tree 3 → Not Selected

# Final → Selected (2 votes)

# 5️⃣ IMPORTANT PARAMETERS

# n_estimators → Number of trees

# max_depth → Tree depth

# min_samples_split → Minimum samples to split

# min_samples_leaf → Minimum samples per leaf

# max_features → Features used at each split

# More trees → Better accuracy (usually) but slower

# ⚫ 6️⃣ DOES IT NEED SCALING?

# No ❌

# Trees do not use distance or gradients.

# Very important interview point 🔥

# 🟠 7️⃣ ADVANTAGES

# ✔ Very high accuracy
# ✔ Handles non-linear data
# ✔ Reduces overfitting
# ✔ Works with many features
# ✔ Gives feature importance

# 🟤 8️⃣ DISADVANTAGES

# ❌ Slower than single tree
# ❌ Large memory usage
# ❌ Harder to interpret

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: What is bagging?
# Training multiple models on random subsets and averaging/voting.

# Q2: Why does Random Forest reduce overfitting?
# Because averaging multiple trees reduces variance.

# Q3: What is OOB score?
# Out-of-Bag score (internal validation using unused samples).

# Q4: Does Random Forest perform feature selection?
# Yes, indirectly using feature importance.

# Q5: Difference between Random Forest and Decision Tree?

# Decision Tree → One tree
# Random Forest → Many trees

# 🟢 10️⃣ FEATURE IMPORTANCE

# You can check:

# model.feature_importances_

# In your Job Portal example:

# Experience → 0.45
# Skill Score → 0.30
# Projects → 0.15
# CGPA → 0.10

# This helps HR understand key factors.

# 11️⃣ DECISION TREE vs RANDOM FOREST

# +----------------+-------------------+
# | Decision Tree  | Random Forest     |
# +----------------+-------------------+
# | One tree       | Many trees        |
# | Overfits easily| Less overfitting  |
# | Fast           | Slower            |
# | Simple         | More accurate     |
# +----------------+-------------------+

# 12️⃣ REAL PROJECT CONNECTION

# In your MERN + ML Job Portal:

# Use RandomForestClassifier to predict:

# Will candidate be selected?

# Features:

# Experience
# Skill Score
# Projects
# Certifications
# CGPA

# It will give:

# 0 → Not selected
# 1 → Selected

# Plus probability.

# Very practical model 🔥

# 🚀 13️⃣ SUMMARY

# RandomForestClassifier:

# ✔ Ensemble method
# ✔ Uses bagging
# ✔ Majority voting
# ✔ High accuracy
# ✔ No scaling needed