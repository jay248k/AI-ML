from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

data = load_iris()
X = data.data
y = data.target

model = RandomForestClassifier()

scores = cross_val_score(model, X, y, cv=5)

print("Scores:", scores)
print("Average score:", scores.mean())


# 1️⃣ What is cross_val_score?

# cross_val_score is used to:

# 👉 Evaluate model performance using Cross Validation

# Instead of splitting data once (train/test),
# it splits multiple times and averages the score.

# This gives more reliable performance.

# 🟡 2️⃣ Why Not Only train_test_split?

# When you do:

# train_test_split()

# You evaluate model on only one split.

# Problem:

# Result depends on random split ❌
# Maybe lucky split → high accuracy
# Maybe unlucky split → low accuracy

# So we use:

# 👉 Cross Validation

# 🔵 3️⃣ What is Cross Validation?

# Most common type:

# K-Fold Cross Validation

# If k=5:

# Split data into 5 parts

# Train on 4 parts

# Test on 1 part

# Repeat 5 times

# Take average score

# Output example:

# Scores: [0.96 0.93 1.00 0.90 0.96]
# Average score: 0.95

# 5️⃣ Important Parameters
# cross_val_score(model, X, y, cv=5, scoring='accuracy')
# model

# Your ML model

# X, y

# Data

# cv

# Number of folds (usually 5 or 10)

# scoring

# Metric to evaluate

# Examples:

# 'accuracy'

# 'precision'

# 'recall'

# 'f1'

# 'r2'

# Very common interview question 🔥

# 🔴 6️⃣ Why It Is Powerful?

# ✔ Uses full dataset for training and testing
# ✔ Reduces overfitting risk
# ✔ Gives stable performance estimate
# ✔ Better than single train/test split

# Companies use cross-validation in model validation pipelines.

# 🟠 7️⃣ For Imbalanced Data (Important)

# Use:

# from sklearn.model_selection import StratifiedKFold

# StratifiedKFold keeps class distribution equal.

# Very important for:

# Fraud Detection

# Spam Detection

# 🧠 8️⃣ Interview Questions

# Q1: What is cross validation?
# → Technique to evaluate model multiple times on different splits.

# Q2: Why is it better than train_test_split?
# → More stable and reliable.

# Q3: What is K in K-Fold?
# → Number of splits.

# Q4: Typical value of K?
# → 5 or 10.

# Q5: Can we use it for regression?
# → Yes (use scoring='r2').

# 🔥 9️⃣ When NOT to Use Simple CV?

# For time-series data:

# Use:

# 👉 TimeSeriesSplit

# Because time order matters.