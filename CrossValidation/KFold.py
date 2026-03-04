from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

data = load_iris()
X = data.data
y = data.target

kf = KFold(n_splits=5)

model = RandomForestClassifier()

scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    scores.append(accuracy_score(y_test, predictions))

print("Scores:", scores)
print("Average:", np.mean(scores))


# 1️⃣ What is KFold?

# KFold is a cross-validation technique where:

# 👉 Dataset is divided into K equal parts (folds)
# 👉 Model trains on K-1 folds
# 👉 Tests on remaining 1 fold
# 👉 Repeats K times

# Finally → Average performance is calculated.

# 🟡 2️⃣ Example (K = 5)

# Suppose you have 100 samples.

# K = 5

# Split like this:

# +------+----------------+--------------+
# | Fold | Training Data  | Testing Data |
# +------+----------------+--------------+
# | 1    | 2,3,4,5        | 1            |
# | 2    | 1,3,4,5        | 2            |
# | 3    | 1,2,4,5        | 3            |
# | 4    | 1,2,3,5        | 4            |
# | 5    | 1,2,3,4        | 5            |
# +------+----------------+--------------+

# Each fold gets chance to be test set.

# 4️⃣ Important Parameters
# KFold(n_splits=5, shuffle=True, random_state=42)
# n_splits

# Number of folds (default 5)

# shuffle

# Whether to shuffle data before splitting

# random_state

# Ensures reproducibility

# Very common interview question 🔥

# 🔴 5️⃣ Important Problem

# ⚠ KFold does NOT maintain class distribution.

# If dataset is imbalanced (like Fraud Detection):

# Some folds may have:

# Very few fraud samples

# Or zero fraud samples

# This causes bad evaluation.

# 🟠 6️⃣ Solution: StratifiedKFold

# For classification problems use:

# from sklearn.model_selection import StratifiedKFold

# StratifiedKFold keeps class ratio same in each fold.

# Very important for:

# Fraud Detection

# Spam Detection

# 🧠 7️⃣ Interview Questions

# Q1: What is KFold?
# → Splits dataset into K equal folds for cross validation.

# Q2: Difference between KFold and StratifiedKFold?
# → Stratified maintains class distribution.

# Q3: Typical value of K?
# → 5 or 10.

# Q4: Why shuffle=True?
# → To avoid biased splits.

# 🔥 8️⃣ KFold vs cross_val_score

# +------------------+------------------+
# | KFold            | cross_val_score  |
# +------------------+------------------+
# | Manual splitting | Automatic        |
# | More control     | Easy to use      |
# | Flexible         | Simple           |
# +------------------+------------------+

# Usually we combine them:

# cross_val_score(model, X, y, cv=KFold(n_splits=5))