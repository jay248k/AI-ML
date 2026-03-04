from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.datasets import load_iris

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Select top 2 features
selector = SelectKBest(score_func=f_classif, k=2)
X_new = selector.fit_transform(X, y)

print("Selected shape:", X_new.shape)
print("Scores:", selector.scores_)


# 1️⃣ What is SelectKBest?

# SelectKBest is a feature selection method that:

# 👉 Selects top K important features
# 👉 Based on statistical test scores

# It is a Filter Method (fast & simple).

# 🟡 2️⃣ Why Use It?

# When dataset has:

# Too many features

# Noise features

# Irrelevant columns

# We choose only top K best features.

# Example:

# If dataset has 20 features
# We can select top 5 best features.

# 🔵 3️⃣ How It Works?

# It:

# 1️⃣ Applies statistical test
# 2️⃣ Calculates score for each feature
# 3️⃣ Selects highest K scores

# 🟣 4️⃣ Important Score Functions

# For Classification:

# chi2 (for categorical / non-negative data)

# f_classif (ANOVA F-test)

# mutual_info_classif

# For Regression:

# f_regression

# mutual_info_regression

# 6️⃣ What Happens Internally?

# Suppose features:

# +---------+-------+
# | Feature | Score |
# +---------+-------+
# | f1      | 2.1   |
# | f2      | 9.3   |
# | f3      | 1.4   |
# | f4      | 7.8   |
# +---------+-------+

# If k=2 → Select:

# ✔ f2
# ✔ f4

# 🟠 7️⃣ Important Parameter
# SelectKBest(score_func=f_classif, k=5)
# score_func

# Statistical test

# k

# Number of features to select

# If k="all" → select all features

# 🧠 8️⃣ Interview Questions

# Q1: Is SelectKBest filter or wrapper?
# → Filter method

# Q2: Does it use model training?
# → No (it uses statistical tests)

# Q3: When should we use chi-square?
# → When features are non-negative and categorical.

# Q4: What is disadvantage?
# → It ignores feature interaction.

# 🔥 9️⃣ When NOT to Use

# ❌ When features are highly dependent
# ❌ When feature interaction matters
# ❌ When data is very small

# Better options then:

# RFE

# Lasso

# Random Forest importance

# 🚀 10️⃣ Summary

# SelectKBest:

# ✔ Fast
# ✔ Simple
# ✔ Selects top K features
# ✔ Based on statistical score
# ✔ Filter method