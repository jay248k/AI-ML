# Linear Regression → uses everything
# Ridge → reduces weight but keeps everything
# Lasso → removes useless features (weight = 0)

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

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

model = Lasso(alpha=0.5)
model.fit(X_train, y_train)

print("Weight:", model.coef_)
print("Intercept:", model.intercept_)

# 1️⃣ VERY EASY EXPLANATION

# Linear Regression → no control on weights

# Ridge → makes weights small

# Lasso → makes some weights ZERO

# That means:

# 👉 Lasso automatically removes useless features.

# So:

# Lasso = Linear Regression + Feature Selection

# 🟡 2️⃣ WHY WE NEED LASSO?

# Imagine your dataset has:

# Experience

# Skill score

# CGPA

# Random noise column

# Useless feature

# Linear Regression → uses everything
# Ridge → reduces weight but keeps everything
# Lasso → removes useless features (weight = 0)

# That is powerful 🔥

# 🔵 3️⃣ MATHEMATICAL FORMULA

# Normal Linear Regression:

# Loss=∑(yᵢ - ŷ)²

# Lasso adds penalty:
# Loss=∑(yᵢ - ŷ)² + α∑|wⱼ|

# Notice:

# Ridge → w²
# Lasso → |w|

# This is called:

# 👉 L1 Regularization

# 🔴 4️⃣ WHAT λ (alpha) DOES

# If:

# λ = 0 → Same as Linear Regression

# λ small → Small penalty

# λ big → More weights become 0

# In sklearn:

# alpha = λ

# If you had multiple features, some coefficients might become 0.

# 🟤 6️⃣ LASSO vs RIDGE (Very Important)

# +---------------------------------+----------------------------------+
# | Ridge                           | Lasso                            |
# +---------------------------------+----------------------------------+
# | L2 penalty                      | L1 penalty                       |
# | Shrinks weights                 | Makes some weights zero          |
# | No feature selection            | Yes feature selection            |
# | Good when all features important| Good when many useless features  |
# +---------------------------------+----------------------------------+

# 7️⃣ GEOMETRIC INTUITION (Advanced Understanding)

# Ridge constraint = Circle
# Lasso constraint = Diamond

# Diamond shape touches axis → some weights become exactly zero.

# That’s why Lasso removes features.

# ⚫ 8️⃣ WHEN TO USE LASSO?

# Use Lasso when:

# ✔ Many features
# ✔ Some features useless
# ✔ Need automatic feature selection
# ✔ High-dimensional data

# Example:

# Resume screening model with 100+ features.

# Lasso selects important ones.

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: What type of regularization does Lasso use?
# Answer: L1 regularization

# Q2: Why does Lasso perform feature selection?
# Because L1 penalty can shrink weights to exactly zero.

# Q3: Which is better Ridge or Lasso?
# Depends:

# All features important → Ridge

# Many useless features → Lasso

# Q4: What is ElasticNet?
# Combination of Ridge + Lasso

# 🔥 10️⃣ REAL-WORLD (Your Job Portal Idea)

# Suppose features:

# Experience
# Projects
# Certifications
# CGPA
# Github stars
# Random column

# Lasso might output:

# Experience = 2.3
# Projects = 1.5
# Certifications = 0
# CGPA = 0
# Github stars = 1.2

# See?

# It removed useless features automatically.

# 🟢 11️⃣ FINAL SUMMARY

# Linear Regression → No control

# Ridge → Control weight size

# Lasso → Control weight size + Remove useless features