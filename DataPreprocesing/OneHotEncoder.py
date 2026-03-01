import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    "Department": ["HR", "Tech", "Finance", "Tech"]
}

df = pd.DataFrame(data)

encoder = OneHotEncoder(sparse_output=False)

encoded = encoder.fit_transform(df[["Department"]])

print(encoded)

# 1️⃣ VERY EASY EXPLANATION

# Suppose we have a column:

# Department:

# HR

# Tech

# Finance

# Machine cannot understand text.

# If we convert:

# HR → 0
# Tech → 1
# Finance → 2

# ❌ Problem:

# Model thinks:

# Finance (2) > Tech (1) > HR (0)

# But departments have NO order.

# So instead we create new columns:

# +----+------+---------+
# | HR | Tech | Finance |
# +----+------+---------+
# | 1  | 0    | 0       |
# | 0  | 1    | 0       |
# | 0  | 0    | 1       |
# +----+------+---------+

# This is called One-Hot Encoding.

# Each category gets its own column.

# 🟡 2️⃣ WHY WE NEED IT?

# Because most ML models assume numeric meaning.

# If we assign numbers directly,
# model assumes order.

# OneHotEncoder removes order problem.

# 4️⃣ WHAT HAPPENS INTERNALLY?

# Step 1 → Find unique categories
# Step 2 → Create separate column for each
# Step 3 → Put 1 where category matches

# If 5 categories → 5 new columns.

# 🟣 5️⃣ INTERVIEW LEVEL EXPLANATION

# Q: What problem does OneHotEncoder solve?

# Answer:
# It prevents artificial ordinal relationship in categorical variables.

# Q: What is dimensionality increase?

# If one column has 10 categories → it becomes 10 columns.

# This increases feature space.

# Q: What is sparse matrix?

# OneHotEncoder usually returns sparse matrix to save memory
# because most values are 0.

# 🟤 6️⃣ VERY IMPORTANT (Dummy Variable Trap)

# If you have 3 categories:

# HR, Tech, Finance

# You only need 2 columns.

# Because:

# If HR=0 and Tech=0,
# then automatically Finance=1.

# If you keep all 3 columns,
# it causes multicollinearity in linear models.

# To avoid this:

# encoder = OneHotEncoder(drop='first', sparse_output=False)

# This drops first category.

# Important for:

# Linear Regression

# Logistic Regression

# Not very important for:

# Tree models

# ⚫ 7️⃣ WHEN NOT NECESSARY?

# Tree-based models:

# DecisionTree

# RandomForest

# XGBoost

# They can handle label encoding better than linear models.

# But still OneHot is safer.

# 🧠 8️⃣ PRODUCTION LEVEL TIP

# Always handle unknown categories:

# encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# If new category appears in test data,
# model will not crash.

# Very important in real applications.

# 🟢 9️⃣ REAL EXAMPLE (Your Resume Ranking Project)

# Suppose features:

# Education:

# B.Tech

# M.Tech

# MBA

# BCA

# OneHotEncoding will create:

# B.Tech | M.Tech | MBA | BCA

# Model will treat them independently.

# No false ranking.

# 🟡 10️⃣ Difference Summary

# LabelEncoder:

# Single column

# Introduces order

# OneHotEncoder:

# Multiple columns

# No order

# Safer

