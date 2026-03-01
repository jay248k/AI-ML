# Machines understand only numbers.

# They do NOT understand words like:

# "Male"

# "Female"

# "High"

# "Low"

# So we convert words → numbers.

# Example:

# Gender:

# Male → 0
# Female → 1

# That conversion is done by LabelEncoder.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "Name": ["Jay", "Amit", "Ravi", "Neha"],
    "Gender": ["Male", "Male", "Male", "Female"]
}

df = pd.DataFrame(data)

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print(df)


# 1️⃣ VERY EASY EXPLANATION

# Machines understand only numbers.

# They do NOT understand words like:

# "Male"

# "Female"

# "High"

# "Low"

# So we convert words → numbers.

# Example:

# Gender:

# Male → 0
# Female → 1

# That conversion is done by LabelEncoder.

# Output:

# Gender column becomes:

# Male → 1
# Female → 0

# (Numbers may vary depending on alphabetical order)

# 🔵 3️⃣ WHAT HAPPENS INTERNALLY?

# LabelEncoder:

# Finds unique values

# Sorts them alphabetically

# Assigns numbers starting from 0

# Example:

# ["Apple", "Banana", "Mango"]

# Alphabetical order:
# Apple → 0
# Banana → 1
# Mango → 2

# 🔴 4️⃣ IMPORTANT RULE (Very Important ⚠)

# 👉 LabelEncoder should be used mainly for target column (y).

# Example:

# Spam detection:

# Spam → 1
# Not Spam → 0

# Correct usage:

# y = le.fit_transform(y)
# ⚠ 5️⃣ WHEN NOT TO USE LabelEncoder

# ❌ Do NOT use LabelEncoder for input features if categories have NO order.

# Example:

# Color:

# Red
# Blue
# Green

# If you encode:

# Blue → 0
# Green → 1
# Red → 2

# Model may think:

# Red > Green > Blue

# But colors have NO mathematical relationship.

# This creates false ordering.

# Instead use:

# 👉 OneHotEncoder

# 🟣 6️⃣ INTERVIEW QUESTIONS

# Q1: What is LabelEncoder?
# Answer:
# It converts categorical labels into numeric values.

# Q2: Why is it dangerous for input features?
# Answer:
# Because it introduces artificial ordinal relationship.

# Q3: Difference between LabelEncoder and OneHotEncoder?

# LabelEncoder:

# Single column

# Assigns numbers

# Introduces order

# OneHotEncoder:

# Creates multiple binary columns

# No order introduced

# Q4: When is LabelEncoder safe?
# Answer:
# When encoding target variable.

# 🟤 7️⃣ REAL-WORLD EXAMPLE (Your Resume Ranking)

# Suppose:

# Resume result:

# Selected
# Rejected

# We can encode:

# Selected → 1
# Rejected → 0

# Perfect use case.

# But if:

# Department:

# HR
# Finance
# Tech

# Using LabelEncoder may confuse model.

# Better to use OneHotEncoder.

# 🧠 ADVANCED CONCEPT

# LabelEncoder does NOT handle unseen categories well.

# If test data has new category:

# It throws error.

# That’s why in production we use:

# OneHotEncoder(handle_unknown='ignore')

# Or custom encoders

# 🟢 SIMPLE SUMMARY

# LabelEncoder:
# ✔ Converts text → numbers
# ✔ Good for target column
# ❌ Not safe for unordered input features