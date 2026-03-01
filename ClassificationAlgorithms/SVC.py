import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Pass":  [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Pass"]

# Scaling is IMPORTANT for SVM
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

model = SVC(kernel="linear")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# SVC (Support Vector Classifier)

# SVC is the classification version of Support Vector Machine (SVM).

# This is very important for interviews.

# 🟢 1️⃣ VERY EASY EXPLANATION

# Imagine two types of students:

# Pass 🔵
# Fail 🔴

# We want to draw a line that separates them.

# Many lines can separate them…

# But SVM chooses:

# 👉 The BEST line
# 👉 With maximum margin

# Margin = Distance between line and nearest points.

# So:

# SVM = Find the widest possible separation line.

# 🟡 2️⃣ WHAT IS MARGIN?

# Imagine:

# 🔴 🔴 🔴 | 🔵 🔵 🔵

# That vertical line separates classes.

# Margin = Distance between line and closest red & blue points.

# SVM tries to:

# Maximize that margin.

# Because:

# Bigger margin → Better generalization → Less overfitting.

# 🔵 3️⃣ WHAT ARE SUPPORT VECTORS?

# The closest points to boundary are called:

# 👉 Support Vectors

# These points decide the boundary.

# If you remove other points, boundary does not change much.

# Very important concept 🔥

# 🟣 4️⃣ MATHEMATICAL IDEA (Simplified)

# Decision boundary:

# w.x + b = 0

# SVM tries to:

# Minimize:

# 1/2||w||₂

# Subject to correct classification constraints.

# Don’t worry too much about math now.

# Concept is more important.

# 🟤 5️⃣ WHAT IF DATA IS NOT LINEAR?

# Example:

# Points arranged in circle shape.

# We cannot separate with straight line.

# Solution:

# 👉 Kernel Trick

# Kernel transforms data into higher dimension.

# Common kernels:

# linear
# poly (Polynomial)
# rbf (Radial Basis Function) ← Most popular
# sigmoid

# 7️⃣ IMPORTANT PARAMETERS

# kernel → linear, rbf, poly

# C → Regularization parameter

# Small C → Large margin, more tolerant to mistakes
# Large C → Smaller margin, less tolerant

# gamma (for rbf kernel)

# Controls influence of single training point.

# ⚫ 8️⃣ VERY IMPORTANT: SCALING REQUIRED

# YES ✔

# SVM is distance-based.

# Always scale data before SVM.

# 🟠 9️⃣ LOGISTIC REGRESSION vs SVM

# +----------------------+-----------------+
# | Logistic Regression  | SVM             |
# +----------------------+-----------------+
# | Probability-based    | Margin-based    |
# | Uses sigmoid         | Uses max margin |
# | Faster               | Slower          |
# | Linear boundary      | Can use kernel  |
# +----------------------+-----------------+

# 10️⃣ INTERVIEW QUESTIONS

# Q1: What is Support Vector?
# Closest data points to decision boundary.

# Q2: What does C parameter do?
# Controls tradeoff between margin size and classification error.

# Q3: What is kernel trick?
# Mapping data into higher dimension.

# Q4: Does SVM need scaling?
# Yes, very important.

# Q5: When to use SVM?
# Medium-sized datasets, clear separation.

# 🟢 11️⃣ WHEN TO USE SVM?

# Use when:

# ✔ Data is high dimensional
# ✔ Clear margin of separation
# ✔ Dataset is medium size
# ✔ Need strong classifier

# Avoid when:

# ❌ Very large dataset (slow)
# ❌ Many noise points

# 🟣 12️⃣ REAL PROJECT (Your Job Portal)

# Predict:

# Selected / Not Selected

# Features:

# Experience
# Skill Score
# Projects
# CGPA

# If classes are well separated → SVM works very well.

# 🔥 13️⃣ SUMMARY

# SVM:

# ✔ Finds best separating boundary
# ✔ Maximizes margin
# ✔ Uses support vectors
# ✔ Uses kernels for non-linear data
# ✔ Needs scaling