import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 1️⃣ VERY EASY EXPLANATION

# Imagine this:

# You move to a new area.

# You want to know:

# Is this area rich or middle class?

# You ask your 5 nearest neighbors.

# If most are rich → You say rich
# If most are middle class → You say middle class

# That is exactly how KNN works.

# KNN = Look at nearest K neighbors → Take majority vote.

# 🟡 2️⃣ HOW IT WORKS

# Step 1: Choose value of K (like 3 or 5)

# Step 2: Calculate distance between new point and all training points

# Step 3: Pick K closest points

# Step 4: Majority vote decides class

# For regression → average
# For classification → majority vote

# 🔵 3️⃣ HOW DISTANCE IS CALCULATED?

# Most common:

# 👉 Euclidean Distance

# Formula:

# d = √((x2 - x1)² + (y2 - y1)²)

# Other distances:

# Manhattan distance
# Minkowski distance

# 5️⃣ IMPORTANT PARAMETERS

# n_neighbors → Value of K

# If K small → High variance (overfitting)
# If K large → High bias (underfitting)

# weights:

# uniform → equal vote
# distance → closer neighbors have more importance

# ⚫ 6️⃣ VERY IMPORTANT: SCALING REQUIRED

# Yes ✔

# Because KNN uses distance.

# If one feature has large values, it dominates.

# Always use:

# StandardScaler or MinMaxScaler before KNN.

# 🟠 7️⃣ DECISION BOUNDARY

# KNN can create:

# Very flexible boundaries.

# Unlike Logistic Regression (linear boundary),
# KNN can create curved boundaries.

# 🟤 8️⃣ ADVANTAGES

# ✔ Simple to understand
# ✔ No training phase (lazy learner)
# ✔ Works well for small datasets
# ✔ Non-linear

# 🔵 9️⃣ DISADVANTAGES

# ❌ Slow prediction for large data
# ❌ Sensitive to scaling
# ❌ Sensitive to noise
# ❌ Memory expensive

# 🧠 10️⃣ INTERVIEW QUESTIONS

# Q1: Why is KNN called lazy learner?
# Because it does not train model. It stores data and calculates during prediction.

# Q2: What happens if K = 1?
# Model overfits.

# Q3: What happens if K is very large?
# Model underfits.

# Q4: Does KNN need scaling?
# Yes, very important.

# Q5: How to choose best K?
# Use cross-validation.

# 🟢 11️⃣ REAL PROJECT CONNECTION (Your Job Portal)

# Suppose candidate features:

# Experience
# Skill Score
# Projects

# New candidate comes.

# KNN finds 5 similar candidates.

# If most were selected → Predict selected.

# Very intuitive approach.

# 🔥 12️⃣ SUMMARY

# KNN = Look at nearest neighbors

# Classification → Majority vote
# Regression → Average

# Needs scaling
# Works well for small data
# No actual training