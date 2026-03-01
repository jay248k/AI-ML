import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

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

model = GaussianNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Predictions:", predictions)

# 📘 Naive Bayes

# This algorithm is:

# ✔ Simple
# ✔ Fast
# ✔ Powerful for text data
# ✔ Very common in interviews

# 🟢 1️⃣ VERY EASY EXPLANATION

# Imagine:

# You receive an email.

# You check words like:

# "Free"
# "Win"
# "Offer"

# If many spam words appear → Email is Spam.

# Naive Bayes works exactly like this.

# It calculates probability and chooses the class with highest probability.

# 🟡 2️⃣ WHY CALLED "NAIVE"?

# Because it assumes:

# 👉 All features are independent of each other.

# Example:

# In job selection:

# Experience
# Skill Score
# Projects

# Naive Bayes assumes:

# These features do not depend on each other.

# In real life, they may depend.

# But model still works surprisingly well.

# That’s why it’s called “Naive”.

# 🔵 3️⃣ BAYES THEOREM (Core Formula)

# P(A|B) = P(B|A) * P(A) / P(B)

# In ML terms:

# P(Class|Features) = P(Features|Class) * P(Class) / P(Features)

# Model predicts:

# Class with highest probability.

# 🟣 4️⃣ TYPES OF NAIVE BAYES

# 1️⃣ Gaussian Naive Bayes
# → For continuous data

# 2️⃣ Multinomial Naive Bayes
# → For text classification (most common)

# 3️⃣ Bernoulli Naive Bayes
# → For binary features (0/1)

# 6️⃣ DOES IT NEED SCALING?

# Usually ❌ No

# Naive Bayes is probability-based.

# Scaling is not required in most cases.

# ⚫ 7️⃣ ADVANTAGES

# ✔ Very fast
# ✔ Works well with small datasets
# ✔ Excellent for text classification
# ✔ Low computational cost

# 🟠 8️⃣ DISADVANTAGES

# ❌ Assumes independence (not realistic)
# ❌ Not good for highly correlated features
# ❌ Less flexible than tree models

# 🧠 9️⃣ INTERVIEW QUESTIONS

# Q1: Why is it called Naive?
# Because it assumes features are independent.

# Q2: What theorem is used?
# Bayes Theorem.

# Q3: Which Naive Bayes is best for text?
# Multinomial Naive Bayes.

# Q4: Does Naive Bayes overfit easily?
# No, usually low variance model.

# Q5: Does it need scaling?
# No.

# 🟢 10️⃣ REAL PROJECT CONNECTION (Your Job Portal)

# Suppose:

# You want to classify resumes as:

# Shortlisted / Not Shortlisted

# Based on keywords:

# Python
# React
# Machine Learning
# Node.js

# Naive Bayes is excellent for this.

# Very commonly used in:

# ✔ Spam detection
# ✔ Sentiment analysis
# ✔ Resume screening
# ✔ News classification

# 11️⃣ NAIVE BAYES vs SVM
# +----------------------+---------------------------------+
# | Naive Bayes          | SVM                             |
# +----------------------+---------------------------------+
# | Probability-based    | Margin-based                    |
# | Very fast            | Slower                          |
# | Works great for text | Works great for structured data |
# | Assumes independence | No such assumption              |
# +----------------------+---------------------------------+

# 12️⃣ SUMMARY

# Naive Bayes:

# ✔ Based on probability
# ✔ Uses Bayes Theorem
# ✔ Assumes independence
# ✔ Fast and simple
# ✔ Great for text data