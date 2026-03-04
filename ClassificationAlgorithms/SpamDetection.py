import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Sample dataset
data = {
    "Message": [
        "Win money now",
        "Hello how are you",
        "Claim your free prize",
        "Meeting tomorrow at office"
    ],
    "Label": [1, 0, 1, 0]  # 1 = Spam, 0 = Not Spam
}

df = pd.DataFrame(data)

X = df["Message"]
y = df["Label"]

# Convert text to numbers
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42
)

model = MultinomialNB()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))

# 📩 Spam Detection using Machine Learning

# This is one of the most classic ML projects.

# Very good for resume + interviews.

# 🟢 1️⃣ VERY EASY EXPLANATION

# Goal:

# Given an email or message → Predict:

# Spam ❌
# Not Spam ✅

# Example:

# "Win money now!!! Click here!!!" → Spam
# "Meeting at 5 PM tomorrow" → Not Spam

# Model learns from words.

# 🟡 2️⃣ HOW SPAM DETECTION WORKS (FULL PIPELINE)

# Step 1 → Collect dataset
# Step 2 → Clean text
# Step 3 → Convert text into numbers
# Step 4 → Train classifier
# Step 5 → Predict

# 🔵 3️⃣ TEXT PREPROCESSING

# Before training, we clean text:

# ✔ Lowercase
# ✔ Remove punctuation
# ✔ Remove stopwords (is, the, at…)
# ✔ Tokenization (split words)

# Example:

# "WIN Money Now!!!"

# Becomes:

# ["win", "money", "now"]

# 🟣 4️⃣ CONVERT TEXT TO NUMBERS

# ML models cannot understand text.

# So we use:

# 👉 Bag of Words (CountVectorizer)
# 👉 TF-IDF Vectorizer

# Example:

# Messages:

# "win money"

# "hello friend"

# Vocabulary:

# win, money, hello, friend

# Converted to:

# +--------------+-----+-------+-------+--------+
# | Message      | win | money | hello | friend |
# +--------------+-----+-------+-------+--------+
# | win money    | 1   | 1     | 0     | 0      |
# | hello friend | 0   | 0     | 1     | 1      |
# +--------------+-----+-------+-------+--------+

# 🟤 5️⃣ BEST ALGORITHMS FOR SPAM DETECTION

# ✔ Naive Bayes (Most popular)
# ✔ Logistic Regression
# ✔ SVM
# ✔ Random Forest

# Most common in interviews:

# 👉 Multinomial Naive Bayes

# 7️⃣ WHY NAIVE BAYES IS GOOD FOR SPAM?

# Because:

# ✔ Fast
# ✔ Works well with word frequency
# ✔ Handles high-dimensional data
# ✔ Good for text classification

# ⚫ 8️⃣ EVALUATION METRICS (VERY IMPORTANT)

# In spam detection:

# Accuracy is NOT enough.

# Better metrics:

# Precision
# Recall
# F1 Score

# Why?

# If spam predicted as not spam → dangerous.

# So recall for spam is important.

# 🟠 9️⃣ REAL-WORLD APPLICATIONS

# ✔ Gmail spam filter
# ✔ SMS spam detection
# ✔ YouTube comment filtering
# ✔ Fraud detection
# ✔ Resume keyword filtering

# 🧠 10️⃣ INTERVIEW QUESTIONS

# Q1: Which algorithm is best for spam detection?
# Multinomial Naive Bayes.

# Q2: Why not use Linear Regression?
# Because it is classification problem.

# Q3: What vectorization techniques are used?
# CountVectorizer, TF-IDF.

# Q4: Why is recall important in spam detection?
# Because missing spam is risky.

# Q5: What is TF-IDF?
# Term Frequency - Inverse Document Frequency.

# 🟣 11️⃣ ADVANCED VERSION

# Modern spam detection may use:

# ✔ Logistic Regression
# ✔ SVM
# ✔ XGBoost
# ✔ Deep Learning (LSTM, BERT)

# But Naive Bayes is still strong baseline.

# 🔥 12️⃣ PROJECT IDEA FOR YOU

# You can build:

# Spam Detection Web App using:

# Frontend → React
# Backend → Flask / FastAPI
# Model → MultinomialNB
# Database → MongoDB

# That will look strong on resume 🔥

# 🚀 13️⃣ SUMMARY

# Spam Detection = Text Classification

# Pipeline:

# Text → Clean → Vectorize → Train → Predict

# Best beginner algorithm → Naive Bayes