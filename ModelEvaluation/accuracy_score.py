from sklearn.metrics import accuracy_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

acc = accuracy_score(y_true, y_pred)
print("Accuracy:", acc)

# accuracy_score in Machine Learning

# accuracy_score is a function from:

# sklearn.metrics

# It calculates:

# Accuracy=Correct Predictions / Total Predictions

# 1️⃣ Simple Meaning

# If model predicted 100 samples
# and 90 are correct

# Accuracy = 90 / 100 = 0.90 (90%)

# 🟡 2️⃣ Formula

# Accuracy = (TP + TN) / (TP + TN + FP + FN)

# Where:

# TP → True Positive

# TN → True Negative

# FP → False Positive

# FN → False Negative

# 4️⃣ When Accuracy is Good?

# ✔ When dataset is balanced
# ✔ When both classes are equally important

# Example:

# Cat vs Dog classification

# Digit recognition

# 🔴 5️⃣ When Accuracy is BAD?

# Very important for interview ⚠

# In Fraud Detection:

# Suppose:

# 1000 transactions
# 990 genuine
# 10 fraud

# If model predicts:

# All transactions = genuine

# Accuracy = 990 / 1000 = 99%

# But it detected 0 fraud ❌

# This is why companies like Visa and PayPal do NOT rely only on accuracy.

# They focus more on:

# ✔ Recall
# ✔ Precision
# ✔ F1-score
# ✔ ROC-AUC

# 🟤 6️⃣ Binary vs Multiclass

# accuracy_score works for:

# ✔ Binary classification
# ✔ Multiclass classification

# Example (Multiclass):

# y_true = [0, 1, 2, 1]
# y_pred = [0, 2, 2, 1]

# It still calculates correctly predicted labels.

# 🟠 7️⃣ Interview Questions

# Q1: What is accuracy?
# → Ratio of correct predictions to total predictions.

# Q2: Why is accuracy not good for imbalanced data?
# → Because model can ignore minority class and still get high accuracy.

# Q3: What is better than accuracy in fraud detection?
# → Recall, F1-score, ROC-AUC.

# 🔥 Final Summary

# accuracy_score is:

# ✔ Simple
# ✔ Easy to understand
# ✔ Good for balanced datasets
# ❌ Misleading for imbalanced datasets