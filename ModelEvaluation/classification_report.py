from sklearn.metrics import classification_report

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

print(classification_report(y_true, y_pred))

#               precision    recall  f1-score   support

#            0       0.67      1.00      0.80         2
#            1       1.00      0.67      0.80         3

#     accuracy                           0.80         5
#    macro avg       0.83      0.83      0.80         5
# weighted avg       0.87      0.80      0.80         5

# 1️⃣ What is classification_report?

# It gives a complete summary of classification metrics in one table:

# ✔ Precision
# ✔ Recall
# ✔ F1-score
# ✔ Support

# Instead of calculating each metric separately

# 3️⃣ What Each Column Means
# 🔹 Precision

# Correct positive predictions.

# 🔹 Recall

# How many actual positives were detected.

# 🔹 F1-score

# Balance between precision & recall.

# 🔹 Support

# Number of actual samples in that class.

# 🟣 4️⃣ Why It Is VERY Useful

# Instead of writing:

# accuracy_score

# precision_score

# recall_score

# f1_score

# You can get everything in one function.

# Very common in projects and interviews 🔥

# 🔴 5️⃣ Important for Imbalanced Data

# In Fraud Detection:

# Class 0 → Genuine (large)

# Class 1 → Fraud (small)

# Accuracy may look high
# But classification_report shows:

# Low recall for fraud

# Low precision for fraud

# Companies like Visa and PayPal analyze per-class performance carefully.

# 🟤 6️⃣ Macro vs Weighted Average (Interview Question)
# 🔹 Macro Avg

# Average of metrics for all classes equally.

# Good when classes are balanced.

# 🔹 Weighted Avg

# Average weighted by support (number of samples).

# Better for imbalanced datasets.

# Very important interview question 🔥

# 🟠 7️⃣ Multiclass Example

# If 3 classes:

# classification_report will show metrics for:

# Class 0

# Class 1

# Class 2

# And overall averages.

# 🧠 8️⃣ Interview Questions

# Q1: What does support mean?
# → Number of true samples for each class.

# Q2: Why is classification_report better than accuracy?
# → It shows per-class performance.

# Q3: What is macro average?
# → Simple average of all classes.

# Q4: What is weighted average?
# → Average considering class imbalance.

# 🔥 Final Understanding

# classification_report = Complete evaluation summary of classification model.

# It is built using:

# ✔ Confusion Matrix
# ✔ Precision
# ✔ Recall
# ✔ F1-score