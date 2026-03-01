from sklearn.metrics import f1_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

f1 = f1_score(y_true, y_pred)
print("F1 Score:", f1)

# 1️⃣ What is F1 Score?

# F1 Score is the balance between Precision and Recall.

# Formula:

# F1=2 * (Precision * Recall) / (Precision + Recall)

# 2️⃣ Simple Meaning

# If:

# Precision = 0.80
# Recall = 0.60

# F1 = 2 * (0.80 * 0.60) / (0.80 + 0.60) = 0.69

# So F1 = 0.68

# 👉 F1 becomes high only if BOTH precision and recall are high.

# 🔵 3️⃣ Why Not Just Accuracy?

# In Fraud Detection:

# Suppose:

# Accuracy = 99%
# Recall = 10%

# That means:

# Model misses most fraud ❌

# Companies like Visa and PayPal prefer F1-score over accuracy in imbalanced datasets.

# 🔴 4️⃣ When F1 Score is Important?

# F1 is important when:

# ✔ Dataset is imbalanced
# ✔ Both false positives and false negatives matter
# ✔ You want balance between precision and recall

# Common in:

# Fraud Detection

# Spam Detection

# Medical Diagnosis

# 6️⃣ Why Harmonic Mean?

# F1 uses harmonic mean (not normal average) because:

# If one value is very low → F1 becomes low.

# Example:

# Precision = 1.0
# Recall = 0.0

# F1 = 0 ❌

# So model must perform well on both.

# 🟠 7️⃣ Multiclass F1
# f1_score(y_true, y_pred, average='macro')

# Types:

# 'micro'

# 'macro'

# 'weighted'

# Interviewers LOVE this question 🔥

# 🧠 8️⃣ Interview Questions

# Q1: What is F1 Score?
# → Harmonic mean of precision and recall.

# Q2: Why use F1 instead of accuracy?
# → Because accuracy fails in imbalanced datasets.

# Q3: When is F1 high?
# → When both precision and recall are high.

# Q4: Can F1 be 1?
# → Yes, if precision = 1 and recall = 1

# Final Understanding

# +-----------+--------------------------------------+
# | Metric    | Measures                             |
# +-----------+--------------------------------------+
# | Accuracy  | Overall correctness                  |
# | Precision | Correctness of positive predictions  |
# | Recall    | Ability to catch positives           |
# | F1        | Balance between precision & recall   |
# +-----------+--------------------------------------+