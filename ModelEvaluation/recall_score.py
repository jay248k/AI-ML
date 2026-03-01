from sklearn.metrics import recall_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

recall = recall_score(y_true, y_pred)
print("Recall:", recall)

# 1️⃣ Simple Meaning

# Recall answers:

# 👉 Out of all actual Positive cases, how many did the model correctly detect?

# Formula:
# Recall = TP / (TP + FN)

# Where:

# TP → True Positive

# FN → False Negative

# 2️⃣ Easy Example (Fraud Detection)

# Suppose:

# There are 20 real fraud transactions.

# Model detected 15 of them.

# But missed 5.

# Recall = 15 / (15 + 5) = 0.75 (75%)

# Meaning:

# Model catches 75% of fraud cases

# 4️⃣ Why Recall is VERY Important?

# Recall is important when:

# ❗ Missing a positive case is dangerous.

# Examples:

# 💳 Fraud Detection

# If fraud is missed → Money loss

# Companies like Visa and Mastercard focus heavily on recall.

# 🏥 Medical Diagnosis

# If cancer is not detected → Life risk

# Here recall must be very high.

# 5️⃣ Precision vs Recall (Clear Difference)

# +-----------+-----------------------+
# | Metric    | Focus                 |
# +-----------+-----------------------+
# | Precision | Avoid False Positives |
# | Recall    | Avoid False Negatives |
# +-----------+-----------------------+

# Easy trick:

# Precision = How correct when model says YES

# Recall = How many YES cases it found

# 🟤 6️⃣ Imbalanced Dataset Case

# In fraud detection:

# High recall → Catch most fraud

# Low recall → Many fraud cases missed

# Accuracy can be 99%
# But recall can be 0% ❌

# This is why recall is more important than accuracy in fraud problems.

# 7️⃣ Multiclass Recall

recall_score(y_true, y_pred, average='macro')

# Average types:

# micro

# macro

# weighted

# Very common interview question 🔥

# 🧠 8️⃣ Interview Questions

# Q1: Formula of recall?
# → TP / (TP + FN)

# Q2: When is recall more important than precision?
# → When missing positive cases is costly.

# Q3: Recall in fraud detection means?
# → Out of all fraud transactions, how many detected.

# Q4: Can recall be 1?
# → Yes, if model detects all positive cases.

# 🔥 Final Summary

# Recall = Model’s ability to catch actual positives.

# High recall = Detect most real positive cases.
# Low recall = Missing important cases.