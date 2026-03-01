from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

precision = precision_score(y_true, y_pred)
print("Precision:", precision)

# 1️⃣ Simple Meaning

# Precision answers:

# 👉 Out of all predicted Positive cases, how many were actually correct?

# Formula:
# Precision = TP / (TP + FP)

# Where:

# TP → True Positive

# FP → False Positive

# 🟡 2️⃣ Easy Example

# Imagine:

# Model predicted 20 transactions as Fraud.

# But actually:

# 15 were real fraud ✅

# 5 were normal transactions ❌

# Precision = 15 / (15 + 5) = 0.75 (75%)

# Meaning:

# When model says “Fraud”,
# 75% of the time it is correct.

# 4️⃣ When Precision is Important?

# Precision is important when:

# ❗ False Positives are costly.

# Example:

# 💳 Fraud Detection

# If system wrongly blocks genuine customer:

# Customer gets angry 😡
# Bad experience

# Companies like PayPal and Mastercard care about this.

# High precision = fewer false alarms.

# 🔴 5️⃣ Precision vs Recall (Important Difference)

# +-----------+-----------------------+
# | Metric    | Focus                 |
# +-----------+-----------------------+
# | Precision | Avoid False Positives |
# | Recall    | Avoid False Negatives |
# +-----------+-----------------------+

# Example:

# Medical Test:

# Precision → When test says cancer, how often correct?

# Recall → How many actual cancer cases detected?

# 🟤 6️⃣ Imbalanced Dataset Case

# In fraud detection:

# If precision is high → Model is careful before labeling fraud.

# If recall is high → Model catches most fraud.

# Usually we balance both using:

# 👉 F1 Score

# 7️⃣ Multiclass Precision

precision_score(y_true, y_pred, average='macro')

# Types of averaging:

# 'micro'

# 'macro'

# 'weighted'

# Very common interview question 🔥

# 🧠 8️⃣ Interview Questions

# Q1: What does precision measure?
# → Correct positive predictions.

# Q2: Formula?
# → TP / (TP + FP)

# Q3: When is precision more important than recall?
# → When False Positives are costly.

# Q4: Precision in fraud detection means?
# → When model says fraud, how often correct.

# 🔥 Final Understanding

# Precision = Model honesty when predicting Positive.

# High precision = Fewer false alarms.
# Low precision = Many wrong positive predictions.