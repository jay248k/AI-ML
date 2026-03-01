from sklearn.metrics import confusion_matrix

y_true = [0, 0, 1, 1, 0, 1, 0]
y_pred = [0, 1, 1, 0, 0, 1, 0]

cm = confusion_matrix(y_true, y_pred)
print(cm)

# 1️⃣ What is Confusion Matrix?

# It is a table that shows:

# 👉 How many predictions were correct
# 👉 How many were wrong
# 👉 What type of errors happened

# It gives full understanding of model performance.

# 🟡 2️⃣ Structure (Binary Classification)
# 	Predicted 0	Predicted 1
# Actual 0	TN	FP
# Actual 1	FN	TP

# Where:

# TP → True Positive

# TN → True Negative

# FP → False Positive

# FN → False Negative

# 🔵 3️⃣ Simple Example (Fraud Detection)

# Suppose:

# 100 total transactions

# 20 are fraud

# 80 are genuine

# Model result:

# TP = 15 (fraud correctly detected)

# FN = 5 (fraud missed ❌)

# FP = 10 (normal wrongly flagged ❌)

# TN = 70 (normal correctly predicted)

# Confusion Matrix:

# [[70 10]
#  [ 5 15]]

# 5️⃣ Why Confusion Matrix is IMPORTANT?

# Because all metrics come from it:

# Accuracy

# Precision

# Recall

# F1-score

# For example:

# Precision=TP/(TP+FP) 
# Recall=TP/(TP+FN)

# 🟤 6️⃣ Real World Importance

# In Fraud Detection:

# Companies like Visa and Mastercard analyze:

# How many fraud missed (FN)

# How many customers blocked wrongly (FP)

# Because:

# FN → Money loss
# FP → Bad customer experience

# 8️⃣ Interview Questions

# Q1: What are TP, TN, FP, FN?

# Q2: Which error is more dangerous in fraud detection?
# → False Negative

# Q3: Which error is more dangerous in spam detection?
# → False Positive

# Q4: Can confusion matrix be used for multiclass?
# → Yes

# 🟢 9️⃣ Multiclass Confusion Matrix

# If 3 classes:

# Matrix becomes 3×3.

# Each row → actual class
# Each column → predicted class

# 🔥 Final Understanding

# Confusion Matrix is the BASE of classification evaluation.

# Without confusion matrix, you cannot deeply understand model performance.