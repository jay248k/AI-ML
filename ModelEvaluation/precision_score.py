from sklearn.metrics import precision_score

y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

precision = precision_score(y_true, y_pred)
print("Precision:", precision)