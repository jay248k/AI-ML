import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Example dataset
data = {
    "Amount": [100, 200, 50000, 150, 300000, 250],
    "Fraud":  [0, 0, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

X = df[["Amount"]]
y = df["Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(classification_report(y_test, predictions))