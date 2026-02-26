import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
# Create sample student data
data = {
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Marks": [35, 40, 50, 55, 60, 65, 75, 80, 85, 95]
}

df = pd.DataFrame(data)

# print(df)
X = df[["Hours"]]   # Independent variable
y = df["Marks"]     # Target variable


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2 #, random_state=3
)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Actual Marks:", list(y_test))
print("Predicted Marks:", y_pred)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))