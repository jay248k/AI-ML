# Linear Regression → uses everything
# Ridge → reduces weight but keeps everything
# Lasso → removes useless features (weight = 0)

import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

data = {
    "Hours": [1,2,3,4,5,6,7,8],
    "Marks": [35,40,50,55,60,65,75,80]
}

df = pd.DataFrame(data)

X = df[["Hours"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = Lasso(alpha=0.5)
model.fit(X_train, y_train)

print("Weight:", model.coef_)
print("Intercept:", model.intercept_)