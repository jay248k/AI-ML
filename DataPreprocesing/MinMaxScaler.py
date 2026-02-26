# Imagine marks are:

# 10, 20, 30, 40, 50

# We want to convert them between:

# 0 and 1

# So:

# 10 → 0
# 50 → 1
# 30 → 0.5

# That’s what MinMaxScaler does.

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [35, 40, 50, 55, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = MinMaxScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)