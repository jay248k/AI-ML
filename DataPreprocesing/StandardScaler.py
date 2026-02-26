#

import pandas as pd
from sklearn.preprocessing import StandardScaler

data = {
    "Hours": [1, 2, 3, 4, 5],
    "Marks": [35, 40, 50, 55, 60]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

scaler = StandardScaler()

scaled_data = scaler.fit_transform(df)

print("Scaled Data:")
print(scaled_data)