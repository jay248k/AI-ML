# Machines understand only numbers.

# They do NOT understand words like:

# "Male"

# "Female"

# "High"

# "Low"

# So we convert words → numbers.

# Example:

# Gender:

# Male → 0
# Female → 1

# That conversion is done by LabelEncoder.

import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = {
    "Name": ["Jay", "Amit", "Ravi", "Neha"],
    "Gender": ["Male", "Male", "Male", "Female"]
}

df = pd.DataFrame(data)

le = LabelEncoder()

df["Gender"] = le.fit_transform(df["Gender"])

print(df)