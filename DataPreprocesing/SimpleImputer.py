# Hours	Marks
# 2	40
# 3	❌
# 4	55

# One mark is missing.

# Machine learning models cannot handle empty values.

# So we must fill them.

# That process is called Imputation.
import pandas as pd
from sklearn.impute import SimpleImputer

data = {
    "Hours": [2, 3, None, 5, 6],
    "Marks": [40, None, 50, 60, 65]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)

imputer = SimpleImputer(strategy="mean")

df_filled = imputer.fit_transform(df)

print("After Imputation:")
print(df_filled)