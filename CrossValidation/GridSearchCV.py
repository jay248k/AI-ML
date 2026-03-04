from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

data = load_iris()
X = data.data
y = data.target

model = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, None]
}

grid = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X, y)

print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# 1️⃣ What is GridSearchCV?

# GridSearchCV is used for:

# 👉 Hyperparameter Tuning

# It finds the best combination of parameters
# by trying all possible combinations.

# 🟡 2️⃣ What is Hyperparameter?

# Hyperparameters are:

# Settings you choose before training.

# Example (Random Forest):

# n_estimators

# max_depth

# min_samples_split

# These are NOT learned from data.

# They must be tuned.

# 🔵 3️⃣ Why Do We Need It?

# Bad parameters → Bad model ❌
# Good parameters → Better accuracy ✅

# Instead of guessing manually:

# We use GridSearchCV.

# 🟣 4️⃣ How It Works

# Suppose:

# n_estimators = [50, 100]
# max_depth = [3, 5]

# GridSearch tries:

# 50, 3

# 50, 5

# 100, 3

# 100, 5

# Trains model for each combination
# Uses cross-validation
# Selects best one.

# 6️⃣ Important Parameters
# GridSearchCV(
#     estimator=model,
#     param_grid=param_grid,
#     cv=5,
#     scoring='accuracy',
#     n_jobs=-1
# )
# estimator

# Model

# param_grid

# Dictionary of parameters

# cv

# Cross validation folds

# scoring

# Metric to evaluate

# n_jobs=-1

# Uses all CPU cores (faster)

# VERY IMPORTANT interview point 🔥

# 🟠 7️⃣ What It Returns

# After fitting:

# grid.best_params_
# grid.best_score_
# grid.best_estimator_

# You can directly use:

# best_model = grid.best_estimator_
# 🧠 8️⃣ Interview Questions

# Q1: What is GridSearchCV?
# → Hyperparameter tuning using exhaustive search + cross-validation.

# Q2: Why CV inside GridSearch?
# → To get reliable performance.

# Q3: Disadvantage?
# → Slow for large parameter space.

# Q4: Alternative?
# → RandomizedSearchCV.

# 🔥 9️⃣ When It Becomes Slow?

# If:

# 5 parameters
# Each has 5 values

# Total combinations:

# 5⁵ = 3125 models 😅

# Very slow.

# Then use:

# 👉 RandomizedSearchCV

# 🚀 10️⃣ Real-World Use

# In real projects:

# Train baseline model

# Use GridSearchCV

# Get best parameters

# Evaluate final model

# Professional ML workflow 💪