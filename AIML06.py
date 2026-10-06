import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("ASSIGNMENT 6: LINEAR REGRESSION")
print("=" * 60)

# Load dataset
diabetes = load_diabetes(as_frame=True)

df = diabetes.frame

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Columns ---")
print(df.columns.tolist())

# Select independent and dependent variables
X = df[["bmi"]]
y = df["target"]

print("\n--- Variables ---")
print("Independent Variable : bmi")
print("Dependent Variable   : target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model parameters
print("\n--- Linear Regression Model ---")
print(f"Intercept : {model.intercept_:.4f}")
print(f"Coefficient : {model.coef_[0]:.4f}")

# Predictions
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- Regression Results ---")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Actual vs predicted plot
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    s=60,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Target")
plt.ylabel("Predicted Target")
plt.title("Actual vs Predicted Values - Linear Regression")
plt.grid(True)
plt.tight_layout()

plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

print("\n--- Sample Predictions ---")

comparison = pd.DataFrame({
    "Actual": y_test.values[:10],
    "Predicted": y_pred[:10]
})

print(comparison.round(2))

print("\n--- Execution Completed Successfully ---")