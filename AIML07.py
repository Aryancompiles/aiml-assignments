import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 60)
print("ASSIGNMENT 7: SVM FOR REGRESSION (SVR)")
print("=" * 60)

# Load dataset
diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

# Select variables
X = df[["bmi"]]
y = df["target"]

print("\n--- Variables ---")
print("Independent Variable : bmi")
print("Dependent Variable   : target")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Compare C values
c_values = [0.1, 1, 10, 100]
svr_scores = []

print("\n--- C Parameter Comparison ---")

for c in c_values:
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf", C=c, epsilon=0.1))
    ])

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    svr_scores.append(r2)

    print(f"C={c}: RMSE={rmse:.4f}, R²={r2:.4f}")

# Select C with highest R²
best_index = svr_scores.index(max(svr_scores))
best_c = c_values[best_index]

print("\n--- Selected SVR Parameters ---")
print(f"Kernel  : RBF")
print(f"C       : {best_c}")
print("Epsilon : 0.1")

# Final SVR model
final_model = Pipeline([
    ("scaler", StandardScaler()),
    ("svr", SVR(
        kernel="rbf",
        C=best_c,
        epsilon=0.1
    ))
])

final_model.fit(X_train, y_train)

# Predictions
y_pred = final_model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- Final SVR Results ---")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# Actual vs predicted
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
plt.title("SVR: Actual vs Predicted Values")
plt.grid(True)
plt.tight_layout()

plt.savefig("svr_actual_vs_predicted.png", dpi=300)
plt.show()

# C comparison graph
plt.figure(figsize=(8, 5))

plt.plot(
    c_values,
    svr_scores,
    marker="o"
)

plt.xscale("log")
plt.xlabel("C Value")
plt.ylabel("R² Score")
plt.title("Effect of C on SVR Performance")
plt.grid(True)
plt.tight_layout()

plt.savefig("svr_c_comparison.png", dpi=300)
plt.show()

print("\n--- Execution Completed Successfully ---")
