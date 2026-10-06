import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("=" * 65)
print("ASSIGNMENT 8: REGULARIZATION TO AVOID OVERFITTING")
print("=" * 65)

# ---------------------------------------------------------
# Create synthetic nonlinear dataset
# ---------------------------------------------------------
np.random.seed(42)

x = np.linspace(-3, 3, 120)

y = (
    2 * x**2
    + 3 * x
    + 5
    + np.random.normal(0, 2.5, size=len(x))
)

X = x.reshape(-1, 1)

df = pd.DataFrame({
    "x": x,
    "y": y
})

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

print("\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------
models = {
    "Linear Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("model", LinearRegression())
    ]),

    "Ridge Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Lasso Regression": Pipeline([
        ("poly", PolynomialFeatures(degree=10)),
        ("scaler", StandardScaler()),
        ("model", Lasso(alpha=0.01, max_iter=100000))
    ])
}

results = []

print("\n--- Model Comparison ---")

for name, model in models.items():

    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    test_mae = mean_absolute_error(y_test, test_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    test_rmse = test_mse ** 0.5

    # Coefficients
    coefficients = model.named_steps["model"].coef_
    coefficient_norm = np.linalg.norm(coefficients)

    nonzero_coefficients = np.sum(
        np.abs(coefficients) > 1e-6
    )

    results.append([
        name,
        train_r2,
        test_r2,
        test_mae,
        test_mse,
        test_rmse,
        coefficient_norm,
        nonzero_coefficients
    ])

    print(f"\n{name}")
    print(f"Training R²       : {train_r2:.4f}")
    print(f"Testing R²        : {test_r2:.4f}")
    print(f"Test MAE          : {test_mae:.4f}")
    print(f"Test MSE          : {test_mse:.4f}")
    print(f"Test RMSE         : {test_rmse:.4f}")
    print(f"Coefficient Norm  : {coefficient_norm:.4f}")
    print(f"Non-zero Coeffs   : {nonzero_coefficients}")

# ---------------------------------------------------------
# Results table
# ---------------------------------------------------------
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train R2",
        "Test R2",
        "Test MAE",
        "Test MSE",
        "Test RMSE",
        "Coefficient Norm",
        "Non-zero Coefficients"
    ]
)

print("\n--- Results Summary ---")
print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------
# Plot model predictions
# ---------------------------------------------------------
X_plot = np.linspace(-3, 3, 300).reshape(-1, 1)

plt.figure(figsize=(9, 6))

plt.scatter(
    X,
    y,
    s=35,
    alpha=0.6,
    label="Actual Data"
)

for name, model in models.items():
    model.fit(X_train, y_train)
    y_plot = model.predict(X_plot)

    plt.plot(
        X_plot,
        y_plot,
        linewidth=2,
        label=name
    )

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Regularization: Model Comparison")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "regularization_comparison.png",
    dpi=300
)

plt.show()

print("\n--- Execution Completed Successfully ---")