import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

print("=" * 60)
print("ASSIGNMENT 3: DIMENSIONALITY REDUCTION USING PCA")
print("=" * 60)

# Load Wine dataset
wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target

print("\n--- Dataset Preview ---")
print(X.head())

print("\n--- Dataset Shape ---")
print(X.shape)

print("\n--- Number of Features ---")
print(X.shape[1])

print("\n--- Number of Classes ---")
print(len(np.unique(y)))

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Standardization Completed ---")

# PCA on all features to calculate explained variance
pca_full = PCA()
X_pca_full = pca_full.fit_transform(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n--- Explained Variance Ratio ---")
for i, value in enumerate(explained_variance, start=1):
    print(f"PC{i}: {value:.4f} ({value * 100:.2f}%)")

print("\n--- Cumulative Explained Variance ---")
for i, value in enumerate(cumulative_variance, start=1):
    print(f"PC1-PC{i}: {value:.4f} ({value * 100:.2f}%)")

# PCA to 2 dimensions
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

print("\n--- PCA Reduction ---")
print(f"Original Dimensions : {X.shape[1]}")
print(f"Reduced Dimensions  : {X_pca_2d.shape[1]}")

print("\n--- Variance Retained by 2 Components ---")
print(
    f"PC1: {pca_2d.explained_variance_ratio_[0] * 100:.2f}%"
)
print(
    f"PC2: {pca_2d.explained_variance_ratio_[1] * 100:.2f}%"
)
print(
    f"Total: {pca_2d.explained_variance_ratio_.sum() * 100:.2f}%"
)

# Cumulative variance plot
plt.figure(figsize=(8, 5))
plt.plot(
    range(1, len(cumulative_variance) + 1),
    cumulative_variance,
    marker="o"
)
plt.axhline(y=0.90, linestyle="--", label="90% variance")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance using PCA")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("cumulative_variance.png", dpi=300)
plt.show()

# 2-D PCA scatter plot
plt.figure(figsize=(8, 6))

for class_label in np.unique(y):
    plt.scatter(
        X_pca_2d[y == class_label, 0],
        X_pca_2d[y == class_label, 1],
        label=f"Class {class_label}",
        s=60
    )

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("Wine Dataset after PCA")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("pca_2d_scatter.png", dpi=300)
plt.show()

print("\n--- Execution Completed Successfully ---")