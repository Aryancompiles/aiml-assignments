import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

print("=" * 60)
print("ASSIGNMENT 2: K-MEANS CLUSTERING")
print("=" * 60)

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())

# Select features
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n--- Selected Features ---")
print(features)

# Evaluate different K values
inertia = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

print("\n--- K Evaluation ---")
for k, i, s in zip(k_values, inertia, silhouette_scores):
    print(f"K={k}: Inertia={i:.4f}, Silhouette Score={s:.4f}")

# Plot Elbow Method
plt.figure(figsize=(8, 5))
plt.plot(k_values, inertia, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal K")
plt.grid(True)
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=300)
plt.show()

# Plot Silhouette Scores
plt.figure(figsize=(8, 5))
plt.plot(k_values, silhouette_scores, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different K Values")
plt.grid(True)
plt.tight_layout()
plt.savefig("silhouette_plot.png", dpi=300)
plt.show()

# Final K-Means model
optimal_k = 5

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

final_silhouette = silhouette_score(X_scaled, df["Cluster"])

print("\n--- Final K-Means Results ---")
print(f"Selected K         : {optimal_k}")
print(f"Silhouette Score   : {final_silhouette:.4f}")

print("\n--- Cluster Sizes ---")
print(df["Cluster"].value_counts().sort_index())

print("\n--- Cluster Means ---")
print(
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)

# Cluster visualization
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="deep",
    s=80
)

centers_original = scaler.inverse_transform(kmeans.cluster_centers_)

plt.scatter(
    centers_original[:, 0],
    centers_original[:, 1],
    marker="X",
    s=200,
    label="Centroids"
)

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()
plt.savefig("cluster_plot.png", dpi=300)
plt.show()

print("\n--- Execution Completed Successfully ---")