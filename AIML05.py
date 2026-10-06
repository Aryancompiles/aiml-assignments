import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 60)
print("ASSIGNMENT 5: KNN CLASSIFICATION")
print("=" * 60)

# Load Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

print("\n--- Dataset Preview ---")
print(X.head())

print("\n--- Dataset Shape ---")
print(X.shape)

print("\n--- Number of Classes ---")
print(len(iris.target_names))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y)

print("\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n--- Feature Scaling ---")
print("StandardScaler applied successfully.")

# Compare different K values
k_values = [1, 3, 5, 7, 9]
accuracies = []

print("\n--- K Value Comparison ---")

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)

    predictions = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    accuracies.append(accuracy)

    print(f"K={k}: Accuracy={accuracy:.4f}")

# Select best K based on highest test accuracy
best_index = accuracies.index(max(accuracies))
best_k = k_values[best_index]

print("\n--- Selected K ---")
print(f"Best K : {best_k}")
print(f"Best Accuracy : {accuracies[best_index]:.4f}")

# Final model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)

y_pred = final_knn.predict(X_test_scaled)

# Final metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

print("\n--- Final KNN Results ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=iris.target_names,
    yticklabels=iris.target_names
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title(f"KNN Confusion Matrix (K={best_k})")
plt.tight_layout()
plt.savefig("knn_confusion_matrix.png", dpi=300)
plt.show()

# Accuracy comparison plot
plt.figure(figsize=(8, 5))

plt.plot(
    k_values,
    accuracies,
    marker="o"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for Different K Values")
plt.xticks(k_values)
plt.grid(True)
plt.tight_layout()
plt.savefig("knn_k_comparison.png", dpi=300)
plt.show()

print("\n--- Execution Completed Successfully ---")