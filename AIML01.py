import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# Assignment 1: Association Between Variables
# --------------------------------------------------

print("=" * 55)
print("ASSIGNMENT 1: ASSOCIATION BETWEEN VARIABLES")
print("=" * 55)

# Load public dataset
df = sns.load_dataset("tips")

# Dataset information
print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Dataset Information ---")
print(df.info())

# Select independent and dependent variables
X = df["total_bill"]
Y = df["tip"]

# Calculate Pearson correlation
correlation = X.corr(Y)

print("\n--- Variable Association ---")
print("Independent Variable : total_bill")
print("Dependent Variable   : tip")
print(f"Pearson Correlation  : {correlation:.4f}")

if correlation > 0:
    print("Association: Positive")
elif correlation < 0:
    print("Association: Negative")
else:
    print("Association: No linear association")

# --------------------------------------------------
# Scatter Plot with Regression Line
# --------------------------------------------------

plt.figure(figsize=(8, 5))

sns.regplot(
    x="total_bill",
    y="tip",
    data=df,
    scatter_kws={"alpha": 0.6}
)

plt.title("Association Between Total Bill and Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.tight_layout()

plt.savefig("association_scatter_plot.png", dpi=300)
plt.show()

# --------------------------------------------------
# Correlation Heatmap
# --------------------------------------------------

numeric_data = df.select_dtypes(include=np.number)

plt.figure(figsize=(7, 5))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")
plt.tight_layout()

plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

print("\n--- Execution Completed Successfully ---")