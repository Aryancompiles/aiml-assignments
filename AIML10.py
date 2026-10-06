import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from PIL import Image
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

print("=" * 70)
print("ASSIGNMENT 10: FAKE NEWS DETECTION FROM MULTIMODAL DATA")
print("=" * 70)

# ---------------------------------------------------------
# Load Fakeddit dataset
# ---------------------------------------------------------
df = pd.read_csv("multimodal_news_titles.tsv", sep="\t")

print("\n--- Dataset Preview ---")
print(df[["clean_title", "image_url", "2_way_label"]].head())

print("\n--- Dataset Shape ---")
print(df.shape)

# Keep usable rows
df = df[
    df["clean_title"].notna()
    & df["image_url"].notna()
    & df["2_way_label"].isin([0, 1])
].copy()

# Keep only posts that have images
if "hasImage" in df.columns:
    df = df[df["hasImage"] == True]

print("\n--- Usable Multimodal Rows ---")
print(len(df))

# ---------------------------------------------------------
# Select balanced small subset
# ---------------------------------------------------------
fake_df = df[df["2_way_label"] == 0].head(80)
real_df = df[df["2_way_label"] == 1].head(80)

df = pd.concat([fake_df, real_df]).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

print("\n--- Selected Balanced Dataset ---")
print(df["2_way_label"].value_counts())

# ---------------------------------------------------------
# Download images and extract visual features
# ---------------------------------------------------------
def extract_visual_features(image_url):

    try:
        response = requests.get(
            image_url,
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"}
        )

        if response.status_code != 200:
            return None

        image = Image.open(
            BytesIO(response.content)
        ).convert("RGB")

        image = image.resize((128, 128))

        arr = np.asarray(image, dtype=np.float32) / 255.0

        # EXACT 7-D visual descriptor:
        # mean R,G,B + std R,G,B + mean grayscale
        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))

        gray = (
            0.299 * arr[:, :, 0]
            + 0.587 * arr[:, :, 1]
            + 0.114 * arr[:, :, 2]
        )

        gray_mean = gray.mean()

        features = np.concatenate([
            mean_rgb,
            std_rgb,
            [gray_mean]
        ])

        return features

    except Exception:
        return None


visual_features = []
valid_rows = []

print("\n--- Extracting Image Features ---")

for index, row in df.iterrows():

    features = extract_visual_features(
        row["image_url"]
    )

    if features is not None:
        visual_features.append(features)
        valid_rows.append(index)

    if len(valid_rows) % 10 == 0:
        print(
            f"Processed: {len(df)} | "
            f"Valid images: {len(valid_rows)}"
        )

    if len(valid_rows) >= 60:
        break

df = df.loc[valid_rows].reset_index(drop=True)

visual_features = np.array(visual_features)

print("\n--- Multimodal Dataset ---")
print(f"Valid samples : {len(df)}")
print(f"Visual features per image : {visual_features.shape[1]}")

# ---------------------------------------------------------
# Text features
# ---------------------------------------------------------
text_vectorizer = TfidfVectorizer(
    max_features=120,
    stop_words="english"
)

X_text = text_vectorizer.fit_transform(
    df["clean_title"]
).toarray()

print(f"Text TF-IDF features : {X_text.shape[1]}")

# ---------------------------------------------------------
# Scale visual features
# ---------------------------------------------------------
scaler = StandardScaler()

X_image = scaler.fit_transform(
    visual_features
)

# ---------------------------------------------------------
# Feature fusion
# ---------------------------------------------------------
X = np.hstack([
    X_text,
    X_image
])

y = df["2_way_label"].astype(int)

print(f"Combined features : {X.shape[1]}")

# ---------------------------------------------------------
# Train/test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n--- Train/Test Split ---")
print(f"Training samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ---------------------------------------------------------
# Logistic Regression
# ---------------------------------------------------------
model = LogisticRegression(
    max_iter=2000,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)
recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)
f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n--- Multimodal Classification Results ---")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-Score  : {f1:.4f}")

print("\n--- Classification Report ---")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Fake", "Real"],
        zero_division=0
    )
)

# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

print("\n--- Confusion Matrix ---")
print(cm)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Fake", "Real"],
    yticklabels=["Fake", "Real"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Multimodal Fake News Detection")
plt.tight_layout()

plt.savefig(
    "multimodal_confusion_matrix.png",
    dpi=300
)

plt.show()

print("\n--- Execution Completed Successfully ---")