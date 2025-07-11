# train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("🔄 Loading and preprocessing data...")

# Load datasets
true_df = pd.read_csv("datasets/True.csv")
fake_df = pd.read_csv("datasets/Fake.csv")

# Add labels
true_df['label'] = 'REAL'
fake_df['label'] = 'FAKE'

# Merge datasets
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1).reset_index(drop=True)  # Shuffle

# Use only title + text
df['content'] = df['title'] + " " + df['text']

# Split features and labels
X = df['content']
y = df['label']

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_tfidf = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {acc * 100:.2f}%")

# Save models
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/fake_news_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=["FAKE", "REAL"])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["FAKE", "REAL"], yticklabels=["FAKE", "REAL"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
os.makedirs("outputs", exist_ok=True)
plt.savefig("outputs/confusion_matrix.png")
plt.close()
print("📊 Confusion matrix saved to outputs/confusion_matrix.png")
