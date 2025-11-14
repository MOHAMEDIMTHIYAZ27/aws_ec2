# test_evaluate.py

import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from evaluate import plot_confusion, plot_roc
from train_model import clean_text
from config import DATA_PATH, MODEL_PATH

import matplotlib.pyplot as plt


def main():
    # -----------------------------
    # 1️⃣ Load Data
    # -----------------------------
    print("🔹 Loading dataset...")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH) if DATA_PATH.endswith(".csv") else pd.read_excel(DATA_PATH)

    if "review" not in df.columns or "rating" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'rating' columns")

    # Clean text
    df["clean_review"] = df["review"].astype(str).apply(clean_text)

    # Map ratings → sentiment
    def map_sentiment(rating):
        if rating <= 2:
            return "Negative"
        elif rating == 3:
            return "Neutral"
        else:
            return "Positive"

    df["sentiment"] = df["rating"].apply(map_sentiment)

    X = df["clean_review"]
    y = df["sentiment"]

    # -----------------------------
    # 2️⃣ Load Combined Pipeline
    # -----------------------------
    print("🔹 Loading TF-IDF + Logistic Regression pipeline...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    pipeline = joblib.load(MODEL_PATH)

    # Extract model and vectorizer from pipeline
    try:
        vectorizer = pipeline.named_steps.get("vectorizer", None)
        model = pipeline.named_steps.get("classifier", None)
        if model is None or vectorizer is None:
            raise AttributeError("Pipeline missing vectorizer or classifier step.")
    except Exception:
        # fallback if pipeline is stored differently
        model = pipeline
        vectorizer = None

    # -----------------------------
    # 3️⃣ Split Data
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Vectorize text
    if vectorizer is not None:
        X_test_vec = vectorizer.transform(X_test)
    else:
        # If pipeline handles vectorization internally
        X_test_vec = X_test

    # -----------------------------
    # 4️⃣ Predictions & Evaluation
    # -----------------------------
    print("🔹 Generating predictions...")
    preds = pipeline.predict(X_test_vec)

    print("\n=== Classification Report ===")
    print(classification_report(y_test, preds, zero_division=0))

    # -----------------------------
    # 5️⃣ Plot Confusion Matrix
    # -----------------------------
    print("🔹 Plotting Confusion Matrix...")
    fig_cm = plot_confusion(y_test, preds, labels=["Positive", "Neutral", "Negative"])
    plt.show()

    # ------------
