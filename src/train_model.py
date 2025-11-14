import os
import re
import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

# ---------------------------
# Paths
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "chatgpt_style_reviews_dataset.csv")  # ✅ corrected filename
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "tfidf_logreg_pipeline.joblib")


# ---------------------------
# Preprocessing
# ---------------------------
def clean_text(text):
    """Lowercase, remove punctuation, special characters, and numbers."""
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"[^a-z\s]", "", text)  # keep only letters
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(df):
    """Prepare dataset with cleaned reviews and sentiment labels."""
    # Ensure required columns exist
    if "review" not in df.columns or "rating" not in df.columns:
        raise ValueError("Dataset must contain 'review' and 'rating' columns")

    df["clean_review"] = df["review"].astype(str).apply(clean_text)

    # Map ratings (1–5) → sentiment
    def map_sentiment(rating):
        if rating <= 2:
            return "negative"
        elif rating == 3:
            return "neutral"
        else:
            return "positive"

    df["sentiment"] = df["rating"].apply(map_sentiment)
    return df


# ---------------------------
# Train Model
# ---------------------------
def train_and_save():
    df = pd.read_csv(DATA_PATH)  # ✅ load CSV instead of Excel
    df = preprocess(df)

    X = df["clean_review"]
    y = df["sentiment"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline (Vectorizer + Classifier)
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("logreg", LogisticRegression(max_iter=1000)),
    ])

    # Train pipeline
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    # Save pipeline (model + vectorizer inside one object)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"✅ Pipeline saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_and_save()
