import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# ---- FIXED IMPORTS ----
from src.config import DATA_PATH, MODEL_PATH
from src.utils import load_dataset, prepare_dataframe
from src.evaluate import plot_confusion


# ----------- Cached helpers -----------
@st.cache_data
def load_data(path=DATA_PATH):
    """Load the dataset once and cache it."""
    return load_dataset(path)


@st.cache_resource
def load_model(path=MODEL_PATH):
    """Load the trained model once and cache it."""
    return joblib.load(path)


# ----------- Main App -----------
def main():
    st.set_page_config(page_title="AI Echo — Sentiment Dashboard", layout="wide")
    st.title("AI Echo — Sentiment Analysis Dashboard")

    # Load dataset
    df = load_data()

    # Sidebar
    st.sidebar.header("Controls")
    show_raw = st.sidebar.checkbox("Show raw data", value=False)
    if show_raw:
        st.dataframe(df.head(200))

    # Preprocessing & EDA
    st.header("Preprocessing & Basic EDA")
    dfp = prepare_dataframe(df)

    # --- Ensure derived columns exist ---
    if "rating" in dfp.columns:
        dfp["sentiment_from_rating"] = dfp["rating"].apply(
            lambda x: "positive" if x >= 4 else ("negative" if x <= 2 else "neutral")
        )
    else:
        dfp["sentiment_from_rating"] = "unknown"

    if "review" in dfp.columns:
        dfp["clean_review"] = dfp["review"].str.lower()
    else:
        dfp["clean_review"] = ""

    # --- Layout ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating distribution")
        if "rating" in dfp.columns:
            fig, ax = plt.subplots()
            dfp["rating"].value_counts().sort_index().plot(kind="bar", ax=ax)
            ax.set_xlabel("Rating")
            ax.set_ylabel("Count")
            st.pyplot(fig)
        else:
            st.info("No rating column found")

    with col2:
        st.subheader("Sentiment from rating")
        if "sentiment_from_rating" in dfp.columns:
            st.write(dfp["sentiment_from_rating"].value_counts())
        else:
            st.info("No sentiment column found")

    # --- Model & Predictions ---
    st.header("Model & Predictions")
    model = None
    try:
        model = load_model()
        st.success("Loaded model successfully")
    except Exception:
        st.warning("Model not found. Please train the model first by running src/train_model.py")

    if model is not None:
        from sklearn.model_selection import train_test_split

        X = dfp["clean_review"]
        y = dfp["sentiment_from_rating"]

        mask = y != "unknown"
        X, y = X[mask], y[mask]

        if len(y.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            preds = model.predict(X_test)

            fig = plot_confusion(
                y_test, preds, labels=["positive", "neutral", "negative"]
            )
            st.subheader("Confusion Matrix (holdout)")
            st.pyplot(fig)

        # --- Single prediction ---
        st.subheader("Predict single review")
        text = st.text_area("Type a review to predict sentiment", height=120)
        if st.button("Predict"):
            if text.strip() == "":
                st.warning("Please enter review text.")
            else:
                pred = model.predict([text])[0]
                proba = model.predict_proba([text])[0]
                st.write("**Predicted sentiment:**", pred)

                probs = {
                    model.classes_[i]: round(proba[i], 3)
                    for i in range(len(model.classes_))
                }
                st.write("**Probabilities:**", probs)

    st.markdown("---")
    st.markdown(
        "**Notes:** This demo uses a TF-IDF + Logistic Regression pipeline trained on `rating` → `sentiment` mapping. "
        "For improved results, consider using transformer-based embeddings (e.g., BERT)."
    )


if __name__ == "__main__":
    main()
