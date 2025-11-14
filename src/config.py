import os

# ---------------------------
# Default Paths
# ---------------------------

# Dataset path (CSV version for consistency)
DATA_PATH = os.getenv(
    "AI_ECHO_DATA_PATH",
    os.path.join("data", "chatgpt_style_reviews_dataset.csv")
)

# Model path (must match train_model.py save location)
MODEL_PATH = os.getenv(
    "AI_ECHO_MODEL_PATH",
    os.path.join("models", "tfidf_logreg_pipeline.joblib")
)

# Random seed for reproducibility
RANDOM_SEED = 42
