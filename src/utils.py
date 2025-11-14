import pandas as pd
import os

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from a given path.
    Supports CSV and Excel files (.xlsx).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at path: {path}")

    if path.endswith(".csv"):
        return pd.read_csv(path)
    elif path.endswith(".xlsx"):
        return pd.read_excel(path, sheet_name=0)  # Default: first sheet
    else:
        raise ValueError(f"Unsupported file format: {path}")


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning / preprocessing for the Streamlit app.
    Example: handle missing reviews and normalize column names.
    """
    df = df.copy()

    # normalize column names (lowercase, strip spaces)
    df.columns = df.columns.str.strip().str.lower()

    # fill missing review text if column exists
    if "review" in df.columns:
        df["review"] = df["review"].fillna("").astype(str)

    # fill missing rating if column exists
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).astype(int)

    return df
