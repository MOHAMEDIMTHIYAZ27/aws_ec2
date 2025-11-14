# This file makes the 'src' directory a Python package.

# Optional: expose commonly used modules
from .config import DATA_PATH, MODEL_PATH
from .utils import load_dataset, prepare_dataframe
from .evaluate import plot_confusion
