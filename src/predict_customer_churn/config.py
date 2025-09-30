"""Configuration and constants for the churn prediction pipeline."""

from pathlib import Path

# Random state for reproducibility
RANDOM_STATE = 42

# Data paths
DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
CSV_PATH = RAW_DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

# Model parameters
TEST_SIZE = 0.2
CV_SPLITS = 5

# Target column
TARGET_COL = "Churn"

# Optuna settings
N_TRIALS_XGB = 50
N_TRIALS_LGBM = 30
N_TRIALS_CAT = 30
