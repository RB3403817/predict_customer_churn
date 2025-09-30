"""Data loading and preprocessing functions."""

import pandas as pd
from pathlib import Path


def load_data(csv_path: Path) -> pd.DataFrame:
    """
    Load the customer churn dataset from CSV.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with customer data
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df


def preprocess_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Convert target column from Yes/No to 1/0.
    
    Args:
        df: Input DataFrame
        target_col: Name of the target column
        
    Returns:
        DataFrame with binary target
    """
    df = df.copy()
    df[target_col] = df[target_col].map({'Yes': 1, 'No': 0}).astype(int)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional features
    """
    df = df.copy()
    
    # Example: total_payment = MonthlyCharges * tenure
    if set(['MonthlyCharges', 'tenure']).issubset(df.columns):
        df['total_payment'] = df['MonthlyCharges'] * df['tenure']
    
    return df
