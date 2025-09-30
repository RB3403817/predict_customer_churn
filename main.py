"""Main script to run the customer churn prediction pipeline."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from predict_customer_churn.config import (
    CSV_PATH, TARGET_COL, RANDOM_STATE, TEST_SIZE
)
from predict_customer_churn.data import (
    load_data, preprocess_target, engineer_features
)
from predict_customer_churn.preprocessing import (
    get_feature_types, create_preprocessor
)
from predict_customer_churn.models import (
    train_baseline_models, tune_xgboost
)
from predict_customer_churn.evaluation import (
    evaluate_model, create_results_summary
)
from sklearn.model_selection import train_test_split


def main():
    """Run the full churn prediction pipeline."""
    print("=" * 60)
    print("Customer Churn Prediction Pipeline")
    print("=" * 60)
    
    # 1. Load data
    print("\n1. Loading data...")
    df = load_data(CSV_PATH)
    print(f"   Loaded {len(df)} rows and {len(df.columns)} columns")
    
    # 2. Preprocess target
    print("\n2. Preprocessing target column...")
    df = preprocess_target(df, TARGET_COL)
    
    # 3. Feature engineering
    print("\n3. Engineering features...")
    df = engineer_features(df)
    
    # 4. Prepare features and target
    print("\n4. Preparing features and target...")
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # 5. Train/test split
    print("\n5. Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"   Training set: {len(X_train)} samples")
    print(f"   Test set: {len(X_test)} samples")
    
    # 6. Create preprocessor
    print("\n6. Creating preprocessing pipeline...")
    numeric_features, categorical_features = get_feature_types(X)
    print(f"   Numeric features: {len(numeric_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    preprocessor = create_preprocessor(numeric_features, categorical_features)
    
    # 7. Train baseline models
    print("\n7. Training baseline models...")
    results = train_baseline_models(
        X_train, X_test, y_train, y_test, preprocessor, RANDOM_STATE
    )
    
    # 8. Display results
    print("\n8. Baseline Model Results:")
    summary = create_results_summary(results)
    print(summary)
    
    # 9. Hyperparameter tuning (optional - commented out for faster execution)
    # print("\n9. Hyperparameter tuning with Optuna...")
    # study = tune_xgboost(X_train, y_train, preprocessor, n_trials=20)
    # print(f"   Best ROC-AUC: {study.best_value:.4f}")
    # print(f"   Best parameters: {study.best_params}")
    
    print("\n" + "=" * 60)
    print("Pipeline completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
