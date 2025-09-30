"""Feature preprocessing pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer


def create_preprocessor(numeric_features, categorical_features):
    """
    Create preprocessing pipeline for numeric and categorical features.
    
    Args:
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        
    Returns:
        ColumnTransformer for preprocessing
    """
    numeric_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor


def get_feature_types(X):
    """
    Identify numeric and categorical features.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple of (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    
    return numeric_features, categorical_features
