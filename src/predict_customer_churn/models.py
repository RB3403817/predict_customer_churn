"""Model training and evaluation functions."""

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
import optuna


def create_model_pipeline(preprocessor, model):
    """
    Create a pipeline with preprocessor and model.
    
    Args:
        preprocessor: Preprocessing transformer
        model: ML model
        
    Returns:
        Pipeline object
    """
    return Pipeline(steps=[
        ('prep', preprocessor),
        ('model', model)
    ])


def train_baseline_models(X_train, X_test, y_train, y_test, preprocessor, random_state=42):
    """
    Train baseline models and return results.
    
    Args:
        X_train, X_test: Training and test features
        y_train, y_test: Training and test targets
        preprocessor: Preprocessing pipeline
        random_state: Random seed
        
    Returns:
        Dictionary with model results
    """
    results = {}
    
    # XGBoost
    xgb_clf = create_model_pipeline(
        preprocessor,
        XGBClassifier(random_state=random_state, eval_metric='logloss')
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)
    y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
    results['XGBoost'] = {
        'roc_auc': roc_auc_score(y_test, y_proba_xgb),
        'accuracy': (y_pred_xgb == y_test).mean(),
        'model': xgb_clf
    }
    
    # LightGBM
    lgbm_clf = create_model_pipeline(
        preprocessor,
        LGBMClassifier(random_state=random_state, verbose=-1)
    )
    lgbm_clf.fit(X_train, y_train)
    y_pred_lgbm = lgbm_clf.predict(X_test)
    y_proba_lgbm = lgbm_clf.predict_proba(X_test)[:, 1]
    results['LightGBM'] = {
        'roc_auc': roc_auc_score(y_test, y_proba_lgbm),
        'accuracy': (y_pred_lgbm == y_test).mean(),
        'model': lgbm_clf
    }
    
    # CatBoost
    cat_clf = create_model_pipeline(
        preprocessor,
        CatBoostClassifier(random_state=random_state, verbose=0)
    )
    cat_clf.fit(X_train, y_train)
    y_pred_cat = cat_clf.predict(X_test)
    y_proba_cat = cat_clf.predict_proba(X_test)[:, 1]
    results['CatBoost'] = {
        'roc_auc': roc_auc_score(y_test, y_proba_cat),
        'accuracy': (y_pred_cat == y_test).mean(),
        'model': cat_clf
    }
    
    # AdaBoost
    ada_clf = create_model_pipeline(
        preprocessor,
        AdaBoostClassifier(random_state=random_state, algorithm='SAMME')
    )
    ada_clf.fit(X_train, y_train)
    y_pred_ada = ada_clf.predict(X_test)
    y_proba_ada = ada_clf.predict_proba(X_test)[:, 1]
    results['AdaBoost'] = {
        'roc_auc': roc_auc_score(y_test, y_proba_ada),
        'accuracy': (y_pred_ada == y_test).mean(),
        'model': ada_clf
    }
    
    return results


def cv_auc(model, X, y, cv=5):
    """
    Calculate cross-validated ROC-AUC score.
    
    Args:
        model: Model pipeline
        X: Features
        y: Target
        cv: Number of cross-validation folds
        
    Returns:
        Mean CV ROC-AUC score
    """
    cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv_obj, scoring='roc_auc')
    return scores.mean()


def tune_xgboost(X_train, y_train, preprocessor, n_trials=50, random_state=42):
    """
    Tune XGBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Optuna study object
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
            'eval_metric': 'logloss'
        }
        model = create_model_pipeline(preprocessor, XGBClassifier(**params))
        score = cv_auc(model, X_train, y_train)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='xgboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def tune_lightgbm(X_train, y_train, preprocessor, n_trials=30, random_state=42):
    """
    Tune LightGBM hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Optuna study object
    """
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': random_state,
            'verbose': -1
        }
        model = create_model_pipeline(preprocessor, LGBMClassifier(**params))
        score = cv_auc(model, X_train, y_train)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='lightgbm_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study


def tune_catboost(X_train, y_train, preprocessor, n_trials=30, random_state=42):
    """
    Tune CatBoost hyperparameters using Optuna.
    
    Args:
        X_train: Training features
        y_train: Training target
        preprocessor: Preprocessing pipeline
        n_trials: Number of Optuna trials
        random_state: Random seed
        
    Returns:
        Optuna study object
    """
    def objective(trial):
        params = {
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'iterations': trial.suggest_int('iterations', 50, 300),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'random_state': random_state,
            'verbose': 0
        }
        model = create_model_pipeline(preprocessor, CatBoostClassifier(**params))
        score = cv_auc(model, X_train, y_train)
        return score
    
    study = optuna.create_study(direction='maximize', study_name='catboost_tuning')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study
