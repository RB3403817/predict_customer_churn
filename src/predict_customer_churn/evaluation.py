"""Evaluation and visualization functions."""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve
)


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model pipeline
        X_test: Test features
        y_test: Test target
        model_name: Name for display
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    results = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'accuracy': (y_pred == y_test).mean(),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    print(f"\n=== {model_name} Evaluation ===")
    print(f"ROC-AUC: {results['roc_auc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"\nConfusion Matrix:\n{results['confusion_matrix']}")
    print(f"\nClassification Report:\n{results['classification_report']}")
    
    return results


def plot_roc_curve(y_test, y_proba, model_name='Model'):
    """
    Plot ROC curve.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        model_name: Name for display
    """
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def plot_precision_recall_curve(y_test, y_proba, model_name='Model'):
    """
    Plot Precision-Recall curve.
    
    Args:
        y_test: True labels
        y_proba: Predicted probabilities
        model_name: Name for display
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=model_name)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt


def create_results_summary(results):
    """
    Create a summary DataFrame from model results.
    
    Args:
        results: Dictionary of model results
        
    Returns:
        DataFrame with sorted results
    """
    summary_data = {
        name: {'roc_auc': info['roc_auc'], 'accuracy': info['accuracy']}
        for name, info in results.items()
    }
    
    summary_df = pd.DataFrame(summary_data).T.sort_values('roc_auc', ascending=False)
    return summary_df
