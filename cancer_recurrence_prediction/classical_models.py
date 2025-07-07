# cancer_recurrence_prediction/classical_models.py

"""
Classical ML models for binary classification of cancer recurrence.

Supports the following models:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier

Provides functions to train models and evaluate them using standard metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, precision_recall_curve

from . import config

def get_model(name):
    """
    Retrieve a model by name.

    Args:
        name (str): Name of the model ("logistic", "svm", "random_forest", "xgboost").

    Returns:
        A scikit-learn compatible classifier.
    """
    name = name.lower()
    if name == "logistic":
        return LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE)
    elif name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=config.RANDOM_STATE)
    elif name == "random_forest":
        return RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_STATE)
    else:
        raise ValueError(f"Unsupported model name: {name}")

def train_model(model_name, X_train, y_train):
    """
    Train a classifier on the training data.

    Args:
        model_name (str): Model name string.
        X_train (array-like): Training features.
        y_train (array-like): Training labels.

    Returns:
        Trained model object.
    """
    model = get_model(model_name)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, generate_curve_data = False):
    """
    Evaluate a trained model on test data.

    Args:
        model: Trained classifier.
        X_test (array-like): Test features.
        y_test (array-like): Test labels.
        generate_curve_data (Boolean): Option to generate roc and precision-recall curve data. (default: False)
        
    Returns:
        Dict: Metrics including accuracy, precision, recall, f1, and curve data (optional)
    """
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }
    
    if generate_curve_data:     
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        metrics["roc_curve"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": roc_auc}
    
        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        metrics["pr_curve"] = {"precision": precision.tolist(), "recall": recall.tolist(), "auc": pr_auc}

    return metrics


def benchmark_models(X_train, y_train, X_test, y_test, models = None, generate_curve_data = False):
    """
    Train and evaluate multiple classical models for benchmarking.

    Args:
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        models (list): List of model names to evaluate. (default = from config)
        generate_curve_data (Boolean): Option to generate roc and precision-recall curve data. (default: False)
        
    Returns:
        Dict[str, Dict]: Dictionary mapping model names to evaluation metrics.
    """
    if models is None:
        models = config.CLASSICAL_MODELS

    results = {}
    for name in models:
        try:
            clf = train_model(name, X_train, y_train)
            results[name] = evaluate_model(clf, X_test, y_test, generate_curve_data = generate_curve_data)
        except Exception as e:
            results[name] = {"error": str(e)}

    return results